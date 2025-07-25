import triton
import triton.language as tl
import torch
import math

@triton.jit
def flash_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, 
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vv, stride_vd, 
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    scale_val = tl.load(scale)
    tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vv, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE, D)
    if is_causal:
        q_mask = tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]
        k_mask = tl.arange(0, K_TILE_SIZE)[None, :]

    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.full((Q_TILE_SIZE,), float(0), dtype=tl.float32)
    O_i = tl.full((Q_TILE_SIZE, D), float(0), dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        kt_tile = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero") # (D, K_TILE_SIZE)
        v_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero") # (K_TILE_SIZE, D)
        prev_m = m

        S = tl.dot(q_tile, kt_tile) / scale_val # (Q_TILE_SIZE, K_TILE_SIZE)

        if is_causal:
            mask = (k_mask + j * K_TILE_SIZE) > q_mask
            S += mask * -1e6

        m = tl.maximum(m, tl.max(S, axis=-1)) # (Q_TILE_SIZE,)
        P = tl.exp(S - m[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)
        l = tl.exp(prev_m - m) * l + tl.sum(P, axis=-1) # (Q_TILE_SIZE,)
        O_i = tl.exp(prev_m - m)[:, None] * O_i + tl.dot(P.to(v_tile.dtype), v_tile) # (Q_TILE_SIZE, D)

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l[:, None]
    L_i = m + tl.log(l)
    tl.store(O_block_ptr, O_i, boundary_check=(0,))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))

@torch.compile
def flash_backward(Q, K, V, O, L, d, dO, is_causal):
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    S = Q @ K.transpose(-1, -2) / math.sqrt(d)
    P = torch.exp(S - L.unsqueeze(-1))

    if is_causal:
        P = torch.tril(P)

    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    dS = P * (dP - D)
    dQ = dS @ K / math.sqrt(d)
    dK = dS.transpose(-1, -2) @ Q / math.sqrt(d)

    return dQ, dK, dV, None

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q, B_k = 16, 16
        B, N_q, D = Q.shape
        _, N_k, _ = K.shape

        stride_qb, stride_qq, stride_qd = N_q * D, D, 1
        stride_kb, stride_kk, stride_kd = N_k * D, D, 1
        stride_vb, stride_vv, stride_vd = N_k * D, D, 1
        stride_ob, stride_oq, stride_od = N_q * D, D, 1
        stride_lb, stride_lq = N_q, 1

        scale = torch.tensor(D, device=torch.device("cuda")).sqrt().to(torch.float32)

        O = torch.zeros(B, N_q, D, dtype=torch.float32, device=torch.device("cuda"))
        L = torch.zeros(B, N_q, dtype=torch.float32, device=torch.device("cuda"))
        flash_forward_kernel[(triton.cdiv(N_q, B_q), B)](Q, K, V, O, L,
                                                       stride_qb, stride_qq, stride_qd,
                                                       stride_kb, stride_kk, stride_kd,
                                                       stride_vb, stride_vv, stride_vd,
                                                       stride_ob, stride_oq, stride_od, 
                                                       stride_lb, stride_lq, N_q, N_k,
                                                       scale, D, B_q, B_k, is_causal)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.d = D
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        d = ctx.d
        is_causal = ctx.is_causal
        return flash_backward(Q, K, V, O, L, d, dO, is_causal)
