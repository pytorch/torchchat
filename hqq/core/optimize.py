# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2023
#####################################################
import torch
from torch import Tensor
import numpy as np


# re-estimate the scale based on the inverse median
def update_scale_inverse_median(
    W_f: Tensor, scale: Tensor, zero: Tensor, axis: int, min_max: list
) -> tuple:
    scale_rng = 2e4
    z_val = 1e-4
    delta = 1e-2

    W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])

    # Correct zero to avoid W_q==zero
    zero_c = zero.clone()
    zero_c_indx = torch.sum(1.0 * ((W_q - zero) == 0), axis=axis, keepdim=True) > 0
    zero_c[zero_c_indx] = zero_c[zero_c_indx] + delta

    # Build scale tensor
    W_f_c = W_f.clone()
    W_f_c_mask = torch.abs(W_f_c) < z_val
    W_f_c[W_f_c_mask] = z_val

    scale_tensor = (W_q - zero_c).float() / W_f_c.float()
    # W_r               = (W_q - zero_c)/scale_tensor

    # Normalize scale_tensor
    scale_b = torch.median(scale_tensor, axis=axis, keepdim=True)[0]
    scale_b = scale_b.clamp(min=-scale_rng, max=scale_rng).half()

    # Mix with older scale
    W_r = (W_q - zero_c) / scale_b
    err_b = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    W_r = (W_q - zero_c) / scale
    err_a = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    mask = (err_b < err_a).half()
    scale_b = mask * scale_b + (1 - mask) * scale

    # W_r   = (W_q - zero_c)/scale_b
    return scale_b, zero_c


# Greedy local search
def update_scale_grid_search(
    W_f: Tensor, scale: Tensor, zero: Tensor, axis: int, min_max: list, N: int = 128 + 1
) -> Tensor:
    # Make sure it's an odd number so that the original scale is included
    assert N % 2 == 1, "Please check whether N: odd number"
    rng_dump = 0.05  # 0.05 / 1.
    z_val = 2e-4

    device = scale.device
    dtype = scale.dtype
    ###############################
    W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
    n_clusters = max(W_q.shape[0], W_q.shape[1])
    rng = torch.abs(scale).mean() * rng_dump if (rng_dump < 1.0) else rng_dump

    scale_shifted = (
        torch.linspace(-rng, rng, N)[:, None].to(dtype).to(device).repeat(1, n_clusters)
        + scale
    )

    # Safe inverse
    scale_shifted[
        torch.logical_and(scale_shifted >= 0, torch.abs(scale_shifted) <= z_val)
    ] = z_val
    scale_shifted[
        torch.logical_and(scale_shifted < 0, torch.abs(scale_shifted) <= z_val)
    ] = -z_val

    err = torch.empty([N, n_clusters], dtype=dtype, device=device)
    for i in range(N):
        W_r = (W_q - zero) / scale_shifted[i][None, :]
        err[i] = torch.abs(W_f - W_r).mean(axis=axis, keepdim=True)

    ind_r = torch.argmin(err, axis=axis).to(torch.int32)
    ind_c = torch.arange(len(ind_r), dtype=torch.int32, device=device)
    scale_b = scale_shifted[ind_r, ind_c]

    return scale_b


# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


# Proximal solver || W - dequantize(quantize(W))||_p^p - Experimental
@torch.inference_mode()
def optimize_weights_proximal_v2(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: str = "cuda",
    opt_params: dict = {
        "lp_norm": 0.7,
        "beta": 1e1,
        "kappa": 1.01,
        "iters": 20,
        "tol": 0.0,
        "early_stop": True,
        "scale_gridsearch": False,
    },
    verbose: bool = False,
) -> tuple:
    # Params
    lp_norm = max(opt_params["lp_norm"], 0.1)
    beta = opt_params["beta"]
    kappa = opt_params["kappa"]
    iters = opt_params["iters"]
    early_stop = opt_params["early_stop"]
    tol = opt_params["tol"]

    # Check
    assert lp_norm <= 1.0, "lp_norm should be <=1"
    assert beta > 0.0, "beta should be > 0"
    assert kappa > 1.0, "kappa should be > 1"
    assert iters > 1, "iters should be > 1"

    # Cast/device
    device = torch.device(device)
    dtype = torch.float16 if (device.type == "cuda") else torch.float32
    W_f = tensor.to(dtype).to(device)
    scale = scale.to(dtype).to(device)
    zero = zero.to(dtype).to(device)

    # Update scale: works slightly better. Tested on Llama2 only
    if opt_params["scale_gridsearch"]:
        scale = update_scale_grid_search(W_f, scale, zero, axis, min_max)

    # Optimize for zero-point
    best_error = 1e4
    scale_prev, zero_prev = scale.clone(), zero.clone()
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale

        # current_error = float(torch.pow(torch.abs(W_f - W_r), max(0.80, lp_norm)).mean())
        current_error = float(torch.abs(W_f - W_r).mean())

        if verbose:
            print(i, np.round(current_error, 6))

        if early_stop:
            if best_error - current_error > tol:
                best_error = current_error
                scale_prev, zero_prev = scale.clone(), zero.clone()
            else:
                scale, zero = scale_prev.clone(), zero_prev.clone()
                break

        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

    # Clean-up
    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e, scale_prev, zero_prev
    torch.cuda.empty_cache()

    return scale, zero


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: str = "cuda",
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose: bool = False,
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    device = torch.device(device)
    dtype = torch.float16 if (device.type == "cuda") else torch.float32
    W_f = tensor.to(dtype).to(device)
    scale = scale.to(dtype).to(device)
    zero = zero.to(dtype).to(device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print(i, np.round(current_error, 6))
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    torch.cuda.empty_cache()

    return scale, zero


optimize_weights_proximal = optimize_weights_proximal_legacy
# optimize_weights_proximal = optimize_weights_proximal_v2


# SGD solver  || W - dequantize(quantize(W))||_1 (p=1 only)
def optimize_weights_autograd(
    tensor: Tensor,
    scale: Tensor,
    zero: Tensor,
    min_max: list,
    axis: int = 0,
    device: str = "cuda",
    opt_params: dict = {"lr": 2e-3, "iters": 2500},
    verbose: bool = False,
) -> tuple:
    W_f = tensor.to(device)
    params = {}
    params["scale"] = torch.nn.Parameter(scale.float().to(device), requires_grad=True)
    params["zero"] = torch.nn.Parameter(zero.float().to(device), requires_grad=True)
    optimizer = torch.optim.AdamW(
        [params[k] for k in params],
        lr=opt_params["lr"],
        betas=(0.9, 0.99),
        eps=1e-06,
        weight_decay=0.0,
    )

    def _loss_fct(output, target):
        return torch.mean(torch.abs(target - output))  # L1

    def _fake_quant(W_f):
        # Quantize
        W_q = torch.round(W_f * params["scale"] + params["zero"]).clamp(
            min_max[0], min_max[1]
        )
        # Dequantize
        W_r = (W_q - params["zero"]) / params["scale"]
        return W_r

    def _step(W_f):
        optimizer.zero_grad()
        loss = _loss_fct(_fake_quant(W_f), W_f)
        loss.backward()
        optimizer.step()
        return np.round(loss.item(), 10)

    with torch.no_grad():
        _init_loss = _loss_fct(_fake_quant(W_f), W_f).item()

    for i in range(opt_params["iters"]):
        loss_out = _step(W_f)
        if verbose and (i % 100) == 0:
            print(i, loss_out)

    with torch.no_grad():
        _final_loss = _loss_fct(_fake_quant(W_f), W_f).item()

    if _final_loss < _init_loss:
        for k in params:
            params[k] = params[k].data.detach().to(tensor.device)
    else:
        if verbose:
            print("optimization failed...")
        params = {"scale": scale, "zero": zero}

    del W_f
    torch.cuda.empty_cache()

    return params["scale"], params["zero"]
