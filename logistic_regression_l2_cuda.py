import time

import numpy as np
from logistic_regression_l2 import LogisticRegressionL2
from numba import cuda, float64
import math


@cuda.jit(device=True)
def warp_reduce_sum(val):
    mask = cuda.warpsize // 2
    while mask > 0:
        val += cuda.shfl_xor_sync(0xFFFFFFFF, val, mask)
        mask //= 2
    return val


@cuda.jit(device=True)
def block_reduce_sum(val):
    warp_size = cuda.warpsize
    num_warps = cuda.blockDim.x // warp_size
    shared = cuda.shared.array(shape=32, dtype=float64)  # max 32 warp

    lane = cuda.threadIdx.x % warp_size
    wid = cuda.threadIdx.x // warp_size

    val = warp_reduce_sum(val)

    if lane == 0 and wid < num_warps:
        shared[wid] = val
    cuda.syncthreads()

    val = 0.0
    if wid == 0 and lane < num_warps:
        val = shared[lane]
    val = warp_reduce_sum(val)
    return val


@cuda.jit(fastmath=True)
def reduce_sum_kernel(input_array, output_array):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    idx = bid * bdim + tid

    val = 0.0
    if idx < input_array.size:
        val = input_array[idx]

    val = block_reduce_sum(val)

    if tid == 0:
        output_array[bid] = val


@cuda.jit(fastmath=True)
def compute_loss_cuda_kernel_global(X_flat, y, w, loss_terms, n_samples, n_features):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    grid_size = cuda.gridDim.x

    for i in range(bx, n_samples, grid_size):
        partial_dot = 0.0
        for j in range(tx, n_features, bw):
            index = i * n_features + j
            partial_dot += X_flat[index] * w[j]

        dot = block_reduce_sum(partial_dot)

        if tx == 0:
            y_i = y[i]
            z = -y_i * dot
            if z > 0:
                loss = z + math.log1p(math.exp(-z))
            else:
                loss = math.log1p(math.exp(z))
            loss_terms[i] = loss


@cuda.jit(fastmath=True)
def reduce_l2_kernel(w, reg_terms):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    idx = bid * bdim + tid

    val = 0.0
    if idx < w.size:
        wi = w[idx]
        val = wi * wi  # L2 term

    val = block_reduce_sum(val)

    if tid == 0:
        reg_terms[bid] = val

@cuda.jit(device=True, inline=True)
def sigmoid(z):

    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

@cuda.jit(fastmath=True)
def compute_r_kernel(X_flat, y, w, r, n_samples, n_features):
    i = cuda.grid(1)
    if i < n_samples:
        dot = 0.0
        base = i * n_features

        for j in range(n_features):
            dot += X_flat[base + j] * w[j]

        yi = y[i]
        z = -yi * dot
        r[i] = -yi * sigmoid(z)

@cuda.jit(fastmath=True)
def compute_XTr_kernel(X_flat, r, grad, n_samples, n_features):
    j = cuda.grid(1)
    if j < n_features:
        acc = 0.0
        # somma su i: acc += X[i, j] * r[i]
        for i in range(n_samples):
            acc += X_flat[i * n_features + j] * r[i]
        grad[j] = acc


class LogisticRegressionL2Cuda(LogisticRegressionL2):
    def __init__(self, lambda_reg=0.1, TPB=256):
        self.lambda_reg = lambda_reg
        self.TPB = TPB

        # cache GPU
        self.X_gpu = None
        self.X_flat_gpu = None
        self.y_gpu = None
        self.w_gpu = None
        self.r_gpu = None
        self.XTr_gpu = None

        self.warmup_all()

    def warmup_all(self, N=128, D=500):
        X_dummy = np.random.randn(N, D).astype(np.float32)
        y_dummy = np.random.choice([-1, 1], size=N).astype(np.float32)
        w_dummy = np.random.randn(D).astype(np.float64)

        Xf_gpu = cuda.to_device(X_dummy.reshape(-1))
        y_gpu = cuda.to_device(y_dummy)
        w_gpu = cuda.to_device(w_dummy)
        loss_terms = cuda.device_array(N, dtype=np.float64)

        blocks = (N + self.TPB - 1) // self.TPB
        compute_loss_cuda_kernel_global[blocks, self.TPB](
            Xf_gpu, y_gpu, w_gpu, loss_terms, N, D
        )
        reduce_l2_kernel[blocks, self.TPB](w_gpu, loss_terms)

        r_gpu = cuda.device_array(N, dtype=np.float64)
        grad_gpu = cuda.device_array(D, dtype=np.float64)

        blocks_r = (N + self.TPB - 1) // self.TPB
        blocks_f = (D + self.TPB - 1) // self.TPB

        compute_r_kernel[blocks_r, self.TPB](
            Xf_gpu, y_gpu, w_gpu, r_gpu, N, D
        )
        compute_XTr_kernel[blocks_f, self.TPB](
            Xf_gpu, r_gpu, grad_gpu, N, D
        )

        cuda.synchronize()

    def cache_data(self, X, y):
        if self.X_gpu is None or self.X_gpu.shape != X.shape:
            self.X_gpu = cuda.to_device(X)
            self.X_flat_gpu = cuda.to_device(X.reshape(-1))
        if self.y_gpu is None or self.y_gpu.shape != y.shape:
            self.y_gpu = cuda.to_device(y)

    def cache_buffers(self, n_samples, n_features):
        if self.r_gpu is None or self.r_gpu.size != n_samples:
            self.r_gpu = cuda.device_array(n_samples, dtype=np.float64)
        if self.XTr_gpu is None or self.XTr_gpu.size != n_features:
            self.XTr_gpu = cuda.device_array(n_features, dtype=np.float64)

    def gpu_sum(self, d_array):
        length = d_array.size
        while length > 1:
            blocks = (length + self.TPB - 1) // self.TPB
            d_out = cuda.device_array(shape=blocks, dtype=d_array.dtype)
            reduce_sum_kernel[blocks, self.TPB](d_array, d_out)
            d_array = d_out
            length = blocks
        return d_array.copy_to_host()[0]

    def compute_loss(self, X, y, w):
        start = time.time()
        n_samples, n_features = X.shape
        self.w_gpu = cuda.to_device(w)

        loss_terms_gpu = cuda.device_array(n_samples, dtype=np.float64)

        reg_blocks = (w.size + self.TPB - 1) // self.TPB
        reg_terms_gpu = cuda.device_array(reg_blocks, dtype=np.float64)

        blocks = (n_samples + self.TPB - 1) // self.TPB

        compute_loss_cuda_kernel_global[blocks, self.TPB](
            self.X_flat_gpu, self.y_gpu, self.w_gpu, loss_terms_gpu, n_samples, n_features
        )

        reduce_l2_kernel[reg_blocks, self.TPB](self.w_gpu, reg_terms_gpu)
        cuda.synchronize()

        total_loss = self.gpu_sum(loss_terms_gpu)
        reg_term = (self.lambda_reg / 2.0) * self.gpu_sum(reg_terms_gpu)
        loss = total_loss + reg_term

        elapsed = time.time() - start
        self.loss_time += elapsed
        self.total_time += elapsed
        return loss

    def compute_loss_gradient(self, X, y, w):
        start = time.time()
        n_samples, n_features = X.shape

        d_w = cuda.to_device(w.astype(np.float64))
        d_r = cuda.device_array(n_samples, dtype=np.float64)
        d_grad = cuda.device_array(n_features, dtype=np.float64)

        threads = self.TPB
        blocks_r = (n_samples + threads - 1) // threads
        blocks_f = (n_features + threads - 1) // threads

        compute_r_kernel[blocks_r, threads](self.X_flat_gpu, self.y_gpu, d_w, d_r, n_samples, n_features)
        compute_XTr_kernel[blocks_f, threads](self.X_flat_gpu, d_r, d_grad, n_samples, n_features)

        cuda.synchronize()
        grad = d_grad.copy_to_host() + self.lambda_reg * w

        elapsed = time.time() - start
        self.gradient_time += elapsed
        self.total_time += elapsed
        return grad
