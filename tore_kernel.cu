// tore_kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <cmath>

using i32 = int32_t;
using i64 = int64_t;
using u64 = uint64_t;

#define CUDA_CHECK(stmt) AT_CUDA_CHECK(stmt)
#define LAUNCH_CHECK()   AT_CUDA_CHECK(cudaGetLastError())

// 从 key 提取 pi / t
struct PiOfKey {
    int shift;
    __host__ __device__ i32 operator()(const u64& k) const {
        return static_cast<i32>(k >> shift);
    }
};
struct TOfKey {
    u64 mask;
    __host__ __device__ i32 operator()(const u64& k) const {
        return static_cast<i32>(k & mask);
    }
};

// pack (pi<<SHIFT)|t
__global__ void pack_keys_kernel(
    const i32* __restrict__ xs,
    const i32* __restrict__ ys,
    const i32* __restrict__ ts,
    const i32* __restrict__ ps, // 0/1
    int H, int W, u64* __restrict__ keys, int N, int SHIFT
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    i32 x = xs[i], y = ys[i], t = ts[i], p = ps[i];
    i32 lin = y * W + x;                 // [0, HW-1]
    i32 HW  = H * W;
    u64 pi  = static_cast<u64>(p * HW + lin);
    u64 key = (pi << SHIFT) | static_cast<u64>(t);
    keys[i] = key;
}

// 将选择的 (pi,t,k_sel) 写入 ts_buf[2,H,W,K]
__global__ void scatter_selected_kernel(
    const u64* __restrict__ keys, // sorted ascending
    const i32* __restrict__ rank_from_end, // 1..group_len
    int N, int H, int W, int K, int SHIFT, u64 t_mask,
    int* __restrict__ ts_buf // (2*H*W*K), init -1
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    i32 r = rank_from_end[i] - 1; // 0..group_len-1
    if (r >= K) return;

    u64 key = keys[i];
    u64 pi  = key >> SHIFT;
    i32 t   = static_cast<i32>(key & t_mask);

    i32 HW = H * W;
    i32 p  = static_cast<i32>(pi / HW);
    i32 lin= static_cast<i32>(pi % HW);
    i32 y  = lin / W;
    i32 x  = lin % W;

    // ts_buf layout: [2,H,W,K] -> index = (((p*H + y)*W + x)*K + r)
    i64 idx = (((static_cast<i64>(p) * H + y) * W + x) * K + r);
    ts_buf[idx] = t; // 每个 (p,y,x,r) 唯一，无需原子
}

// 计算 dt/log1p 并输出
template<typename scalar_t>
__global__ void make_tore_kernel(
    const int* __restrict__ ts_buf, // [2,H,W,K], int32
    int H, int W, int K,
    float t_query, float t0, float tmax,
    bool out_chw, // true:(2K,H,W) ; false:(2,K,H,W)
    scalar_t* __restrict__ out
){
    int P = 2;
    i64 HW = (i64)H * W;
    i64 total = (i64)P * HW * K;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int k = i % K;
    i64 tmp = i / K;   // [0..P*HW-1]
    int xy = (int)(tmp % HW);
    int p  = (int)(tmp / HW);

    int y = xy / W;
    int x = xy % W;

    i64 idx_ts = (((i64)p * H + y) * W + x) * K + k;
    int ts = ts_buf[idx_ts];

    float dt = (ts < 0) ? tmax : (t_query - (float)ts);
    if (dt < t0)   dt = t0;
    if (dt > tmax) dt = tmax;
    float val = log1pf(dt);

    if (out_chw) {
        // (2K,H,W) : c = p*K + k
        int c = p * K + k;
        i64 out_idx = ((i64)c * H + y) * W + x;
        out[out_idx] = static_cast<scalar_t>(val);
    } else {
        // (2,K,H,W)
        i64 out_idx = ((((i64)p * K + k) * H + y) * W + x);
        out[out_idx] = static_cast<scalar_t>(val);
    }
}

static int decide_shift_from_tmax(int max_t){
    int SHIFT = 24;
    if (max_t >= (1<<SHIFT)) {
        // ceil(log2(max_t+1))
        int s = 0;
        int v = max_t + 1;
        while (v > 0) { v >>= 1; ++s; }
        SHIFT = s;
        if (SHIFT > 30) SHIFT = 30; // 安全上限（t 用 int32）
    }
    return SHIFT;
}

template<typename scalar_t>
at::Tensor launch_make_tore(
    const at::Tensor& ts_buf, int H, int W, int K,
    float t_query, float t0, float tmax, bool out_chw
){
    auto opts = at::TensorOptions().dtype(at::CppTypeToScalarType<scalar_t>()).device(ts_buf.device());
    at::Tensor out = out_chw
        ? at::empty({2 * K, H, W}, opts)
        : at::empty({2, K, H, W}, opts);

    int P = 2;
    i64 total = (i64)P * H * W * K;
    int threads = 256;
    int blocks  = (int)((total + threads - 1) / threads);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    make_tore_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        ts_buf.data_ptr<int>(),
        H, W, K,
        t_query, t0, tmax,
        out_chw,
        out.data_ptr<scalar_t>()
    );
    LAUNCH_CHECK();
    return out;
}

// 单样本 CUDA 路径
at::Tensor build_tore_volume_single_cuda(
    const at::Tensor& events_in, // [N,4]
    int64_t H, int64_t W, int64_t K,
    double t0_d, double tmax_d,
    c10::optional<int64_t> t_query_opt,
    bool out_chw,
    c10::ScalarType out_dtype
){
    TORCH_CHECK(events_in.dim() == 2 && events_in.size(1) == 4, "events must be [N,4]");
    TORCH_CHECK(events_in.is_cuda(), "events must be CUDA");
    c10::cuda::CUDAGuard guard(events_in.device());
    auto events = events_in.contiguous();

    const int N = (int)events.size(0);
    const float t0   = (float)t0_d;
    const float tmax = (float)tmax_d;

    // 空样本：直接填底
    if (N == 0) {
        float base = log1pf(tmax);
        auto opts = at::TensorOptions().dtype(out_dtype).device(events.device());
        if (out_chw) return at::full({2*(int)K, (int)H, (int)W}, base, opts);
        else         return at::full({2, (int)K, (int)H, (int)W}, base, opts);
    }

    // 拆列并转窄精度 (int32)
    auto x = events.select(1, 0).toType(at::kInt).contiguous();
    auto y = events.select(1, 1).toType(at::kInt).contiguous();
    auto t = events.select(1, 2).toType(at::kInt).contiguous();
    auto p = events.select(1, 3);
    auto p_idx = (p > 0).toType(at::kInt).contiguous();

    // t_query
    int max_t = (int) t.max().item<int>();
    int tq    = t_query_opt.has_value() ? (int)(*t_query_opt) : max_t;
    if (tq < 0) tq = max_t;

    // 过滤：t <= tq 且 坐标在 [0,H)×[0,W)
    auto keep = (t <= tq) & x.ge(0) & x.lt((int)W) & y.ge(0) & y.lt((int)H);
    auto xs = x.masked_select(keep);
    auto ys = y.masked_select(keep);
    auto ts = t.masked_select(keep);
    auto ps = p_idx.masked_select(keep);
    int M = (int)ts.size(0);
    if (M == 0) {
        float base = log1pf(tmax);
        auto opts = at::TensorOptions().dtype(out_dtype).device(events.device());
        if (out_chw) return at::full({2*(int)K, (int)H, (int)W}, base, opts);
        else         return at::full({2, (int)K, (int)H, (int)W}, base, opts);
    }

    // SHIFT/Mask
    int SHIFT = decide_shift_from_tmax(std::max(tq, (int)ts.max().item<int>()));
    u64 T_MASK = (SHIFT >= 63) ? (~0ULL) : ((1ULL << SHIFT) - 1ULL);

    // 使用当前流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 1) 打包 key 并排序
    at::Tensor keys = at::empty({M}, events.options().dtype(at::kLong)); // 存放 u64
    {
        int threads = 256;
        int blocks  = (M + threads - 1) / threads;
        pack_keys_kernel<<<blocks, threads, 0, stream>>>(
            xs.data_ptr<i32>(), ys.data_ptr<i32>(), ts.data_ptr<i32>(), ps.data_ptr<i32>(),
            (int)H, (int)W,
            reinterpret_cast<u64*>(keys.data_ptr<i64>()), M, SHIFT
        );
        LAUNCH_CHECK();

        auto d_keys = thrust::device_ptr<u64>(reinterpret_cast<u64*>(keys.data_ptr<i64>()));
        thrust::sort(thrust::cuda::par.on(stream), d_keys, d_keys + M); // 升序：(pi,t)
    }

    // 2) 计算从段尾的秩（1..count），在同一 stream 上执行
    at::Tensor rank_from_end = at::empty({M}, events.options().dtype(at::kInt));
    {
        auto d_keys = thrust::device_ptr<u64>(reinterpret_cast<u64*>(keys.data_ptr<i64>()));
        auto keys_pi_begin = thrust::make_transform_iterator(d_keys, PiOfKey{SHIFT});
        auto keys_pi_end   = keys_pi_begin + M;

        auto rkeys_begin = thrust::make_reverse_iterator(keys_pi_begin + M);
        auto rkeys_end   = thrust::make_reverse_iterator(keys_pi_begin);

        auto ones_begin  = thrust::make_constant_iterator(1);
        auto rvals_begin = thrust::make_reverse_iterator(ones_begin + M);

        auto drank_begin = thrust::device_pointer_cast(rank_from_end.data_ptr<int>());
        auto r_out_begin = thrust::make_reverse_iterator(drank_begin + M);

        thrust::inclusive_scan_by_key(
            thrust::cuda::par.on(stream),
            rkeys_begin, rkeys_end,
            rvals_begin,
            r_out_begin
        );
    }

    // 3) 选择末 K 个并 scatter 到 ts_buf:[2,H,W,K] (int32)
    at::Tensor ts_buf = at::full({2, (int)H, (int)W, (int)K}, -1, events.options().dtype(at::kInt));
    {
        int threads = 256;
        int blocks  = (M + threads - 1) / threads;
        scatter_selected_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const u64*>(keys.data_ptr<i64>()),
            rank_from_end.data_ptr<i32>(),
            M, (int)H, (int)W, (int)K, SHIFT, T_MASK,
            ts_buf.data_ptr<int>()
        );
        LAUNCH_CHECK();
    }

    // 4) 计算 Δt/log1p → 输出（仍在当前流）
    at::Tensor out;
    float tq_f = (float)tq;
    switch (out_dtype) {
        case at::kFloat:
            out = launch_make_tore<float>(ts_buf, (int)H, (int)W, (int)K, tq_f, t0, tmax, out_chw); break;
        case at::kBFloat16:
            out = launch_make_tore<at::BFloat16>(ts_buf, (int)H, (int)W, (int)K, tq_f, t0, tmax, out_chw); break;
        case at::kHalf:
            out = launch_make_tore<at::Half>(ts_buf, (int)H, (int)W, (int)K, tq_f, t0, tmax, out_chw); break;
        default:
            TORCH_CHECK(false, "Unsupported out dtype");
    }
    return out;
}

// 批量：循环调用单样本（B 通常很小）
std::vector<at::Tensor> build_tore_volume_batch_cuda(
    const std::vector<at::Tensor>& events_list,
    int64_t H, int64_t W, int64_t K,
    double t0, double tmax,
    const c10::optional<std::vector<int64_t>>& t_query_list,
    bool out_chw,
    c10::ScalarType out_dtype
){
    const int B = (int)events_list.size();
    std::vector<at::Tensor> outs;
    outs.reserve(B);
    for (int i = 0; i < B; ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && (int)t_query_list->size() == B) {
            tq = (*t_query_list)[i];
        }
        outs.push_back(build_tore_volume_single_cuda(
            events_list[i], H, W, K, t0, tmax, tq, out_chw, out_dtype
        ));
    }
    return outs;
}
