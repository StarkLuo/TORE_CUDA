// tore_cuda.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// 声明 CUDA 实现
at::Tensor build_tore_volume_single_cuda(
    const at::Tensor& events, // [N,4] (x,y,t,p) on CUDA
    int64_t H, int64_t W, int64_t K,
    double t0, double tmax,
    c10::optional<int64_t> t_query,
    bool out_chw,              // true: (2K,H,W), false: (2,K,H,W)
    c10::ScalarType out_dtype  // kFloat / kBFloat16 / kHalf
);

std::vector<at::Tensor> build_tore_volume_batch_cuda(
    const std::vector<at::Tensor>& events_list,
    int64_t H, int64_t W, int64_t K,
    double t0, double tmax,
    const c10::optional<std::vector<int64_t>>& t_query_list,
    bool out_chw,
    c10::ScalarType out_dtype
);

// Python 端 API：单样本
at::Tensor tore_build_single(
    const at::Tensor& events,
    int64_t H, int64_t W, int64_t K,
    double t0, double tmax,
    c10::optional<int64_t> t_query,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(events.is_cuda(), "events must be CUDA tensor");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    c10::cuda::CUDAGuard guard(events.device());
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    return build_tore_volume_single_cuda(events, H, W, K, t0, tmax, t_query, out_chw, out_dtype);
}

// Python 端 API：批量（返回 list，再在 Python 里 stack）
std::vector<at::Tensor> tore_build_batch(
    const std::vector<at::Tensor>& events_list,
    int64_t H, int64_t W, int64_t K,
    double t0, double tmax,
    c10::optional<std::vector<int64_t>> t_query_list,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(!events_list.empty(), "events_list is empty");
    auto dev = events_list[0].device();
    for (auto& ev : events_list) {
        TORCH_CHECK(ev.is_cuda(), "all events must be CUDA tensors");
        TORCH_CHECK(ev.device() == dev, "all events must be on the same device");
        TORCH_CHECK(ev.dim() == 2 && ev.size(1) == 4, "each events must be [N,4]");
    }
    c10::cuda::CUDAGuard guard(dev);
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    return build_tore_volume_batch_cuda(events_list, H, W, K, t0, tmax, t_query_list, out_chw, out_dtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tore_build_single", &tore_build_single,
          "Build TORE for a single sample on CUDA");
    m.def("tore_build_batch",  &tore_build_batch,
          "Build TORE for a batch on CUDA (returns list of tensors)");
}