// tore_cuda.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <algorithm>

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

// 事件resize函数：按比例resize事件坐标
at::Tensor resize_events(const at::Tensor& events, int64_t orig_H, int64_t orig_W, int64_t target_H, int64_t target_W) {
    TORCH_CHECK(events.is_cuda(), "events must be CUDA tensor");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    
    if (orig_H == target_H && orig_W == target_W) {
        return events.clone();
    }
    
    // 只允许向下resize或等比例resize
    TORCH_CHECK(target_H <= orig_H && target_W <= orig_W,
                "Only downward resize is allowed, got target_H=", target_H, " orig_H=", orig_H,
                " target_W=", target_W, " orig_W=", orig_W);
    
    // 自动转换为浮点类型以支持resize计算
    auto events_float = events.dtype() == at::kFloat ? events : events.to(at::kFloat);
    auto resized_events = events_float.clone();
    auto x_coords = resized_events.select(1, 0);
    auto y_coords = resized_events.select(1, 1);
    
    // 计算缩放比例
    float scale_x = static_cast<float>(target_W) / static_cast<float>(orig_W);
    float scale_y = static_cast<float>(target_H) / static_cast<float>(orig_H);
    
    // 缩放坐标
    x_coords.mul_(scale_x);
    y_coords.mul_(scale_y);
    
    return resized_events;
}

// 事件resize为正方形：518x518
at::Tensor resize_events_square(const at::Tensor& events, int64_t orig_H, int64_t orig_W, int64_t target_size) {
    TORCH_CHECK(events.is_cuda(), "events must be CUDA tensor");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    
    if (orig_H == target_size && orig_W == target_size) {
        return events.clone();
    }
    
    // 自动转换为浮点类型以支持resize计算
    auto events_float = events.dtype() == at::kFloat ? events : events.to(at::kFloat);
    auto resized_events = events_float.clone();
    auto x_coords = resized_events.select(1, 0);
    auto y_coords = resized_events.select(1, 1);
    auto t_coords = resized_events.select(1, 2);
    auto p_coords = resized_events.select(1, 3);
    
    // 计算缩放比例：以宽度为基准
    float scale = static_cast<float>(target_size) / static_cast<float>(orig_W);
    float scaled_H = orig_H * scale;
    
    // 缩放x坐标
    x_coords.mul_(scale);
    
    if (scaled_H >= target_size) {
        // 高度大于等于目标尺寸，需要中心crop
        float crop_offset = (scaled_H - target_size) / 2.0f;
        
        // 缩放y坐标并应用crop
        y_coords.mul_(scale).sub_(crop_offset);
        
        // 过滤在有效范围内的坐标
        auto valid_mask = (x_coords >= 0) & (x_coords < target_size) &
                         (y_coords >= 0) & (y_coords < target_size);
        
        return resized_events.index_select(0, valid_mask.nonzero().squeeze(1));
    } else {
        // 高度小于目标尺寸，需要填充空白区域
        y_coords.mul_(scale);
        
        // 计算y方向的偏移，使图像居中
        float y_offset = (target_size - scaled_H) / 2.0f;
        y_coords.add_(y_offset);
        
        // 过滤在有效范围内的坐标
        auto valid_mask = (x_coords >= 0) & (x_coords < target_size) &
                         (y_coords >= 0) & (y_coords < target_size);
        
        return resized_events.index_select(0, valid_mask.nonzero().squeeze(1));
    }
}

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

// Python 端 API：带resize功能的单样本
at::Tensor tore_build_single_resized(
    const at::Tensor& events,
    int64_t orig_H, int64_t orig_W,  // 原始尺寸
    int64_t target_H, int64_t target_W,  // 目标尺寸
    int64_t K,
    double t0, double tmax,
    c10::optional<int64_t> t_query,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(events.is_cuda(), "events must be CUDA tensor");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    c10::cuda::CUDAGuard guard(events.device());
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    
    // 先resize事件
    auto resized_events = resize_events(events, orig_H, orig_W, target_H, target_W);
    
    return build_tore_volume_single_cuda(resized_events, target_H, target_W, K, t0, tmax, t_query, out_chw, out_dtype);
}

// Python 端 API：带正方形resize功能的单样本
at::Tensor tore_build_single_square(
    const at::Tensor& events,
    int64_t orig_H, int64_t orig_W,  // 原始尺寸
    int64_t target_size,  // 目标正方形尺寸（如518）
    int64_t K,
    double t0, double tmax,
    c10::optional<int64_t> t_query,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(events.is_cuda(), "events must be CUDA tensor");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    c10::cuda::CUDAGuard guard(events.device());
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    
    // 先resize事件为正方形
    auto resized_events = resize_events_square(events, orig_H, orig_W, target_size);
    
    return build_tore_volume_single_cuda(resized_events, target_size, target_size, K, t0, tmax, t_query, out_chw, out_dtype);
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

// Python 端 API：带resize功能的批量（返回 list）
std::vector<at::Tensor> tore_build_batch_resized(
    const std::vector<at::Tensor>& events_list,
    const std::vector<int64_t>& orig_H_list,  // 每个样本的原始H
    const std::vector<int64_t>& orig_W_list,  // 每个样本的原始W
    int64_t target_H, int64_t target_W,  // 目标尺寸
    int64_t K,
    double t0, double tmax,
    c10::optional<std::vector<int64_t>> t_query_list,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(!events_list.empty(), "events_list is empty");
    TORCH_CHECK(events_list.size() == orig_H_list.size() && events_list.size() == orig_W_list.size(),
                "events_list size must match orig_H_list and orig_W_list size");
    
    auto dev = events_list[0].device();
    for (auto& ev : events_list) {
        TORCH_CHECK(ev.is_cuda(), "all events must be CUDA tensors");
        TORCH_CHECK(ev.device() == dev, "all events must be on the same device");
        TORCH_CHECK(ev.dim() == 2 && ev.size(1) == 4, "each events must be [N,4]");
    }
    c10::cuda::CUDAGuard guard(dev);
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    
    std::vector<at::Tensor> outs;
    outs.reserve(events_list.size());
    
    for (size_t i = 0; i < events_list.size(); ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && t_query_list->size() == events_list.size()) {
            tq = (*t_query_list)[i];
        }
        
        // 先resize事件
        auto resized_events = resize_events(events_list[i], orig_H_list[i], orig_W_list[i], target_H, target_W);
        
        outs.push_back(build_tore_volume_single_cuda(
            resized_events, target_H, target_W, K, t0, tmax, tq, out_chw, out_dtype
        ));
    }
    return outs;
}

// Python 端 API：带正方形resize功能的批量（返回 list）
std::vector<at::Tensor> tore_build_batch_square(
    const std::vector<at::Tensor>& events_list,
    const std::vector<int64_t>& orig_H_list,  // 每个样本的原始H
    const std::vector<int64_t>& orig_W_list,  // 每个样本的原始W
    int64_t target_size,  // 目标正方形尺寸（如518）
    int64_t K,
    double t0, double tmax,
    c10::optional<std::vector<int64_t>> t_query_list,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(!events_list.empty(), "events_list is empty");
    TORCH_CHECK(events_list.size() == orig_H_list.size() && events_list.size() == orig_W_list.size(),
                "events_list size must match orig_H_list and orig_W_list size");
    
    auto dev = events_list[0].device();
    for (auto& ev : events_list) {
        TORCH_CHECK(ev.is_cuda(), "all events must be CUDA tensors");
        TORCH_CHECK(ev.device() == dev, "all events must be on the same device");
        TORCH_CHECK(ev.dim() == 2 && ev.size(1) == 4, "each events must be [N,4]");
    }
    c10::cuda::CUDAGuard guard(dev);
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;
    
    std::vector<at::Tensor> outs;
    outs.reserve(events_list.size());
    
    for (size_t i = 0; i < events_list.size(); ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && t_query_list->size() == events_list.size()) {
            tq = (*t_query_list)[i];
        }
        
        // 先resize事件为正方形
        auto resized_events = resize_events_square(events_list[i], orig_H_list[i], orig_W_list[i], target_size);
        
        outs.push_back(build_tore_volume_single_cuda(
            resized_events, target_size, target_size, K, t0, tmax, tq, out_chw, out_dtype
        ));
    }
    return outs;
}

// Python 端 API：批量（直接返回堆叠后的 batched tensor）
at::Tensor tore_build_batch_stacked(
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

    const int64_t B = (int64_t)events_list.size();
    at::Tensor out;
    auto opts = at::TensorOptions().dtype(out_dtype).device(dev);
    if (out_chw) {
        out = at::empty({B, 2*(int64_t)K, (int64_t)H, (int64_t)W}, opts);
    } else {
        out = at::empty({B, 2, (int64_t)K, (int64_t)H, (int64_t)W}, opts);
    }
    for (int64_t i = 0; i < B; ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && (int64_t)t_query_list->size() == B) {
            tq = (*t_query_list)[i];
        }
        at::Tensor one = build_tore_volume_single_cuda(events_list[i], H, W, K, t0, tmax, tq, out_chw, out_dtype);
        out.select(0, i).copy_(one);
    }
    return out;
}

// Python 端 API：带resize功能的批量（直接返回堆叠后的 batched tensor）
at::Tensor tore_build_batch_stacked_resized(
    const std::vector<at::Tensor>& events_list,
    const std::vector<int64_t>& orig_H_list,
    const std::vector<int64_t>& orig_W_list,
    int64_t target_H, int64_t target_W,
    int64_t K,
    double t0, double tmax,
    c10::optional<std::vector<int64_t>> t_query_list,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(!events_list.empty(), "events_list is empty");
    TORCH_CHECK(events_list.size() == orig_H_list.size() && events_list.size() == orig_W_list.size(),
                "events_list size must match orig_H_list and orig_W_list size");
    
    auto dev = events_list[0].device();
    for (auto& ev : events_list) {
        TORCH_CHECK(ev.is_cuda(), "all events must be CUDA tensors");
        TORCH_CHECK(ev.device() == dev, "all events must be on the same device");
        TORCH_CHECK(ev.dim() == 2 && ev.size(1) == 4, "each events must be [N,4]");
    }
    c10::cuda::CUDAGuard guard(dev);
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;

    const int64_t B = (int64_t)events_list.size();
    at::Tensor out;
    auto opts = at::TensorOptions().dtype(out_dtype).device(dev);
    if (out_chw) {
        out = at::empty({B, 2*(int64_t)K, (int64_t)target_H, (int64_t)target_W}, opts);
    } else {
        out = at::empty({B, 2, (int64_t)K, (int64_t)target_H, (int64_t)target_W}, opts);
    }
    
    for (int64_t i = 0; i < B; ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && (int64_t)t_query_list->size() == B) {
            tq = (*t_query_list)[i];
        }
        
        // 先resize事件
        auto resized_events = resize_events(events_list[i], orig_H_list[i], orig_W_list[i], target_H, target_W);
        
        at::Tensor one = build_tore_volume_single_cuda(resized_events, target_H, target_W, K, t0, tmax, tq, out_chw, out_dtype);
        out.select(0, i).copy_(one);
    }
    return out;
}

// Python 端 API：带正方形resize功能的批量（直接返回堆叠后的 batched tensor）
at::Tensor tore_build_batch_stacked_square(
    const std::vector<at::Tensor>& events_list,
    const std::vector<int64_t>& orig_H_list,
    const std::vector<int64_t>& orig_W_list,
    int64_t target_size,
    int64_t K,
    double t0, double tmax,
    c10::optional<std::vector<int64_t>> t_query_list,
    bool out_chw,
    c10::optional<at::ScalarType> dtype_opt
) {
    TORCH_CHECK(!events_list.empty(), "events_list is empty");
    TORCH_CHECK(events_list.size() == orig_H_list.size() && events_list.size() == orig_W_list.size(),
                "events_list size must match orig_H_list and orig_W_list size");
    
    auto dev = events_list[0].device();
    for (auto& ev : events_list) {
        TORCH_CHECK(ev.is_cuda(), "all events must be CUDA tensors");
        TORCH_CHECK(ev.device() == dev, "all events must be on the same device");
        TORCH_CHECK(ev.dim() == 2 && ev.size(1) == 4, "each events must be [N,4]");
    }
    c10::cuda::CUDAGuard guard(dev);
    auto out_dtype = dtype_opt.has_value() ? *dtype_opt : at::kFloat;

    const int64_t B = (int64_t)events_list.size();
    at::Tensor out;
    auto opts = at::TensorOptions().dtype(out_dtype).device(dev);
    if (out_chw) {
        out = at::empty({B, 2*(int64_t)K, (int64_t)target_size, (int64_t)target_size}, opts);
    } else {
        out = at::empty({B, 2, (int64_t)K, (int64_t)target_size, (int64_t)target_size}, opts);
    }
    
    for (int64_t i = 0; i < B; ++i) {
        c10::optional<int64_t> tq;
        if (t_query_list.has_value() && (int64_t)t_query_list->size() == B) {
            tq = (*t_query_list)[i];
        }
        
        // 先resize事件为正方形
        auto resized_events = resize_events_square(events_list[i], orig_H_list[i], orig_W_list[i], target_size);
        
        at::Tensor one = build_tore_volume_single_cuda(resized_events, target_size, target_size, K, t0, tmax, tq, out_chw, out_dtype);
        out.select(0, i).copy_(one);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tore_build_single", &tore_build_single,
          "Build TORE for a single sample on CUDA");
    m.def("tore_build_batch",  &tore_build_batch,
          "Build TORE for a batch on CUDA (returns list of tensors)");
    m.def("tore_build_batch_stacked",  &tore_build_batch_stacked,
          "Build TORE for a batch on CUDA (returns stacked tensor)");
    m.def("tore_build_single_resized", &tore_build_single_resized,
          "Build TORE for a single sample with resize on CUDA");
    m.def("tore_build_single_square", &tore_build_single_square,
          "Build TORE for a single sample with square resize on CUDA");
    m.def("tore_build_batch_resized", &tore_build_batch_resized,
          "Build TORE for a batch with resize on CUDA (returns list of tensors)");
    m.def("tore_build_batch_square", &tore_build_batch_square,
          "Build TORE for a batch with square resize on CUDA (returns list of tensors)");
    m.def("tore_build_batch_stacked_resized", &tore_build_batch_stacked_resized,
          "Build TORE for a batch with resize on CUDA (returns stacked tensor)");
    m.def("tore_build_batch_stacked_square", &tore_build_batch_stacked_square,
          "Build TORE for a batch with square resize on CUDA (returns stacked tensor)");
    m.def("tore_version", [](){ return std::string("0.0.2"); }, "Return tore_cuda version");
    m.attr("__version__") = "0.0.2";
}
