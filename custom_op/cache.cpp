#include <op.h>

XNNWeightsCache::XNNWeightsCache() {
    weights_cache_.context = this;
    weights_cache_.look_up = (size_t(*)(
        void*, const xnn_weights_cache_look_up_key*))XNNWeightsCache::look_up;
    weights_cache_.reserve_space =
        (void* (*)(void*, size_t))XNNWeightsCache::reserve_space;
    weights_cache_.look_up_or_insert =
        (size_t(*)(void*, const xnn_weights_cache_look_up_key*, void*, size_t))
            XNNWeightsCache::look_up_or_insert;
    weights_cache_.is_finalized = (bool (*)(void*))XNNWeightsCache::is_finalized;
    weights_cache_.offset_to_addr =
        (void* (*)(void*, size_t))XNNWeightsCache::offset_to_addr;
    weights_cache_.delete_cache =
        (enum xnn_status(*)(void*))XNNWeightsCache::delete_cache;
}

XNNWeightsCache::XNNWeightsCache(const at::Tensor& weight) : XNNWeightsCache() {
    from_tensor_ = true;
    buffer_for_packed_weights_.resize(weight.numel() + alignment_, 5);
    void* start = get_aligned(buffer_for_packed_weights_.data());
    memcpy(start, weight.data_ptr(), weight.numel());
}

at::Tensor XNNWeightsCache::to_tensor() {
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    void* start = get_aligned(buffer_for_packed_weights_.data());
    size_t packed_weight_size = buffer_for_packed_weights_.size() - alignment_;
    auto t =  at::from_blob(
            start,
            {(int64_t)(packed_weight_size)},
            options);
    return t.clone();
}

bool XNNWeightsCache::registerWeight(std::string name, const void* data) {
    weights_to_name_[data] = name;
    return true;
}

bool XNNWeightsCache::Finalize() {
    is_finalized_ = true;
    return true;
}

xnn_weights_cache_t XNNWeightsCache::Get() {
    return (xnn_weights_cache_t)&weights_cache_;
}

size_t XNNWeightsCache::look_up(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key) {
    const void* unpacked_weights_ptr = cache_key->kernel;
    if (context->from_tensor_) {
        return 0;
    }
    if (auto entry = context->weights_to_name_.find(unpacked_weights_ptr);
        entry != context->weights_to_name_.end()) {
        std::string weight_name = entry->second;
        if (auto entry = context->name_to_offset_size_.find(weight_name);
            entry != context->name_to_offset_size_.end()) {
        return (size_t)(entry->second.first);
        }
    }
    return SIZE_MAX;
}

void* XNNWeightsCache::reserve_space(XNNWeightsCache* context, size_t n) {
    context->buffer_for_packed_weights_.resize(n + context->alignment_);
    void* maybe_aligned_space = context->buffer_for_packed_weights_.data();
    void* aligned_ptr = context->get_aligned(maybe_aligned_space);
    return aligned_ptr;
}

size_t XNNWeightsCache::look_up_or_insert(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key,
    void* ptr,
    size_t size) {
    size_t offset = context->look_up(context, cache_key);

    if (offset != SIZE_MAX) {
        void* saved_ptr = offset_to_addr(context, offset);
        if (0 == memcmp(ptr, saved_ptr, size)) {
            return offset;
        }
        // Failure, cache is out of date
        return SIZE_MAX;
    }

    const void* unpacked_weights_ptr = cache_key->kernel;
    if (auto entry = context->weights_to_name_.find(unpacked_weights_ptr);
        entry != context->weights_to_name_.end()) {
        std::string weight_name = entry->second;
        size_t offset = context->offset_tracking_++;
        context->name_to_offset_size_[weight_name] = std::make_pair(offset, size);
        context->offset_to_addr_[offset] = ptr;
        return offset;
    }
    return SIZE_MAX;
}

bool XNNWeightsCache::is_finalized(XNNWeightsCache* context) {
    return context->is_finalized_;
}

void* XNNWeightsCache::offset_to_addr(XNNWeightsCache* context, size_t offset) {
    if (context->from_tensor_ and !offset) {
        void* ptr = context->buffer_for_packed_weights_.data();
        void* aligned_ptr = context->get_aligned(ptr);
        return aligned_ptr;
    }
    if (auto entry = context->offset_to_addr_.find(offset);
        entry != context->offset_to_addr_.end()) {
        return entry->second;
    }
    return nullptr;
}

enum xnn_status XNNWeightsCache::delete_cache(XNNWeightsCache* context) {
    return xnn_status_success;
}
