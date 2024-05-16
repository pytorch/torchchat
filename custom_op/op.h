#pragma once

#include <torch/library.h>
#include <torch/script.h>
#include <xnnpack.h>

class XNNWeightsCache {
   public:
    XNNWeightsCache();
    bool registerWeight(std::string name, const void* data);
    bool Finalize();
    xnn_weights_cache_t Get();
    XNNWeightsCache(const at::Tensor& tensor);
    at::Tensor to_tensor();

   private:
    bool is_finalized_ = false;
    size_t offset_tracking_ = 0;
    std::map<const void*, std::string> weights_to_name_;
    std::map<std::string, std::pair<size_t, size_t>> name_to_offset_size_;
    std::map<size_t, void*> offset_to_addr_;
    bool from_tensor_ = false;
    size_t alignment_ = 64;
    size_t buffer_size = 0;
    std::string buffer_for_packed_weights_;
    xnn_weights_cache_provider weights_cache_;

    void* get_aligned(void* ptr) {
        return (void*)((intptr_t)ptr + alignment_ - (intptr_t)ptr % alignment_);
    }

    static size_t look_up(
        XNNWeightsCache* context,
        const xnn_weights_cache_look_up_key* cache_key);

    static void* reserve_space(XNNWeightsCache* context, size_t n);

    static size_t look_up_or_insert(
        XNNWeightsCache* context,
        const xnn_weights_cache_look_up_key* cache_key,
        void* ptr,
        size_t size);

    static bool is_finalized(XNNWeightsCache* context);

    static void* offset_to_addr(XNNWeightsCache* context, size_t offset);

    static enum xnn_status delete_cache(XNNWeightsCache* context);
};
