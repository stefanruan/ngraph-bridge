/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Allocator for nGraph 

#ifndef NGRAPH_ALLOCATOR_H_
#define NGRAPH_ALLOCATOR_H_

#include <cstdlib>
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"

namespace tf=tensorflow;

namespace ngraph_bridge {

class NGraphSubAllocator : public tf::BasicCPUAllocator {
 public:
  NGraphSubAllocator() : BasicCPUAllocator(tf::port::kNUMANoAffinity, {}, {}) {}
  ~NGraphSubAllocator() override {}
};

// CPU allocator that handles small-size allocations by calling
// suballocator directly. Mostly, it is just a wrapper around a suballocator
// (that calls malloc and free directly) with support for bookkeeping.
class NGraphAllocator : public tf::Allocator {
 public:
  NGraphAllocator(tf::SubAllocator* sub_allocator,
                        const tf::string& name)
      : sub_allocator_(sub_allocator), name_(name) {
    //stats_.bytes_limit = total_memory;
  }
  ~NGraphAllocator() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphAllocator);

  inline tf::string Name() override { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    std::cout << "SmallSizeAllocator::AllocateRaw called: Size: " << num_bytes << " bytes" << std::endl;
    void* ptr = tf::port::AlignedMalloc(num_bytes, alignment);
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    std::cout << "DeallocateRaw called" << std::endl;
    if (ptr == nullptr) {
      LOG(ERROR) << "tried to deallocate nullptr";
      return;
    }

    tf::port::AlignedFree(ptr);
  }

  absl::optional<tf::AllocatorStats> GetStats() override {
    tf::mutex_lock l(mutex_);
    return stats_;
  }

  void ClearStats() override {
    tf::mutex_lock l(mutex_);
    stats_.num_allocs = 0;
    stats_.peak_bytes_in_use = 0;
    stats_.largest_alloc_size = 0;
    stats_.bytes_in_use = 0;
    stats_.bytes_limit = 0;
  }

 private:
  // Increment statistics for the allocator handling small allocations.
  inline void IncrementStats(size_t alloc_size) LOCKS_EXCLUDED(mutex_) {
    tf::mutex_lock l(mutex_);
    ++stats_.num_allocs;
    stats_.bytes_in_use += alloc_size;
    stats_.peak_bytes_in_use =
        std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
    stats_.largest_alloc_size =
        std::max(alloc_size, static_cast<size_t>(stats_.largest_alloc_size));
  }

  // Decrement statistics for the allocator handling small allocations.
  inline void DecrementStats(size_t dealloc_size) LOCKS_EXCLUDED(mutex_) {
    tf::mutex_lock l(mutex_);
    stats_.bytes_in_use -= dealloc_size;
  }

  tf::SubAllocator* sub_allocator_;  // Not owned by this class.

  // Mutex for protecting updates to map of allocations.
  mutable tf::mutex mutex_;

  // Allocator name
  tf::string name_;

  // Allocator stats for small allocs
  tf::AllocatorStats stats_ GUARDED_BY(mutex_);
};

}  // namespace ngraph_bridge

#endif  // NGRAPH_ALLOCATOR_H_
