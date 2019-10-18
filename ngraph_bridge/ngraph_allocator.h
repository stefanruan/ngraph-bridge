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
#include <sstream>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/pool_allocator.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"

namespace tf = tensorflow;

//
// Useful API examples for host/device tensor movement in:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/copy_tensor.cc
// TBD
// Instead of exposing the Allocator class - define a factory mwethod?

namespace ngraph_bridge {

class NGraphAllocator : public tf::Allocator {
 public:
  // Given the backend - this will return an actual allocator
  //  For CPU - this will be a malloc wrapper - just for keeping track of
  //  memory that's not written to (i.e., read only etc. e.g., freshness
  //  tracker)
  //
  //  For Other devices - this will delegate the allocate call to the device
  //  The device provide a mem copy interface
  // The allocator class will also take the atrributes - which will
  // help determine whether the tensor is on device or on host
  //
  static std::unique_ptr<NGraphAllocator> CreateAllocator(
      const std::string& backend_name, const std::string& tensor_name) {
    // Construct a new allocator and return
    return std::unique_ptr<NGraphAllocator>(
        new NGraphAllocator(backend_name, tensor_name));
  }

  // Need to make it public so that the object can be destroyed by the
  // unique_ptr
  ~NGraphAllocator() override {}

 private:
  NGraphAllocator(const std::string& backend_name,
                  const std::string& tensor_name) {
    // Update the name of this allocator
    std::ostringstream oss;
    oss << "NGraphAllocator_" << backend_name << "_" << tensor_name;
    m_name = oss.str();

    // Add this name to the catalog
    // TBD

    // Anything else?
  }

  TF_DISALLOW_COPY_AND_ASSIGN(NGraphAllocator);

  inline tf::string Name() override { return m_name; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    std::cout << "NGraphAllocator::AllocateRaw called: Size: " << num_bytes
              << " bytes" << std::endl;
    // If this is a device tensor (i.e., the backend is non CPU)
    // then we will get a buffer from the backend and return 
    // that instead. 
    // TF will copy the data into this buffer
    //
    // else - it's a CPU buffer so return the mem. usong malloc as 
    // done below
    void* ptr = tf::port::AlignedMalloc(num_bytes, alignment);
    return ptr;
  }

  void DeallocateRaw(void* ptr) override {
    std::cout << "NGraphAllocator::DeallocateRaw called" << std::endl;
    if (ptr == nullptr) {
      LOG(ERROR) << "tried to deallocate nullptr";
      return;
    }

    tf::port::AlignedFree(ptr);
  }

 private:
  // Allocator name
  tf::string m_name;
};

}  // namespace ngraph_bridge

#endif  // NGRAPH_ALLOCATOR_H_
