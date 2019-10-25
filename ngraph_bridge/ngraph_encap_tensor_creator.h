
/*******************************************************************************
 * Copyright 2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef NGRAPH_TF_ENCAP_TENSOR_CREATOR_H_
#define NGRAPH_TF_ENCAP_TENSOR_CREATOR_H_
#pragma once

namespace tensorflow {

namespace ngraph_bridge {

class NGraphEncapTensorCreator {
  NGraphEncapTensorCreator(ng::runtime::Backend, ng::runtime::Executable, OpKernelContext* );
  
    // default type which is the backend->create_tensor
    shared_ptr<ng::runtime::tensor> LookUpOrCreateTensor(NGraphEncapTensorMeta) {
        return backend->create_tensor();
        // needs data type and shape so should NGraphEncapTensorMeta store it
    }

    vector<shared_ptr<ng::runtime::tensor>> LookUpOrCreateTensor(NGraphEncapTensorMeta,
                                                    int pipeline_depth) {

                                                    }
    // Var create tensor
  shared_ptr<ng::runtime::tensor> LookUpOrCreateTensor(NGraphEncapVariableTensorMeta) {
    String var_shared_name = NGraphEncapVariableTensorMeta->GetSharedName()

    // 1. Find Var from the ctx
    // 2. Sync for input , do not sync for output
    // Can be determined from NGraphEncapVariableTensorMeta->IsInput()
  }

 
  
    // executable create tensor   
  shared_ptr<ng::runtime::tensor> LookUpOrCreateTensor(NGraphEncapDataTensorMeta) {
    if(NGraphEncapDataTensorMeta->GetIsInput()){
        ng_exec->create_input_tensor(NGraphEncapDataTensorMeta->GetIndex());
    }
    else{
        ng_exec->create_output_tensor(NGraphEncapDataTensorMeta->GetIndex());
  
  }

  vector<shared_ptr<ng::runtime::tensor>> LookUpOrCreateTensor(NGraphEncapDataTensorMeta,
                                                    int pipeline_depth) {
    if(NGraphEncapDataTensorMeta->GetIsInput()){
        return ng_exec->create_input_tensor(NGraphEncapDataTensorMeta->GetIndex(), pipeline_depth);
    }
    else{ 
        return ng_exec->create_output_tensor(NGraphEncapDataTensorMeta->GetIndex(),
                                       pipeline_depth);
    }
  }
};

members:
backend
exec
context

}  // ngraph_bridge
}  // tensorflow

#endif // NGRAPH_TF_ENCAP_TENSOR_CREATOR_H_
