

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

#ifndef NGRAPH_TF_ENCAP_TENSOR_META_H_
#define NGRAPH_TF_ENCAP_TENSOR_META_H_
#pragma once

namespace tensorflow {

namespace ngraph_bridge {

enum NGraphEncapTensorType { Variable, DataInput, Default };

class NGraphEncapTensorMeta {
  NGraphEncapTensor(int index, bool is_input, string name =””);
  virtual int GetIndex();
  virtual bool GetIsInput();
  virtual NGraphEncapTensorType GetTensorType();
  virtual string GetName();

  // Accept Visitor to Create the Tensor
  shared_ptr<ng::runtime::tensor> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this);
  }

  vector<shared_ptr<ng::runtime::tensor>> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator, int pipeline_depth) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this, pipeline_depth);
  }

  // Write/Read
  WriteToBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                           TF::Tensor* tf_ptr) {}

  ReadFromBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                            TF::Tensor* tf_ptr) {}

  Members : 
  string name;  // ng_cluster_6_input_0, optional;
  int index;
  bool is_input;
  NGraphEncapTensorType tensor_type = Default;
};

class NGraphEncapDataTensorMeta : public NGraphEncapTensorMeta {
  NGraphEncapDataTensorMeta(int index, bool is_input, string name =””);

  // Accept Visitor to Create the Tensor
  shared_ptr<ng::runtime::tensor> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this);
  }

  vector<shared_ptr<ng::runtime::tensor>> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator, int pipeline_depth) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this, pipeline_depth);
  }

  // Write/Read
  WriteToBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                           TF::Tensor* tf_ptr) {}

  ReadFromBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                            TF::Tensor* tf_ptr) {}

  members : int index;
  bool is_input;
  NGraphEncapTensorType tensor_type = DataInput;
};

class NGraphEncapVariableTensorMeta : public NGraphEncapTensorMeta {
  NGraphEncapVariableTensorMeta(int index, bool is_input, string name =””);

  // Accept Visitor to Create the Tensor
  shared_ptr<ng::runtime::tensor> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this);
  }
  
  vector<shared_ptr<ng::runtime::tensor>> GetTensor(
      NGraphEncapTensorCreator ng_tensor_creator, int pipeline_depth) {
    return ng_tensor_creator->LookUpOrCreateTensor(*this, pipeline_depth);
  }

  // Write/Read
  WriteToBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                           TF::Tensor* tf_ptr) {
    // Nothing is done
  }

  ReadFromBackendIfRequired(shared_ptr<ngraph::runtime::tensor> ng_tensor,
                            TF::Tensor* tf_ptr) {}

  members : int index;
  bool is_input;
  string shared_name_;
  NGraphEncapTensorType tensor_type = Variable;
};

}  // ngraph_bridge
}  // tensorflow

#endif  // NGRAPH_TF_ENCAP_TENSOR_META_H_
