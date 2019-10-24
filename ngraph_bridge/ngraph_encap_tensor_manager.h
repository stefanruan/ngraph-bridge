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
#ifndef NGRAPH_TF_ENCAP_TENSOR_MANAGER_H_
#define NGRAPH_TF_ENCAP_TENSOR_MANAGER_H_
#pragma once

namespace tensorflow {

namespace ngraph_bridge {


class NGraphEncapTensorManager{

NGraphEncapTensorManager(int num_inputs, int num_ouputs, string name=””);

// Takes in a graph and initializes the input and output tensor objects
// using the catalog
Initialize(Graph* , num_of_inputs_, num_of_outputs_){
    //inputs
// For each input in this graph
// 	initial NGEncapTensorMeta;
//  Checks the catalog
//  If input is var:
//      NGEncapTensorMeta = Creates object of NGraphEncapVariableTensorMeta(shared_name)
//  Else If input is data_pipelined:
//      NGEncapTensorMeta = Creates object of NGraphEncapDataTensorMeta(shared_name)
//  Else If:
//      NGEncapTensorMeta = ….
//  Else:
	
// 	input_tensors.add(NGEncapTensorMeta)
	
	//outputs
		// Same thing for outputs
}

vector<NGraphEncapTensorMeta> GetInputTensorMeta();
vector<NGraphEncapTensorMeta> GetOutputTensorMeta();

GetInput(index);
GetOutput(index);

private:
String name; // eg. ngraph_cluster_6_tensor_manager;
int num_of_inputs_;
int num_of_outputs_;
vector<NGraphEncapTensorMeta> input_tensors_meta;
vector<NGraphEncapTensorMeta> output_tensors_meta;
};


}  // ngraph_bridge
}  // tensorflow

#endif  // NGRAPH_TF_ENCAP_TENSOR_MANAGER_H_