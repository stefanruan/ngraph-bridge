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

// NGraphExecutor::Constructor

class NGraphExecutor{
	
// Constructor
NGraphExecutor(){

// find and set the number of inputs and outputs of the encapsulated graph
// could be a util function, need to validate this assumption
// that no of “_Args” == no of inputs
m_number_of_inputs_ = FindNumberOfNodes(Graph* graph, “_Args”);
m_number_of_outputs_ = FindNumberOfNodes(Graph* graph, “_Retvals”);
ng_encap_tensor_manager_.Initialize(Graph* graph, m_number_of_inputs_, m_number_of_outputs_);

}

// OThers

Status GetTensorsFromPipeline(
      const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
      NGraphEncapTensorCreator* ng_tensor_creator,
      std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>&
          io_tensors);


Status WriteTensorsToBackend (vector<shared_ptr<ngraph::runtime::tensor>>, vector<Tensor>&tf_input_tensors);



Status ReadTensorsFomBackend (vector<shared_ptr<ngraph::runtime::tensor>>, vector<Tensor>&tf_output_tensors);

private:
    int m_number_of_inputs_;
    int m_number_of_outputs_;
    NGraphEncapTensorManager ng_encap_tensor_manager_;
}


};



// Status NGraphExecutor::GetTensorsFromPipeline
Status NGraphExecutor::GetTensorsFromPipeline(){
	InitializeIOTensorPipeline(){
		//Initializes only tensors that need to be “pipelined”
		// ie data tensors : 
		if(NGraphEncapTensorMeta->GetType() is DataInput)
            NGraphEncapTensorMeta->GetTensor(ng_tensor_creator, m_depth)
        }

    // Other tensors e.g. variables are fitted here
    if(NGraphEncapTensorMeta->GetType() is Variable){
        NGraphEncapTensorMeta->GetTensor(ng_tensor_creator);
    }

    //The PipelinedTensorStore will have to change from a matrix to a map of indexes and vectors
}



Status NGraphExecutor::GetTensorsFromPipeline(){
	InitializeIOTensorPipeline(){
		// Creates a complete matrix of m_depth 
		// With the same variable_tensor being fed in for each depth	
        For each NGraphEncapTensorMeta in ng_encap_tensor_manager_->GetInputTensorMeta:
            NGraphEncapTensorMeta->GetTensor(ng_tensor_creator, m_depth)
        }
}

// WriteTensorsToBackend
Status NGraphExecutor::WriteTensorsToBackend (vector<shared_ptr<ngraph::runtime::tensor>>, vector<Tensor>&tf_input_tensors)
{

    For each NGraphEncapTensorMeta in NGraphEncapTensorManager::GetInputs:
	    NGraphEncapTensorMeta->WriteToBackendIfRequired(shared_ptr<ngraph::runtime::tensor>, tf_ptr)
}

// ReadTensorsToBackend
Status ReadTensorsFomBackend (vector<shared_ptr<ngraph::runtime::tensor>>, vector<Tensor>&tf_output_tensors)
{

    For each NGraphEncapTensorMeta in ng_encap_tensor_manager_->GetOutputTensorMeta:
	    NGraphEncapTensorMeta->ReadFromBackendIfRequired(shared_ptr<ngraph::runtime::tensor>, tf_ptr)
}
