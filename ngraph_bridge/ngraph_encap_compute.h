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


NGraphEncapsulateOp::Compute(OpKernelContext* ctx)
{
    // We create ng-exec
    m_parallel_executor->GetNgExecutable()

    // At this point we have the exec, backend, context 
    // These 3 are required for creating the different types of tensors
    NGraphEncapTensorCreator ng_tensor_creator(backend, exec, context);

    // Now 
	io_tensors = m_parallel_executor->GetTensorsFromPipeline(ng_exec, ng_tensor_creator, io_tensors)
    // At this ng_tensor_creator can be freed or destroyed.

	// Write the Tensors
	m_parallel_executor->WriteTensorsToBackend(io_tensors<1>, tf_input_tensors)
	
	// ngraph execute call
	ng_exec->call(get<2>(io_tensors), get<1>(io_tensors));
	

	// Read the Tensors
	// TF output allocation can happen here as it requires context
	// ie we allocate TF tensors of the right shape and data type here
	// the actual read happens inside the function ReadTensorsFromBackend
	
	m_parallel_executor->ReadTensorsFromBackend(io_tensors<2>, tf_input_tensors)
}
