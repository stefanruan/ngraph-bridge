/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
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
#include "ngraph_bridge/grappler/ngraph_add_identityn.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status AddIdentityN(Graph* input_graph, std::map<string, std::set<int>> fetch_node_info) {
  for (auto node : input_graph->op_nodes()) {
    bool fetch_node = false;
    bool ref_type = false;
    fetch_node = fetch_node_info.find(node->name()) != fetch_node_info.end();
    if (fetch_node) {
      NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Fetch Node " << node->name();
      // Check the number of outputs of the 'fetch_node'
      // Only move further to create an IdentityN node
      // if it is greater than 0
      // Also, make sure that none of the output types is
      // a ref type because IdentityN does not support
      // an input of type ref type
      if (node->num_outputs() > 0) {
        set<int> output_slots = fetch_node_info.at(node->name());
        if (output_slots.size() == 0) {
          return errors::Internal(node->name(), " is an output node with ", node->num_outputs(), " outputs, but none of them were requested in fetch_node_info");
        }
        std::vector<NodeBuilder::NodeOut> inputs;
        std::vector<DataType> input_types;
        for (const int& i : output_slots) {
          if (IsRefType(node->output_type(i))) {
            NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: "
                           << "Datatype for the node output"
                           << " at index " << i << " is ref type";
            ref_type = true;
            break;
          }
          input_types.push_back(node->output_type(i));
          inputs.push_back(NodeBuilder::NodeOut(node, i));
        }

        if (ref_type) {
          NGRAPH_VLOG(5)
              << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
          continue;
        }

        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Creating an IdentityN node";
        Node* identityN_node;
        TF_RETURN_IF_ERROR(NodeBuilder(node->name(), "IdentityN")
                               .Attr("T", input_types)
                               .Input(inputs)
                               .Device(node->assigned_device_name())
                               .Finalize(input_graph, &identityN_node));

        identityN_node->set_assigned_device_name(node->assigned_device_name());

        // Rename the skip node
        // Get a new name for the node with the given prefix
        // We will use the 'original-node-name_ngraph' as the prefix
        string new_name = input_graph->NewName(node->name() + "_ngraph");
        // TODO: Use (guaranteed) unique name here
        node->set_name(new_name);
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: New name for fetch node "
                       << node->name();
      } else {
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: num outputs " << node->num_outputs();
        NGRAPH_VLOG(5) << "NGTF_OPTIMIZER: Cannot construct an IdentityN node";
      }
    }
  }
  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
