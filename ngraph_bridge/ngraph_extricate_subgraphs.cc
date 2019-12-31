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

#include "ngraph_bridge/ngraph_extricate_subgraphs.h"

using namespace std;

namespace tensorflow {

namespace ngraph_bridge {

Status ExtricateSubgraph(Graph* graph, SubgraphExtractionResults& results) {
  std::map<int, std::string>& device_name_map = results.device_name_map;
  std::map<int, std::string>& backend_name_map = results.backend_name_map;
  std::map<std::tuple<int, int>, std::tuple<int, int>>& output_remap_map =
      results.output_remap_map;
  std::map<std::tuple<int, std::string, int>, string>& input_rename_map =
      results.input_rename_map;

  std::map<std::tuple<int, int, int>, int> input_remap_map;

  std::map<int, std::vector<DataType>>& cluster_output_dt_map =
      results.cluster_output_dt_map;

  std::map<int, std::vector<std::tuple<int, int, DataType>>>&
      cluster_input_map = results.cluster_input_map;

  // Pass 1: Populate the cluster-index-to-device name map for each existing
  // cluster. PIGGYBACKING BACKEND TEST HERE, THEY WILL GET COMBINED INTO ONE
  for (auto node : graph->op_nodes()) {
    int cluster_idx;

    if (GetNodeCluster(node, &cluster_idx) != Status::OK()) {
      continue;
    }

    string node_backend;
    if (GetNodeBackend(node, &node_backend) != Status::OK()) {
      continue;
    }

    auto it = device_name_map.find(cluster_idx);

    if (it != device_name_map.end()) {
      if (it->second != node->assigned_device_name()) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned device " << node->assigned_device_name()
               << " but another node with assigned device " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested device to '" << node->assigned_device_name()
                     << "'";
      device_name_map[cluster_idx] = node->assigned_device_name();
    }

    auto itr = backend_name_map.find(cluster_idx);

    if (itr != backend_name_map.end()) {
      if (itr->second != node_backend) {
        std::stringstream ss_err;
        ss_err << "Node " << node->name() << " in cluster " << cluster_idx
               << " has assigned backend " << node_backend
               << " but another node with assigned backend " << it->second
               << " has already been seen in the same cluster";

        return errors::Internal(ss_err.str());
      }
    } else {
      NGRAPH_VLOG(3) << "setting cluster " << cluster_idx
                     << " requested backend to '" << node_backend << "'";
      backend_name_map[cluster_idx] = node_backend;
    }
  }

  // Pass 2: Find all nodes that are feeding into/out of each cluster, and
  // add inputs for them to the corresponding FunctionDef(s).
  std::map<int, int> retval_index_count;
  std::map<int, int> arg_index_count;
  int count_arg = 0, count_retval = 0, count_both_arg_retval = 0,
      count_free = 0, count_encapsulated = 0, count_tot = 0;
  for (auto edge : graph->edges()) {
    count_tot++;
    // TODO(amprocte): should actually keep of these. During clustering we
    // will already have identified any intra-cluster control deps. Should
    // maintain inter-cluster control deps.
    if (edge->IsControlEdge()) {
      count_free++;
      continue;
    }

    Node* src = edge->src();
    Node* dst = edge->dst();

    // TODO(amprocte): the following rejects edges involving source/sink. Is
    // that what we want to do?
    if (!src->IsOp() || !dst->IsOp()) {
      count_free++;
      continue;
    }

    int dst_cluster_idx;
    bool dst_clustered =
        (GetNodeCluster(dst, &dst_cluster_idx) == Status::OK());

    int src_cluster_idx;
    bool src_clustered =
        (GetNodeCluster(src, &src_cluster_idx) == Status::OK());

    // Ignore edges within a cluster. (Note that this test also works when
    // both nodes are unclustered; GetNodeCluster gives us -1 in that case.
    if (dst_cluster_idx == src_cluster_idx) {
      count_encapsulated++;
      continue;
    }

    // Some debug logging...
    DataType dt = dst->input_type(edge->dst_input());
    std::string flow_kind = dst_clustered && src_clustered
                                ? "cross-flow"
                                : dst_clustered ? "in-flow" : "out-flow";

    NGRAPH_VLOG(4) << "found " << flow_kind << ": " << src->name() << "["
                   << edge->src_output() << "] in " << src_cluster_idx << " to "
                   << dst->name() << "[" << edge->dst_input() << "] in "
                   << dst_cluster_idx << ", datatype: " << dt;

    bool edge_is_retval = false, edge_is_arg = false;

    // If the source node lies within a cluster, we must create an output for
    // it from the source cluster. For the moment we will just store this
    // fact in the output_remap_map.
    if (src_clustered &&
        output_remap_map.find(std::make_tuple(src->id(), edge->src_output())) ==
            output_remap_map.end()) {
      output_remap_map[std::make_tuple(src->id(), edge->src_output())] =
          std::make_tuple(src_cluster_idx,
                          cluster_output_dt_map[src_cluster_idx].size());

      std::stringstream ss;
      ss << "ngraph_output_" << cluster_output_dt_map[src_cluster_idx].size();
      string output_name = ss.str();
      auto new_output_node_def =
          NGraphClusterManager::GetClusterGraph(src_cluster_idx)->add_node();
      new_output_node_def->set_name(output_name);
      new_output_node_def->set_op("_Retval");
      edge_is_retval = true;

      std::stringstream ss_input_to_retval;
      ss_input_to_retval << src->name() << ":" << edge->src_output();

      new_output_node_def->add_input(ss_input_to_retval.str());

      SetAttrValue(dt, &((*(new_output_node_def->mutable_attr()))["T"]));
      SetAttrValue(retval_index_count[src_cluster_idx],
                   &((*(new_output_node_def->mutable_attr()))["index"]));

      retval_index_count[src_cluster_idx]++;

      cluster_output_dt_map[src_cluster_idx].push_back(dt);
    }

    // If the destination node lies within a cluster, we must create an input
    // for the source node to the destination cluster. For the moment we will
    // just store this fact in the input_remap_map.
    if (dst_clustered &&
        input_remap_map.find(
            std::make_tuple(dst_cluster_idx, src->id(), edge->src_output())) ==
            input_remap_map.end()) {
      input_remap_map[std::make_tuple(dst_cluster_idx, src->id(),
                                      edge->src_output())] =
          cluster_input_map[dst_cluster_idx].size();

      std::stringstream ss;
      ss << "ngraph_input_" << cluster_input_map[dst_cluster_idx].size();
      std::string new_input_name = ss.str();

      input_rename_map[std::make_tuple(dst_cluster_idx, src->name(),
                                       edge->src_output())] = new_input_name;
      string input_prov_tag = src->name();

      auto new_input_node_def =
          NGraphClusterManager::GetClusterGraph(dst_cluster_idx)->add_node();
      new_input_node_def->set_name(new_input_name);
      new_input_node_def->set_op("_Arg");
      edge_is_arg = true;

      SetAttrValue(dt, &((*(new_input_node_def->mutable_attr()))["T"]));
      SetAttrValue(arg_index_count[dst_cluster_idx],
                   &((*(new_input_node_def->mutable_attr()))["index"]));
      SetAttrValue(input_prov_tag,
                   &((*(new_input_node_def->mutable_attr()))["_prov_tag"]));

      arg_index_count[dst_cluster_idx]++;

      cluster_input_map[dst_cluster_idx].push_back(
          std::make_tuple(src->id(), edge->src_output(), dt));
    }

    if (config::IsLoggingPlacement()) {
      if (edge_is_arg && edge_is_retval) {
        count_both_arg_retval++;
      } else {
        if (edge_is_arg) {
          count_arg++;
        } else {
          count_retval++;
        }
      }
    }
  }

  if (config::IsLoggingPlacement()) {
    int computed_edge_number = count_arg + count_retval +
                               count_both_arg_retval + count_free +
                               count_encapsulated;
    std::cout << "NGTF_SUMMARY: Types of edges:: args: " << count_arg
              << ", retvals: " << count_retval
              << ", both arg and retval: " << count_both_arg_retval
              << ", free: " << count_free
              << ", encapsulated: " << count_encapsulated
              << ", total: " << count_tot
              << ", computed total: " << computed_edge_number << endl;
    std::cout << "\n=============Ending sub-graph logs=============\n";
    if (!(computed_edge_number == count_tot &&
          count_tot == graph->num_edges())) {
      return errors::Internal("Computed number of edges ", computed_edge_number,
                              " and counted number of edges ", count_tot,
                              " and number of edges from querying TF api ",
                              graph->num_edges(), " do not match up\n");
    }
  }

  return Status::OK();
}

}  // namespace ngraph_bridge
}  // namespace tensorflow
