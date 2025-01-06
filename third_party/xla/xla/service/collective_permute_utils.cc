/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/collective_permute_utils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/graphcycles/graphcycles.h"

namespace xla {
namespace cp_utils {

using xla::HloCollectivePermuteInstruction;

std::string SourceTargetPairsString(const HloCollectivePermuteInstruction& cp) {
  auto formatter = absl::PairFormatter(
      [](std::string* out, int64_t value) { absl::StrAppend(out, "{", value); },
      ",",
      [](std::string* out, int64_t value) {
        absl::StrAppend(out, value, "}");
      });
  std::string pairs_str =
      absl::StrJoin(cp.source_target_pairs(), ",", formatter);
  return absl::StrCat("{", pairs_str, "}");
}

bool HasCycles(const SourceTargetPairs& pairs) {
  // Build a direct graph to check for cycles in (source, target) relationship.
  GraphCycles graph;

  // Map replica numbers to graph node ids.
  absl::flat_hash_map<int64_t, int32_t> replica_to_node_id;
  auto get_node_id = [&](int64_t replica) {
    auto it_and_inserted = replica_to_node_id.emplace(replica, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      // First time to see the replica, create a node for it.
      it->second = graph.NewNode();
    }
    return it->second;
  };

  for (auto pair : pairs) {
    int source = get_node_id(pair.first);
    int target = get_node_id(pair.second);
    VLOG(3) << "See source " << source << " -> target " << target;
    if (!graph.InsertEdge(source, target)) {
      VLOG(3) << "Detected cycles";
      return true;
    }
  }
  return false;
}

// TODO remove backed in assumptions that pairs are ordered and start with 0.
bool IsForwardCycle(const SourceTargetPair& backedge,
                    const SourceTargetPairs& others) {
  int64_t num_pairs = others.size() + 1;
  if (backedge.first != num_pairs - 1 || backedge.second != 0) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i || pair.second != i + 1) {
      return false;
    }
  }
  return true;
}

bool IsBackwardCycle(const SourceTargetPair& backedge,
                     const SourceTargetPairs& others) {
  int64_t num_pairs = others.size() + 1;
  if (backedge.first != 0 || backedge.second != num_pairs - 1) {
    return false;
  }
  for (int64_t i = 0; i < num_pairs - 1; ++i) {
    const SourceTargetPair& pair = others[i];
    if (pair.first != i + 1 || pair.second != i) {
      return false;
    }
  }
  return true;
}

}  // namespace cp_utils
}  // namespace xla
