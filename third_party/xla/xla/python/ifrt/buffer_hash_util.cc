/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/buffer_hash_util.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace ifrt {

absl::StatusOr<uint64_t> HashBufferPhysical(absl::Span<const char> buffer,
                                            const xla::Layout& layout) {
  // Take the whole buffer and compute the hash.
  uint64_t hash =
      tsl::Fingerprint64(absl::string_view(buffer.data(), buffer.size()));

  // Combine with the layout hash.
  xla::LayoutProto layout_proto = layout.ToProto();
  std::string serialized_layout = layout_proto.SerializeAsString();
  uint64_t layout_hash = tsl::Fingerprint64(serialized_layout);
  return tsl::FingerprintCat64(hash, layout_hash);
}

absl::StatusOr<uint64_t> HashBufferLogical(absl::Span<const char> buffer,
                                           int64_t element_byte_size,
                                           const IndexDomain& index_domain) {
  // Each element is hashed with its coordinate. The resulting per-element hash
  // values can be combined using XOR in a commutative and associative way.
  const Shape& shard_shape = index_domain.shape();
  const int64_t num_elements = shard_shape.num_elements();
  if (buffer.size() != element_byte_size * num_elements) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Buffer byte size does not match the expected size. buffer byte size: ",
        buffer.size(), " vs. element byte size: ", element_byte_size,
        " and index domain: ", index_domain));
  }

  const Index& origin = index_domain.origin();
  const int num_dims = shard_shape.dims().size();

  uint64_t hash = 0;
  // Coordinates in the array.
  std::vector<int64_t> dims(origin.elements().begin(), origin.elements().end());
  std::vector<int64_t> limits(num_dims);
  for (int d = 0; d < num_dims; ++d) {
    limits[d] = origin.elements()[d] + shard_shape.dims()[d];
  }

  for (int64_t k = 0; k < num_elements; ++k) {
    uint64_t coord_hash = tsl::Fingerprint64(
        absl::string_view(reinterpret_cast<const char*>(dims.data()),
                          num_dims * sizeof(int64_t)));

    absl::string_view element_bytes(buffer.data() + k * element_byte_size,
                                    element_byte_size);
    uint64_t element_hash = tsl::Fingerprint64(element_bytes);

    uint64_t combined = tsl::FingerprintCat64(coord_hash, element_hash);
    hash ^= combined;

    // Increment the multi-dimensional index in the array.
    for (int d = num_dims - 1; d >= 0; --d) {
      ++dims[d];
      if (dims[d] < limits[d]) {
        break;
      }
      dims[d] = origin.elements()[d];
    }
  }
  return hash;
}

std::vector<int> GetReplicaGroupIds(
    absl::Span<const IndexDomain> index_domains) {
  std::vector<int> replica_group_ids;
  replica_group_ids.reserve(index_domains.size());
  absl::flat_hash_map<IndexDomain, int> domain_to_group_id;
  for (const IndexDomain& index_domain : index_domains) {
    auto it = domain_to_group_id.find(index_domain);
    if (it == domain_to_group_id.end()) {
      it = domain_to_group_id.insert({index_domain, domain_to_group_id.size()})
               .first;
    }
    replica_group_ids.push_back(it->second);
  }
  return replica_group_ids;
}

absl::StatusOr<uint64_t> AggregateShardHashes(
    absl::Span<const uint64_t> shard_hashes,
    absl::Span<const int> replica_group_ids, Client::HashMode mode) {
  if (shard_hashes.size() != replica_group_ids.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Shard hashes and replica group IDs must have the same size, but have ",
        shard_hashes.size(), " vs. ", replica_group_ids.size()));
  }

  std::map<int, uint64_t> group_id_to_hash;
  for (int i = 0; i < shard_hashes.size(); ++i) {
    int group_id = replica_group_ids[i];
    auto it = group_id_to_hash.find(group_id);
    if (it == group_id_to_hash.end()) {
      group_id_to_hash.insert({group_id, shard_hashes[i]});
    } else {
      if (it->second != shard_hashes[i]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Hash mismatch across replicas for replica group ", group_id));
      }
    }
  }

  uint64_t aggregated_hash = 0;
  switch (mode) {
    case Client::HashMode::kPhysical: {
      for (const auto& [_, shard_hash] : group_id_to_hash) {
        aggregated_hash = tsl::FingerprintCat64(aggregated_hash, shard_hash);
      }
      break;
    }
    case Client::HashMode::kLogical: {
      for (const auto& [_, shard_hash] : group_id_to_hash) {
        aggregated_hash ^= shard_hash;
      }
      break;
    }
  }
  return aggregated_hash;
}

}  // namespace ifrt
}  // namespace xla
