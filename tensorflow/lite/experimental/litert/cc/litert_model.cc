// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"

namespace litert {

bool Tensor::IsSubgraphOutput() const { return Uses().empty(); }

bool Tensor::IsSubgraphInput() const {
  return !HasWeights() && !DefiningOp().has_value();
}

bool Tensor::IsConstant() const {
  return HasWeights() && !DefiningOp().has_value();
}

SmallVec<Tensor::TensorUse> Tensor::Uses() const {
  LiteRtParamIndex num_uses;
  litert::internal::AssertOk(LiteRtGetNumTensorUses, Get(), &num_uses);

  SmallVec<Tensor::TensorUse> uses;
  for (auto i = 0; i < num_uses; ++i) {
    LiteRtOp user;
    LiteRtParamIndex user_arg_index;
    litert::internal::AssertOk(LiteRtGetTensorUse, Get(), i, &user,
                               &user_arg_index);
    uses.emplace_back(Op(user), user_arg_index);
  }
  return uses;
}

SmallVec<Tensor> Op::Inputs() const {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumOpInputs, Get(), &num_inputs);

  SmallVec<Tensor> inputs;
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensor input;
    internal::AssertOk(LiteRtGetOpInput, Get(), i, &input);
    inputs.emplace_back(Tensor(input));
  }
  return inputs;
}

SmallVec<Tensor> Op::Outputs() const {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumOpOutputs, Get(), &num_outputs);

  SmallVec<Tensor> outputs;
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensor output;
    internal::AssertOk(LiteRtGetOpOutput, Get(), i, &output);
    outputs.emplace_back(Tensor(output));
  }
  return outputs;
}

SmallVec<Tensor> Subgraph::Inputs() const {
  LiteRtParamIndex num_inputs;
  internal::AssertOk(LiteRtGetNumSubgraphInputs, Get(), &num_inputs);

  SmallVec<Tensor> inputs;
  for (auto i = 0; i < num_inputs; ++i) {
    LiteRtTensor input;
    internal::AssertOk(LiteRtGetSubgraphInput, Get(), i, &input);
    inputs.emplace_back(Tensor(input));
  }
  return inputs;
}

SmallVec<Tensor> Subgraph::Outputs() const {
  LiteRtParamIndex num_outputs;
  internal::AssertOk(LiteRtGetNumSubgraphOutputs, Get(), &num_outputs);

  SmallVec<Tensor> outputs;
  for (auto i = 0; i < num_outputs; ++i) {
    LiteRtTensor output;
    internal::AssertOk(LiteRtGetSubgraphOutput, Get(), i, &output);
    outputs.emplace_back(Tensor(output));
  }
  return outputs;
}

std::vector<Op> Subgraph::Ops() const {
  LiteRtParamIndex num_ops;
  internal::AssertOk(LiteRtGetNumSubgraphOps, Get(), &num_ops);

  std::vector<Op> ops;
  for (auto i = 0; i < num_ops; ++i) {
    LiteRtOp op;
    litert::internal::AssertOk(LiteRtGetSubgraphOp, Get(), i, &op);
    ops.emplace_back(Op(op));
  }
  return ops;
}

}  // namespace litert
