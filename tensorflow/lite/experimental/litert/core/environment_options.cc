// Copyright 2025 Google LLC.
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

#include "tensorflow/lite/experimental/litert/core/environment_options.h"

#include <algorithm>
#include <string>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

litert::Expected<LiteRtAny> LiteRtEnvironmentOptionsT::GetOption(
    LiteRtEnvOptionTag tag) const {
  if (auto it = options_.find(tag); it != options_.end()) {
    return it->second;
  }
  return litert::Error(kLiteRtStatusErrorNotFound,
                       "Option was not set for this environment.");
}

litert::Expected<void> LiteRtEnvironmentOptionsT::SetOption(
    LiteRtEnvOption option) {
  if (option.value.type == kLiteRtAnyTypeString) {
    auto string_value_it = string_option_values_.end();
    if (auto it = options_.find(option.tag); it != options_.end()) {
      string_value_it =
          std::find_if(begin(string_option_values_), end(string_option_values_),
                       [&it](const std::string& str) {
                         return str.data() == it->second.str_value;
                       });
    }
    std::string& option_value = string_value_it != string_option_values_.end()
                                    ? *string_value_it
                                    : string_option_values_.emplace_back();
    option_value = option.value.str_value;

    LiteRtAny value{/*type=*/kLiteRtAnyTypeString};
    value.str_value = option_value.c_str();
    options_[option.tag] = value;
  } else {
    options_[option.tag] = option.value;
  }
  return {};
}
