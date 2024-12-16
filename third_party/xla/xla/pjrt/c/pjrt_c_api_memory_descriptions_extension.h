/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_

#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_MEMORY_DESCRIPTIONS_EXTENSION_VERSION 0

typedef struct PJRT_MemoryDescription PJRT_MemoryDescription;

struct PJRT_DeviceDescription_MemorySpaces_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const PJRT_MemoryDescription* const* memory_spaces;  // out
  size_t num_memory_spaces;                            // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_MemorySpaces_Args,
                          num_memory_spaces);

// Returns all memory spaces attached to this device.
// The memory spaces are in no particular order.
typedef PJRT_Error* PJRT_DeviceDescription_MemorySpaces(
    PJRT_DeviceDescription_MemorySpaces_Args* args);

struct PJRT_MemoryDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_MemoryDescription* memory_description;
  // `kind` has same lifetime as `memory_description`.
  const char* kind;  // out
  size_t kind_size;  // out
  int kind_id;       // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MemoryDescription_Kind_Args, kind);

// Return the kind of a given memory space description. This is a
// platform-dependent string that uniquely identifies the kind of
// memory space. We also return a `kind_id` that is unique among
// memory spaces attached to the same client.
typedef PJRT_Error* PJRT_MemoryDescription_Kind(
    PJRT_MemoryDescription_Kind_Args* args);

typedef struct PJRT_MemoryDescriptions_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;
  PJRT_Extension_Base* next;
  PJRT_DeviceDescription_MemorySpaces* PJRT_DeviceDescription_MemorySpaces;
  PJRT_MemoryDescription_Kind* PJRT_MemoryDescription_Kind;
} PJRT_MemoryDescriptions_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MemoryDescriptions_Extension,
                          PJRT_MemoryDescription_Kind);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_
