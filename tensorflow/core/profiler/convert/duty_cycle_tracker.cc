/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/duty_cycle_tracker.h"

#include <iterator>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "xla/tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

using tsl::profiler::Timespan;

template <typename Func>
void DutyCycleTracker::MergeOverlappingTimespans(
    ActiveTimeSpans::const_iterator begin, Func should_continue) {
  if (begin == active_time_spans_.end()) return;
  ActiveTimeSpans::const_iterator prev = begin, curr = std::next(begin);
  while (curr != active_time_spans_.end() && should_continue(*curr)) {
    if (prev->end_ps() > curr->begin_ps()) {
      if (prev->end_ps() < curr->end_ps()) {
        Timespan merged =
            Timespan::FromEndPoints(prev->begin_ps(), curr->end_ps());
        prev =
            active_time_spans_.insert(active_time_spans_.erase(prev), merged);
        curr = std::next(prev);
      }
      curr = active_time_spans_.erase(curr);
      prev = std::prev(curr);
    } else {
      prev = curr++;
    }
  }
}

void DutyCycleTracker::AddInterval(tsl::profiler::Timespan time_span,
                                   bool is_active) {
  if (is_active) {
    auto [it, inserted] = active_time_spans_.insert(time_span);
    if (!inserted) {
      // Exact timespan already exists.
      return;
    }
    // Start from timespan before the inserted one, if any.
    if (it != active_time_spans_.begin()) --it;
    TimePs end_ps(time_span.end_ps());
    MergeOverlappingTimespans(it, [&](const Timespan& timespan) {
      return timespan.begin_ps() <= end_ps;
    });
  }
  total_time_span_.ExpandToInclude(time_span);
}

void DutyCycleTracker::Union(const DutyCycleTracker& other) {
  active_time_spans_.insert(other.active_time_spans_.begin(),
                            other.active_time_spans_.end());
  MergeOverlappingTimespans(active_time_spans_.begin(),
                            [](const Timespan& timespan) { return true; });
  total_time_span_.ExpandToInclude(other.total_time_span_);
}

DutyCycleTracker::TimePs DutyCycleTracker::GetActiveTime() const {
  TimePs active_time_ps = TimePs(0);
  for (const auto& interval : active_time_spans_) {
    DCHECK(!interval.Empty());
    active_time_ps += interval.duration_ps();
  }
  return active_time_ps;
}

DutyCycleTracker::TimePs DutyCycleTracker::GetIdleTime() const {
  return TimePs(total_time_span_.duration_ps()) - GetActiveTime();
}
}  // namespace profiler
}  // namespace tensorflow
