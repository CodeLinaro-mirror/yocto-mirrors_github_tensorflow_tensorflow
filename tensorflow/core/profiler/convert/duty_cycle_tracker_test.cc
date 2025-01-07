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

#include "xla/tsl/profiler/utils/timespan.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::Timespan;

TEST(DutyCycleTrackerTest, TimeIntervalsTest) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(0, 10), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);
  EXPECT_EQ(tracker.GetActiveTime(), 20);
  EXPECT_EQ(tracker.GetIdleTime(), 10);
  EXPECT_EQ(tracker.GetDuration(), 30);
}

TEST(DutyCycleTrackerTest, UnionTest) {
  DutyCycleTracker tracker;
  tracker.AddInterval(Timespan::FromEndPoints(0, 10), true);
  tracker.AddInterval(Timespan::FromEndPoints(20, 30), true);

  DutyCycleTracker other_tracker;
  other_tracker.AddInterval(Timespan::FromEndPoints(10, 20), true);
  other_tracker.AddInterval(Timespan::FromEndPoints(30, 40), true);

  tracker.Union(other_tracker);
  EXPECT_EQ(tracker.GetActiveTime(), 40);
  EXPECT_EQ(tracker.GetIdleTime(), 0);
  EXPECT_EQ(tracker.GetDuration(), 40);
}

TEST(DutyCycleTrackerTest, ActiveTimeTest) {
  DutyCycleTracker tracker;
  EXPECT_EQ(tracker.GetActiveTime(), 0);
  tracker.AddInterval(Timespan::FromEndPoints(0, 10), true);
  EXPECT_EQ(tracker.GetActiveTime(), 10);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
