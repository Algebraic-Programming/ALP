
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 */

/**
 * @dir include/graphblas/utils/telemetry
 * This folder contains all telemetry functionalities, i.e., those meant to measure
 * and report code execution in detail. They are designed with two goals in mind:
 *   -# <b>compile-time control</b>: all functionalities can be activated or deactivated
 * 		at compile-time; if deactivated, they incur no runtime and memory cost
 *   -# <b>fine granularity</b>: since telemetry is complex and very application-specific,
 * 		they allow fine-grained measurement and reporting; hence, they are also meant
 * 		to be conveniently integrated into an existing application at fine granularity
 *   -# <b>no pre-processor cluttering</b>: multiple specializations may exist for the same functionality,
 * 		for example to avoid memory or runtime costs if telemetry is deactivated; all
 * 		implementations \b must compile against the same code paths, to avoid verbose
 * 		insertion of #ifdef or similar directives on user's behalf.
 *
 * See the documentation of TelemetryController.hpp for some basic examples.
 */

/**
 * @file OutputStream.hpp
 * @author Alberto Scolari (alberto.scolar@huawei.com)
 *
 * Convenience all-include header for all telemetry-related functionalities.
 */

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY

#include "TelemetryController.hpp"
#include "Stopwatch.hpp"
#include "Timeable.hpp"
#include "CSVWriter.hpp"
#include "OutputStream.hpp"

#endif // _H_GRB_UTILS_TELEMETRY_TELEMETRY
