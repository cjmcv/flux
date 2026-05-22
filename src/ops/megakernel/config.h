/* Copyright 2023-2024 CMU
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

#pragma once
#include <cstddef>
#include <cstdint>

namespace megakernel {
namespace config {

size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096;
constexpr int MAX_TENSOR_DIMS = 4;
constexpr int MAX_TMA_DESC_PER_TENSOR = 3;

} // namespace config
} // namespace megakernel
