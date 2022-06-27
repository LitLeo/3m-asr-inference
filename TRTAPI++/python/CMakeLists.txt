# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

add_custom_target(python)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/trt_helper DESTINATION ${INSTALL_DIR} COMPONENT devel)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/trt_helper/version.py "__version__=\"${TRT_PLUGIN_VERSION}_${TRT_PLUGIN_BUILD}\"")
