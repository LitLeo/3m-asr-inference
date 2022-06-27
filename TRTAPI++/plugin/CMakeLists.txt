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

add_custom_target(plugin)

set(TARGET_NAME trtplugin++)
set(SHARED_TARGET ${TARGET_NAME})
set(STATIC_TARGET ${TARGET_NAME}_static)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_EXPORT_MAP ${TARGET_DIR}/exports.map)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)

include_directories(common)
#include_directories(common common/kernels)

foreach(PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach(PLUGIN_ITER)

# Add common
add_subdirectory(common)

# Set gencodes
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${GENCODES})
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")
if (BERT_CU_SOURCES)
    set_source_files_properties(${BERT_CU_SOURCES} PROPERTIES COMPILE_FLAGS ${BERT_GENCODES})
    list(APPEND PLUGIN_SOURCES "${BERT_CU_SOURCES}")
endif()

#list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/infer_plugin.cpp")
#list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/trt_plugin_creator.cpp")
list(APPEND PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/trt_plugin_plus.cpp")

#message(${PLUGIN_SOURCES})

################################## SHARED LIBRARY #######################################

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
    ${PLUGIN_CU_SOURCES}
)

target_include_directories(${SHARED_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)

set_target_properties(${SHARED_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
)

set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")

set_target_properties(${SHARED_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${SHARED_TARGET} PROPERTIES VERSION ${TRT_PLUGIN_VERSION} SOVERSION ${TRT_PLUGIN_MAJOR} )

set_property(TARGET ${SHARED_TARGET} PROPERTY CUDA_STANDARD 14)

target_link_libraries(${SHARED_TARGET}
    nvinfer
    cudart cublas cublasLt cudnn
    ${CMAKE_DL_LIBS}
    pthread
)

if (BUILD_LIBTORCH_PLUGINS)
    target_link_libraries(${SHARED_TARGET}
        ${TORCH_LIBRARIES}
    )
endif()

add_dependencies(plugin ${STATIC_TARGET} ${SHARED_TARGET})

################################## STATIC LIBRARY #######################################

add_library(${STATIC_TARGET} STATIC
    ${PLUGIN_SOURCES}
    ${PLUGIN_CU_SOURCES}
)

target_include_directories(${STATIC_TARGET}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
    PRIVATE ${TORCH_INCLUDE_DIRS}
)

set_target_properties(${STATIC_TARGET} PROPERTIES
    CXX_STANDARD "14"
    CXX_STANDARD_REQUIRED "YES"
    CXX_EXTENSIONS "NO"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_OUT_DIR}"
)

set_target_properties(${STATIC_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(${STATIC_TARGET} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${STATIC_TARGET} PROPERTIES VERSION ${TRT_PLUGIN_VERSION} SOVERSION ${TRT_PLUGIN_MAJOR} )

set_property(TARGET ${STATIC_TARGET} PROPERTY CUDA_STANDARD 14)

add_dependencies(plugin ${STATIC_TARGET} ${SHARED_TARGET})

#########################################################################################


################################### INSTALLATION ########################################
install(TARGETS ${SHARED_TARGET} ${STATIC_TARGET}
    RUNTIME DESTINATION ${INSTALL_DIR}
    LIBRARY DESTINATION ${INSTALL_DIR}
    ARCHIVE DESTINATION ${INSTALL_DIR}
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/trt_plugin_plus.h DESTINATION ${INSTALL_DIR} COMPONENT devel)
