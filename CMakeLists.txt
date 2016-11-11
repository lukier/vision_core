# Copyright (c) 2016, Robert Lukierski.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)

# ------------------------------------------------------------------------------
# Default build configuration
# ------------------------------------------------------------------------------
if(DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Build type.")
else()
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type.")
endif()

# ------------------------------------------------------------------------------
# Project name & version
# ------------------------------------------------------------------------------
project(VisionCore)
set(${PROJECT_NAME}_VERSION 1.0.0)

# ------------------------------------------------------------------------------
# Enable C++11 and high level warnings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -pedantic -Wno-unused-parameter -Wno-unused-variable)

# ------------------------------------------------------------------------------
# Ready for testing
# ------------------------------------------------------------------------------
option(BUILD_TESTS "Enable to build unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
    
  # Code Coverage Build Type
  set(CMAKE_CXX_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by the C++ compiler during coverage builds." FORCE)
  set(CMAKE_C_FLAGS_COVERAGE "-g -O0 --coverage -fprofile-arcs -ftest-coverage" CACHE STRING "Flags used by the C compiler during coverage builds." FORCE )
  set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "" CACHE STRING "Flags used for linking binaries during coverage builds." FORCE )
  set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "" CACHE STRING "Flags used by the shared libraries linker during coverage builds." FORCE)
  mark_as_advanced(CMAKE_CXX_FLAGS_COVERAGE CMAKE_C_FLAGS_COVERAGE CMAKE_EXE_LINKER_FLAGS_COVERAGE CMAKE_SHARED_LINKER_FLAGS_COVERAGE)
endif()

include(GenerateExportHeader)

# ---------------------------------------------
# Where to look for Find*.cmake
# ---------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts)

# ------------------------------------------------------------------------------
# Print Project Info
# ------------------------------------------------------------------------------
message("Project: ${PROJECT_NAME} / ${${PROJECT_NAME}_VERSION}, build type: ${CMAKE_BUILD_TYPE}, compiled on: ${CMAKE_SYSTEM}, flags: ${CMAKE_CXX_FLAGS}")

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED QUIET)
find_package(Sophus REQUIRED QUIET)
find_package(TBB REQUIRED QUIET)

find_package(CUDA QUIET)
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS Auto)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;-std=c++11;-use_fast_math;-O2;-Xcompiler;-rdynamic)
find_package(OpenCL QUIET)

find_package(OpenCV QUIET)
find_package(Ceres QUIET)
find_package(cereal QUIET)

if(NOT FORCE_NO_GLBINDING)
    find_package(glbinding REQUIRED QUIET)
else()
    message("VisionCore: GLBinding disabled")
    find_package(OpenGL REQUIRED QUIET)
    find_package(GLEW REQUIRED QUIET)
endif()

find_package(Vulkan QUIET)
find_package(FREEGLUT QUIET)
find_package(X11 QUIET)

# ------------------------------------------------------------------------------
# Our headers
# ------------------------------------------------------------------------------
include_directories(.)

# ---------------------------------------------
# List of sources
# ---------------------------------------------
set(HEADERS
CUDAException.hpp
CUDATypes.hpp
EigenSerializers.hpp
LaunchUtils.hpp
MemoryPolicyCUDA.hpp
MemoryPolicyHost.hpp
MemoryPolicy.hpp
MemoryPolicyOpenCL.hpp
Platform.hpp
SophusInterpolations.hpp
buffers/Buffer1D.hpp
buffers/Buffer2D.hpp
buffers/Buffer3D.hpp
buffers/BufferPyramid.hpp
buffers/CUDATexture.hpp
buffers/GPUVariable.hpp
buffers/Image2D.hpp
buffers/ImagePyramid.hpp
buffers/PyramidBase.hpp
buffers/ReductionSum2D.hpp
buffers/Volume.hpp
control/PID.hpp
control/VelocityProfile.hpp
image/ImagePatch.hpp
image/PixelConvert.hpp
math/Angles.hpp
math/DenavitHartenberg.hpp
math/Divergence.hpp
math/Fitting.hpp
math/HammingDistance.hpp
math/Kalman.hpp
math/LeastSquares.hpp
math/LiangBarsky.hpp
math/LocalParamSE3.hpp
math/LossFunctions.hpp
math/PolarSpherical.hpp
math/Random.hpp
math/RANSAC.hpp
math/Statistics.hpp
types/AxisAlignedBoundingBox.hpp
types/CostVolumeElement.hpp
types/Endian.hpp
types/Gaussian.hpp
types/Hypersphere.hpp
types/Polynomial.hpp
types/Rectangle.hpp
types/SDF.hpp
wrapgl/WrapGLBuffer.hpp
wrapgl/WrapGLContext.hpp
wrapgl/WrapGLFramebuffer.hpp
wrapgl/WrapGLShader.hpp
wrapgl/WrapGLTexture.hpp
wrapgl/WrapGLVertexArrayObject.hpp
wrapgl/WrapGLCommon.hpp
)

set(SOURCES
sources/VisionCore.cpp
wrapgl/sources/WrapGLCommon.cpp
wrapgl/sources/WrapGLFramebuffer.cpp
wrapgl/sources/WrapGLBuffer.cpp
)

if(glbinding_FOUND)
    if(FREEGLUT_FOUND)
        list(APPEND SOURCES wrapgl/sources/context_glut.cpp)
    endif()

    if(X11_FOUND)
        list(APPEND SOURCES wrapgl/sources/context_x11.cpp wrapgl/sources/context_windowless.cpp)
    endif()
endif()

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:>
    PRIVATE sources)
    
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${Sophus_INCLUDE_DIRS})

if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC CORE_HAVE_CUDA)
endif()
if(OPENCL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCL_LIBRARY})
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIR})
    target_compile_definitions(${PROJECT_NAME} PUBLIC CORE_HAVE_OPENCL CL_HPP_MINIMUM_OPENCL_VERSION=120 CL_HPP_TARGET_OPENCL_VERSION=${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0 CL_HPP_ENABLE_EXCEPTIONS)
endif()
if(TBB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TBB_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${TBB_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC CORE_HAVE_TBB ${TBB_DEFINITIONS})
endif()
if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC CORE_HAVE_OPENCV)
endif()
if(cereal_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC cereal)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CORE_HAVE_CEREAL)
endif()
if(CERES_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${CERES_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PUBLIC ${CERES_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC ${CERES_DEFINITIONS} CORE_HAVE_CERES)
endif()
if(glbinding_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC WRAPGL_HAVE_GLBINDING)
    target_link_libraries(${PROJECT_NAME} PUBLIC glbinding)
else()
    target_include_directories(${PROJECT_NAME} PUBLIC ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES})
endif()
if(FREEGLUT_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FREEGLUT_LIBRARIES})
endif()
if(X11_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_LIBRARIES})
endif()
if(VULKAN_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${VULKAN_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${VULKAN_LIBRARY})
endif()

# ------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------
if(BUILD_TESTS)
    add_subdirectory(tests)
    add_subdirectory(buffers/tests)
    add_subdirectory(control/tests)
    add_subdirectory(image/tests)
    add_subdirectory(math/tests)
    add_subdirectory(types/tests)
    add_subdirectory(wrapgl/tests)
endif()

# ------------------------------------------------------------------------------
# Installation - library
# ------------------------------------------------------------------------------
generate_export_header(${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${${PROJECT_NAME}_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION 1)
set_property(TARGET ${PROJECT_NAME} PROPERTY INTERFACE_${PROJECT_NAME}_MAJOR_VERSION 1)
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include)
install(FILES ${HEADERS} "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_export.h" DESTINATION include)

# ------------------------------------------------------------------------------
# Installation - generate version file
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${${PROJECT_NAME}_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# ------------------------------------------------------------------------------
# Installation - Export Targets
# ------------------------------------------------------------------------------
export(EXPORT ${PROJECT_NAME}Targets FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake") 
set(VISIONCORE_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
set(FORCE_NO_GLBINDING ${FORCE_NO_GLBINDING})
configure_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
set(ConfigPackageLocation lib/cmake/${PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE ${PROJECT_NAME}Targets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES ${PROJECT_NAME}Config.cmake "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" DESTINATION ${ConfigPackageLocation})
export(PACKAGE ${PROJECT_NAME})
