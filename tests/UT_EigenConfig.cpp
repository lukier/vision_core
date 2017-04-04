/**
 * ****************************************************************************
 * Copyright (c) 2015, Robert Lukierski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * ****************************************************************************
 */

// system
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <exception>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <valarray>
#include <functional>

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <Platform.hpp>

#include <GetEigenConfig.hpp>

class Test_EigenConfig: public ::testing::Test
{
public:   
    Test_EigenConfig()
    {
        
    }
    
    virtual ~Test_EigenConfig()
    {
        
    }
    
    void reportEigen(float* data)
    {
        LOG(INFO) << "EIGEN_WORLD_VERSION = " << data[0];
        LOG(INFO) << "EIGEN_MAJOR_VERSION = " << data[1];
        LOG(INFO) << "EIGEN_MINOR_VERSION = " << data[2];
        LOG(INFO) << "EIGEN_COMP_GNUC = " << data[3];
        LOG(INFO) << "EIGEN_COMP_CLANG = " << data[4];
        LOG(INFO) << "EIGEN_COMP_LLVM = " << data[5];
        LOG(INFO) << "EIGEN_ARCH_x86_64 = " << data[6];
        LOG(INFO) << "EIGEN_ARCH_i386 = " << data[7];
        LOG(INFO) << "EIGEN_ARCH_ARM = " << data[8];
        LOG(INFO) << "EIGEN_ARCH_ARM64 = " << data[9];
        LOG(INFO) << "EIGEN_HAS_CONSTEXPR = " << data[10];
        LOG(INFO) << "EIGEN_IDEAL_MAX_ALIGN_BYTES = " << data[11];
        LOG(INFO) << "EIGEN_MAX_STATIC_ALIGN_BYTES = " << data[12];
        LOG(INFO) << "EIGEN_DONT_ALIGN = " << data[13];
        LOG(INFO) << "EIGEN_DONT_ALIGN_STATICALLY = " << data[14];
        LOG(INFO) << "EIGEN_MAX_ALIGN_BYTES = " << data[15];
        LOG(INFO) << "EIGEN_UNALIGNED_VECTORIZE = " << data[16];
        LOG(INFO) << "sizeof(Eigen::Vector2f) = " << data[17];
        LOG(INFO) << "sizeof(Eigen::Vector3f) = " << data[18];
        LOG(INFO) << "sizeof(Eigen::Vector4f) = " << data[19];
        LOG(INFO) << "sizeof(Sophus::SE3f) = " << data[20];
        LOG(INFO) << "EIGEN_DONT_VECTORIZE = " << data[21];
        LOG(INFO) << "EIGEN_VECTORIZE_CUDA = " << data[22];
        LOG(INFO) << "CUDACC_VS_CUDA_ARCH = " << data[23];
        LOG(INFO) << "EIGEN_CUDA_MAX_ALIGN_BYTES = " << data[24];
    }
};

extern void GetEigenConfigCPU(float* data);

TEST_F(Test_EigenConfig, CPU)
{    
    float buf[MaxEigenConfigurationCount];
    GetEigenConfigCPU(buf);
    reportEigen(buf);
    
}

#ifdef CORE_HAVE_CUDA
extern void GetEigenConfigCUDAHost(float* data);
extern void GetEigenConfigCUDADevice(float* data);

TEST_F(Test_EigenConfig, CUDAHost)
{   
    float buf[MaxEigenConfigurationCount];
    GetEigenConfigCUDAHost(buf);
    reportEigen(buf);
}

TEST_F(Test_EigenConfig, CUDADevice)
{    
    float buf[MaxEigenConfigurationCount];
    GetEigenConfigCUDADevice(buf);
    reportEigen(buf);
}

#endif // CORE_HAVE_CUDA
