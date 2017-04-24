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
 * Basic CUDA Buffer1D Tests.
 * ****************************************************************************
 */

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <VisionCore/Buffers/Buffer1D.hpp>

#include <VisionCore/LaunchUtils.hpp>

static constexpr std::size_t BufferSize = 4097;
typedef uint32_t BufferElementT;

class Test_CUDABuffer1D : public ::testing::Test
{
public:   
    Test_CUDABuffer1D()
    {
        
    }
    
    virtual ~Test_CUDABuffer1D()
    {
        
    }
};

__global__ void Kernel_WriteBuffer1D(vc::Buffer1DView<BufferElementT,vc::TargetDeviceCUDA> buf, std::size_t cbs)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(buf.size() != cbs)
    {
        asm("trap;");
    }
    
    if(buf.inBounds(x)) // is valid
    {
        BufferElementT& elem = buf[x];
        elem = x;
    }
}

TEST_F(Test_CUDABuffer1D, TestHostDevice) 
{
    dim3 gridDim, blockDim;
    
    // CPU buffer 1
    vc::Buffer1DManaged<BufferElementT, vc::TargetHost> buffer_cpu1(BufferSize);
    
    // Fill H
    for(std::size_t i = 0 ; i < BufferSize ; ++i) { buffer_cpu1[i] = i * 10; }
    
    // GPU Buffer
    vc::Buffer1DManaged<BufferElementT, vc::TargetDeviceCUDA> buffer_gpu(BufferSize);
    
    // H->D
    buffer_gpu.copyFrom(buffer_cpu1);
    
    // CPU buffer 2
    vc::Buffer1DManaged<BufferElementT, vc::TargetHost> buffer_cpu2(BufferSize);
    
    // D->H
    buffer_cpu2.copyFrom(buffer_gpu);
    
    // Check
    for(std::size_t i = 0 ; i < BufferSize ; ++i) 
    {
        ASSERT_EQ(buffer_cpu2(i), i * 10) << "Wrong data at " << i;
    }
    
    // Now write from kernel
    vc::InitDimFromBuffer(blockDim, gridDim, buffer_gpu);
    Kernel_WriteBuffer1D<<<gridDim,blockDim>>>(buffer_gpu, BufferSize);
    
    // Wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw vc::CUDAException(err, "Error launching the kernel");
    }
    
    // D->H
    buffer_cpu1.copyFrom(buffer_gpu);
    
    // Check
    for(std::size_t i = 0 ; i < BufferSize ; ++i) 
    {
        ASSERT_EQ(buffer_cpu1(i), i) << "Wrong data at " << i;
    }
}
