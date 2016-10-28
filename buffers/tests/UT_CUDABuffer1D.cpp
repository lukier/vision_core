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

#include <Platform.hpp>

#include <buffers/Buffer1D.hpp>

#include <LaunchUtils.hpp>

static constexpr std::size_t BufferSize = 1000;
typedef float BufferElementT;

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

__global__ void Kernel_ReadBuffer1D(const core::Buffer1DView<BufferElementT,core::TargetDeviceCUDA> buf, std::size_t cbs)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(buf.size() != cbs)
    {
        asm("trap;");
    }
    
    if(buf.inBounds(x)) // is valid
    {
        BufferElementT elem = buf[x];
    }
}

TEST_F(Test_CUDABuffer1D, TestRead) 
{
    dim3 gridDim, blockDim;
    
    core::Buffer1DManaged<BufferElementT, core::TargetDeviceCUDA> buffer(BufferSize);
    
    core::InitDimFromLinearBuffer(blockDim, gridDim, buffer);
    
    Kernel_ReadBuffer1D<<<gridDim,blockDim>>>(buffer, BufferSize);
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    // --------------------------------
    
    buffer.resize(BufferSize/2);
    
    core::InitDimFromLinearBuffer(blockDim, gridDim, buffer);
    
    Kernel_ReadBuffer1D<<<gridDim,blockDim>>>(buffer, BufferSize/2);
    
    // wait for it
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

__global__ void Kernel_WriteBuffer1D(core::Buffer1DView<BufferElementT,core::TargetDeviceCUDA> buf, std::size_t cbs)
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

TEST_F(Test_CUDABuffer1D, TestWrite) 
{
    dim3 gridDim, blockDim;
    
    core::Buffer1DManaged<BufferElementT, core::TargetDeviceCUDA> buffer(BufferSize);
    
    core::InitDimFromLinearBuffer(blockDim, gridDim, buffer);
    
    Kernel_WriteBuffer1D<<<gridDim,blockDim>>>(buffer, BufferSize);
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    core::Buffer1DManaged<BufferElementT, core::TargetHost> cpu_buffer(BufferSize);
    cpu_buffer.copyFrom(buffer);
    
    for(std::size_t i = 0 ; i < BufferSize ; ++i)
    {
        const BufferElementT& elem = cpu_buffer[i];
        ASSERT_TRUE( ((float)i - elem) < std::numeric_limits<float>::epsilon() ) << "Error too big " << i << " vs " << elem; 
    }
    
    // --------------------------------
    
    buffer.resize(BufferSize/2);
    
    core::InitDimFromLinearBuffer(blockDim, gridDim, buffer);
    
    Kernel_WriteBuffer1D<<<gridDim,blockDim>>>(buffer, BufferSize/2);
    
    // wait for it
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    cpu_buffer.resize(buffer.size());
    cpu_buffer.copyFrom(buffer);
    
    for(std::size_t i = 0 ; i < BufferSize/2 ; ++i)
    {
        const BufferElementT& elem = cpu_buffer[i];
        ASSERT_TRUE( ((float)i - elem) < std::numeric_limits<float>::epsilon() ) << "Error too big " << i << " vs " << elem; 
    }
}
