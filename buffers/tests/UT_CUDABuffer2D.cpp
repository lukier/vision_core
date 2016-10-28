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
 * Basic CUDA Buffer2D Tests.
 * ****************************************************************************
 */

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <Platform.hpp>

#include <buffers/Buffer2D.hpp>

#include <LaunchUtils.hpp>

static constexpr std::size_t BufferSizeX = 1234;
static constexpr std::size_t BufferSizeY = 768;
typedef float BufferElementT;

class Test_CUDABuffer2D : public ::testing::Test
{
public:   
    Test_CUDABuffer2D() : buffer(BufferSizeX,BufferSizeY)
    {
        
    }
    
    virtual ~Test_CUDABuffer2D()
    {
        
    }
    
    core::Buffer2DManaged<BufferElementT, core::TargetDeviceCUDA> buffer;
};

__global__ void Kernel_ReadBuffer2D(const core::Buffer2DView<BufferElementT,core::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT elem = buf(x,y);
    }
}

TEST_F(Test_CUDABuffer2D, TestRead) 
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImage(blockDim, gridDim, buffer);
    
    Kernel_ReadBuffer2D<<<gridDim,blockDim>>>(buffer);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

__global__ void Kernel_WriteBuffer2D(core::Buffer2DView<BufferElementT,core::TargetDeviceCUDA> buf)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf.inBounds(x,y)) // is valid
    {
        BufferElementT& elem = buf(x,y);
        elem = x * y;
    }
}

TEST_F(Test_CUDABuffer2D, TestWrite) 
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImage(blockDim, gridDim, buffer);
    
    Kernel_WriteBuffer2D<<<gridDim,blockDim>>>(buffer);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}
