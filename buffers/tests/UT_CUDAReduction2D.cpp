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
 * Basic CUDA Reduction2D Tests.
 * ****************************************************************************
 */

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <Platform.hpp>

#include <buffers/Buffer2D.hpp>
#include <buffers/ReductionSum2D.hpp>

#include <LaunchUtils.hpp>

static constexpr std::size_t BufferSizeX = 128;
static constexpr std::size_t BufferSizeY = 128;
static constexpr std::size_t BlockSizeX = 8;
static constexpr std::size_t BlockSizeY = 8;
typedef float BufferElementT;

class Test_CUDAReduction2D : public ::testing::Test
{
public:   
    Test_CUDAReduction2D() : buffer(BufferSizeX,BufferSizeY), buffer_cpu(BufferSizeX,BufferSizeY)
    {
        ground_truth = 0.0f;
        
        // fill
        for(std::size_t y = 0 ; y< buffer_cpu.height() ; ++y)
        {
            for(std::size_t x = 0 ; x < buffer_cpu.width() ; ++x)
            {
                const BufferElementT val = 1.0f;
                buffer_cpu(x,y) = val;
                ground_truth += val;
            }
        }
    }
    
    virtual ~Test_CUDAReduction2D()
    {
        
    }
    
    core::Buffer2DManaged<BufferElementT, core::TargetDeviceCUDA> buffer;
    core::Buffer2DManaged<BufferElementT, core::TargetHost> buffer_cpu;
    BufferElementT ground_truth;
};

__global__ void Kernel_SumBufferStatic(const core::Buffer2DView<BufferElementT,core::TargetDeviceCUDA> buf, core::HostReductionSum2DView<BufferElementT> reductor)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    typedef core::DeviceReductionSum2D<BufferElementT,BlockSizeX,BlockSizeY> ReductionT;
    
    FIXED_SIZE_SHARED_VAR(sumIt, ReductionT);
    
    if(buf.inBounds(x,y)) // is valid
    {
        sumIt.getThisBlock() = buf(x,y);
    }
    
    sumIt.reduceBlock(reductor);
}

TEST_F(Test_CUDAReduction2D, TestStatic) 
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImage(blockDim, gridDim, buffer, BlockSizeX, BlockSizeY);
    
    buffer.copyFrom(buffer_cpu);
    
    core::HostReductionSum2DManaged<BufferElementT> reductor(gridDim);
    
    Kernel_SumBufferStatic<<<gridDim,blockDim>>>(buffer, reductor);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    BufferElementT final_sum = 0.0f;
    
    reductor.getFinalSum(final_sum);
    
    ASSERT_FLOAT_EQ(ground_truth, final_sum);
}

__global__ void Kernel_SumBufferDynamic(const core::Buffer2DView<BufferElementT,core::TargetDeviceCUDA> buf, core::HostReductionSum2DView<BufferElementT> reductor)
{
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    typedef core::DeviceReductionSum2DDynamic<BufferElementT> ReductionT;
    
    ReductionT sumIt;
    
    if(buf.inBounds(x,y)) // is valid
    {
        sumIt.getThisBlock() = buf(x,y);
    }
    
    sumIt.reduceBlock(reductor);
}

TEST_F(Test_CUDAReduction2D, TestDynamic) 
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImage(blockDim, gridDim, buffer);
    
    buffer.copyFrom(buffer_cpu);
    
    core::HostReductionSum2DManaged<BufferElementT> reductor(gridDim);
    
    Kernel_SumBufferDynamic<<<gridDim,blockDim,reductor.getSharedMemorySize(blockDim)>>>(buffer, reductor);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    BufferElementT final_sum = 0.0f;
    
    reductor.getFinalSum(final_sum);
    
    ASSERT_FLOAT_EQ(ground_truth, final_sum);
}
