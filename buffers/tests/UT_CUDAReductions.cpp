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
#include <buffers/Reductions.hpp>
#include <buffers/ReductionSum2D.hpp>

#include <LaunchUtils.hpp>

static constexpr std::size_t BufferSizeX = 640;
static constexpr std::size_t BufferSizeY = 480;
static constexpr std::size_t ThreadsPerBlock = 512;
typedef float BufferElementT;
typedef Eigen::Vector2f EigenElementT;

class Test_CUDAReductions : public ::testing::Test
{
public:   
    Test_CUDAReductions() : 
        buffer(BufferSizeX,BufferSizeY),  
        buffer_cpu(BufferSizeX,BufferSizeY),
        ebuffer(BufferSizeX,BufferSizeY), 
        ebuffer_cpu(BufferSizeX,BufferSizeY)
    {
        ground_truth = 0.0f;
        eground_truth << 0.0f, 0.0f;
        
        // fill
        for(std::size_t y = 0 ; y < buffer_cpu.height() ; ++y)
        {
            for(std::size_t x = 0 ; x < buffer_cpu.width() ; ++x)
            {
                const BufferElementT val = 1.0f;
                const EigenElementT eval(1.0f,2.0f);
                buffer_cpu(x,y) = val;
                ground_truth += val;
                ebuffer_cpu(x,y) = eval;
                eground_truth += eval;
            }
        }
        
        buffer.copyFrom(buffer_cpu);
        ebuffer.copyFrom(ebuffer_cpu);
    }
    
    virtual ~Test_CUDAReductions()
    {
        
    }
    
    void printBuffer(const core::Buffer2DView<BufferElementT, core::TargetDeviceCUDA>& b)
    {
        buffer_cpu.copyFrom(b);
        
        for(std::size_t y = 0 ; y < buffer_cpu.height() ; ++y)
        {
            for(std::size_t x = 0 ; x < buffer_cpu.width() ; ++x)
            {
                std::cout << "(" << ebuffer_cpu(x,y) << ") ";
            }
            std::cout << " | " << std::endl;
        }
    }
    
    void printBuffer(const core::Buffer2DView<EigenElementT, core::TargetDeviceCUDA>& b)
    {
        ebuffer_cpu.copyFrom(b);
        
        for(std::size_t y = 0 ; y < buffer_cpu.height() ; ++y)
        {
            for(std::size_t x = 0 ; x < buffer_cpu.width() ; ++x)
            {
                std::cout << "(" << ebuffer_cpu(x,y)(0) << " , " << ebuffer_cpu(x,y)(1) << ") ";
            }
            std::cout << " | " << std::endl;
        }
    }
    
    core::Buffer2DManaged<BufferElementT, core::TargetDeviceCUDA> buffer;
    core::Buffer2DManaged<BufferElementT, core::TargetHost> buffer_cpu;
    core::Buffer2DManaged<EigenElementT, core::TargetDeviceCUDA> ebuffer;
    core::Buffer2DManaged<EigenElementT, core::TargetHost> ebuffer_cpu;
    BufferElementT ground_truth;
    EigenElementT eground_truth;
};

template<typename T>
__global__ void reduce2DBuffer(
    core::Buffer2DView<T, core::TargetDeviceCUDA> bin, 
    core::Buffer1DView<T, core::TargetDeviceCUDA> bout,
    unsigned int Nblocks)
{
    T sum = core::zero<T>();
    
    core::runReductions(Nblocks, [&] __device__ (unsigned int i) 
    { 
        const unsigned int y = i / bin.width();
        const unsigned int x = i - (y * bin.width());
        
        sum = bin(x,y);
    });
    
    core::finalizeReduction(bout.ptr(), &sum, core::internal::warpReduceSum<T>, core::zero<T>());
}

template<typename T>
__global__ void reduce1DBuffer(
    core::Buffer1DView<T, core::TargetDeviceCUDA> bin, 
    core::Buffer1DView<T, core::TargetDeviceCUDA> bout,
    unsigned int Nblocks)
{
    T sum = core::zero<T>();
    
    core::runReductions(Nblocks, [&] __device__ (unsigned int i) 
    { 
        sum = bin(i);
    });
    
    core::finalizeReduction(bout.ptr(), &sum, core::internal::warpReduceSum<T>, core::zero<T>());
}

TEST_F(Test_CUDAReductions, TestNewFloat) 
{
    const std::size_t threads = ThreadsPerBlock;
    const std::size_t blocks = std::min((buffer.area() + threads - 1) / threads, (std::size_t)1024);
    
    LOG(INFO) << "F Reducing1 " << blocks << " / " << threads << " = " << (blocks * threads);
    
    core::Buffer1DManaged<BufferElementT, core::TargetDeviceCUDA> scratch_buffer(blocks);
    core::Buffer1DManaged<BufferElementT, core::TargetHost> scratch_buffer_cpu(blocks);
    
    auto tpt1 = std::chrono::steady_clock::now();
    
    // run kernel
    reduce2DBuffer<BufferElementT><<<blocks, threads, 32 * sizeof(BufferElementT)>>>(buffer, scratch_buffer, buffer.area());
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    auto tpt2 = std::chrono::steady_clock::now();
    
    // final reduction - my way
    
    reduce1DBuffer<BufferElementT><<<1, 1024, 32 * sizeof(BufferElementT)>>>(scratch_buffer, scratch_buffer, blocks);
    
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    auto tpt3 = std::chrono::steady_clock::now();

    scratch_buffer_cpu.copyFrom(scratch_buffer);
    
    LOG(INFO) << "F Result1: " << scratch_buffer_cpu(0) << " vs GT " <<  ground_truth;
    
    LOG(INFO) << "F Reduction2D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt2-tpt1).count() << " [us]";
    LOG(INFO) << "F Reduction1D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt3-tpt2).count() << " [us]";
    LOG(INFO) << "F ReductionMy: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt3-tpt1).count() << " [us]";
}

TEST_F(Test_CUDAReductions, TestNewVector2f) 
{
    const std::size_t threads = ThreadsPerBlock;
    const std::size_t blocks = std::min((buffer.area() + threads - 1) / threads, (std::size_t)1024);
    
    LOG(INFO) << "V2 Reducing1 " << blocks << " / " << threads << " = " << (blocks * threads);
    
    core::Buffer1DManaged<EigenElementT, core::TargetDeviceCUDA> scratch_buffer(blocks);
    core::Buffer1DManaged<EigenElementT, core::TargetHost> scratch_buffer_cpu(blocks);
    
    auto tpt1 = std::chrono::steady_clock::now();
    
    // run kernel
    reduce2DBuffer<EigenElementT><<<blocks, threads, 32 * sizeof(EigenElementT)>>>(ebuffer, scratch_buffer, ebuffer.area());
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    auto tpt2 = std::chrono::steady_clock::now();
    
    // final reduction - my way
    
    reduce1DBuffer<EigenElementT><<<1, 1024, 32 * sizeof(EigenElementT)>>>(scratch_buffer, scratch_buffer, blocks);
    
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    auto tpt3 = std::chrono::steady_clock::now();

    scratch_buffer_cpu.copyFrom(scratch_buffer);
    
    LOG(INFO) << "V2 Result1: " << scratch_buffer_cpu(0)(0) << " , " << scratch_buffer_cpu(0)(1) << " vs GT " <<  eground_truth(0) << " , " << eground_truth(1);
    
    LOG(INFO) << "V2 Reduction2D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt2-tpt1).count() << " [us]";
    LOG(INFO) << "V2 Reduction1D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt3-tpt2).count() << " [us]";
    LOG(INFO) << "V2 ReductionMy: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt3-tpt1).count() << " [us]";
}

template<typename T>
__global__ void ReduceOld2D(core::HostReductionSum2DView<T> sum, core::Buffer2DView<T, core::TargetDeviceCUDA> bin)
{
    // current pixel
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    // reduction for perr
    typedef core::DeviceReductionSum2D<T, 32, 32> ReductionT;

    FIXED_SIZE_SHARED_VAR(sumIt, ReductionT);
    
    if(bin.inBounds(x,y))
    {
        sumIt.getThisBlock() = bin(x,y);
    }
    
    sumIt.reduceBlock(sum);
}

TEST_F(Test_CUDAReductions, TestOldFloat) 
{
    dim3 blockDim, gridDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buffer, 16, 16);
    
    LOG(INFO) << "OF Running with " << gridDim.x << " x " << gridDim.y << " / " << blockDim.x << " x " << blockDim.y;
    
    core::HostReductionSum2DManaged<BufferElementT> scratch(gridDim);
    
    auto tpt1 = std::chrono::steady_clock::now();
    
    ReduceOld2D<BufferElementT><<<gridDim,blockDim>>>(scratch, buffer);
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    BufferElementT sum = core::zero<BufferElementT>();
    scratch.getFinalSum(sum);
    
    auto tpt2 = std::chrono::steady_clock::now();
    
    LOG(INFO) << "OF Result: " << sum << " vs GT " << ground_truth;
    LOG(INFO) << "OF Reduction2D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt2-tpt1).count() << " [us]";
}

TEST_F(Test_CUDAReductions, TestOldVector2f) 
{
    dim3 blockDim, gridDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, ebuffer, 16, 16);
    
    LOG(INFO) << "OV2 Running with " << gridDim.x << " x " << gridDim.y << " / " << blockDim.x << " x " << blockDim.y;
    
    core::HostReductionSum2DManaged<EigenElementT> scratch(gridDim);
    
    auto tpt1 = std::chrono::steady_clock::now();
    
    ReduceOld2D<EigenElementT><<<gridDim,blockDim>>>(scratch, ebuffer);
    
    // wait for it
    cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    EigenElementT sum = core::zero<EigenElementT>();
    scratch.getFinalSum(sum);
    
    auto tpt2 = std::chrono::steady_clock::now();
    
    LOG(INFO) << "OV2 Result: " << sum(0) << " , " << sum(1) << " vs GT " << eground_truth(0) << " , " << eground_truth(1);
    LOG(INFO) << "OV2 Reduction2D: " << std::chrono::duration_cast<std::chrono::microseconds>(tpt2-tpt1).count() << " [us]";
}
