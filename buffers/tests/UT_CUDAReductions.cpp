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

#include <LaunchUtils.hpp>

static constexpr std::size_t BufferSizeX = 32;
static constexpr std::size_t BufferSizeY = 32;
static constexpr std::size_t BlockSizeX = 4;
static constexpr std::size_t BlockSizeY = 4;
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
    
    auto sum_op = [&] __device__ (T& outval, const T& inval) 
    {
        outval += inval;
    };
    
    core::runReductions<T>(Nblocks, [&] __device__ (unsigned int i) 
    { 
        const unsigned int y = i / bin.width();
        const unsigned int x = i - (y * bin.width());
        
        sum_op(sum, bin(x,y));
    });
    
    core::finalizeReduction(bout.ptr(), &sum, sum_op);
}

template<typename T>
__global__ void reduce1DBuffer(
    core::Buffer1DView<T, core::TargetDeviceCUDA> bin, 
    core::Buffer1DView<T, core::TargetDeviceCUDA> bout,
    unsigned int Nblocks)
{
    T sum = core::zero<T>();
    
    auto sum_op = [&] __device__ (T& outval, const T& inval) 
    {
        outval += inval;
    };
    
    core::runReductions<T>(Nblocks, [&] __device__ (unsigned int i) 
    { 
        sum_op(sum, bin(i));
    });
    
    core::finalizeReduction(bout.ptr(), &sum, sum_op);
}

TEST_F(Test_CUDAReductions, TestFloat) 
{
    const unsigned int block_size = 32;
    const unsigned int block_dim = core::detail::Gcd<unsigned int>(buffer.area(), block_size);
    const unsigned int grid_dim = buffer.area() / block_dim;
    
    core::Buffer1DManaged<BufferElementT, core::TargetDeviceCUDA> scratch_buffer(block_dim);
    core::Buffer1DManaged<BufferElementT, core::TargetDeviceCUDA> scratch_buffer2(block_dim);
    core::Buffer1DManaged<BufferElementT, core::TargetHost> scratch_buffer_cpu(block_dim);
    
    LOG(INFO) << "Running with " << grid_dim << " / " << block_dim << " = " << (grid_dim * block_dim);
     
    printBuffer(buffer);
    
    // run kernel
    reduce2DBuffer<BufferElementT><<<grid_dim,block_dim>>>(buffer, scratch_buffer, buffer.area());
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    LOG(INFO) << "Whoa";
    
    scratch_buffer_cpu.copyFrom(scratch_buffer);
    
    for(unsigned int i = 0 ; i < block_dim ; ++i)
    {
        std::cout << "(" << scratch_buffer_cpu(i) << ") | ";
    }
    std::cout << std::endl;
    
    // final reduction
    
    reduce1DBuffer<BufferElementT><<<1,block_dim>>>(scratch_buffer, scratch_buffer2, block_dim);
    scratch_buffer_cpu.copyFrom(scratch_buffer2);
    
    LOG(INFO) << "Result: " << scratch_buffer_cpu(0) << " vs GT " <<  ground_truth;
}

TEST_F(Test_CUDAReductions, TestVector2f) 
{
    const unsigned int block_size = 32;
    const unsigned int block_dim = core::detail::Gcd<unsigned int>(ebuffer.area(), block_size);
    const unsigned int grid_dim = ebuffer.area() / block_dim;
    
    core::Buffer1DManaged<EigenElementT, core::TargetDeviceCUDA> scratch_buffer(block_dim);
    core::Buffer1DManaged<EigenElementT, core::TargetDeviceCUDA> scratch_buffer2(block_dim);
    core::Buffer1DManaged<EigenElementT, core::TargetHost> scratch_buffer_cpu(block_dim);
    
    LOG(INFO) << "Running with " << grid_dim << " / " << block_dim << " = " << (grid_dim * block_dim);
     
    printBuffer(ebuffer);
    
    // run kernel
    reduce2DBuffer<EigenElementT><<<grid_dim,block_dim>>>(ebuffer, scratch_buffer, ebuffer.area());
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
    
    LOG(INFO) << "Whoa";
    
    scratch_buffer_cpu.copyFrom(scratch_buffer);
    
    for(unsigned int i = 0 ; i < block_dim ; ++i)
    {
        std::cout << "(" << scratch_buffer_cpu(i)(0) << "," << scratch_buffer_cpu(i)(1) << ") | ";
    }
    std::cout << std::endl;
    
    // final reduction
    
    reduce1DBuffer<EigenElementT><<<1,block_dim>>>(scratch_buffer, scratch_buffer2, block_dim);
    scratch_buffer_cpu.copyFrom(scratch_buffer2);
    
    LOG(INFO) << "Result: " << scratch_buffer_cpu(0)(0) << " , " << scratch_buffer_cpu(0)(1) << " vs GT " << eground_truth(0) << " , " << eground_truth(1);
}

