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
 * Memory Policies for the Host.
 * ****************************************************************************
 */

#ifndef CORE_MEMORY_POLICY_HOST_HPP
#define CORE_MEMORY_POLICY_HOST_HPP

#include <cstdlib>

#include <Platform.hpp>

namespace core
{

template<typename T>
struct TargetHost
{
    typedef void* PointerType;
    typedef int TextureHandleType;
    
    inline static void AllocateMem(PointerType* devPtr, size_t s)
    {
        /// @todo is there much difference?
#ifdef CORE_HAVE_CUDA
        const cudaError err = cudaMallocHost(devPtr, sizeof(T) * s);
        if( err != cudaSuccess )  { throw CUDAException(err, "Unable to cudaMallocHost"); }
#else // CORE_HAVE_CUDA
        *devPtr = malloc(sizeof(T) * s);
        if(*devPtr == nullptr) { throw std::bad_alloc(); }
#endif // CORE_HAVE_CUDA
    }
    
    inline static void AllocatePitchedMem(PointerType* hostPtr, size_t *pitch, size_t w, size_t h)
    {
        *pitch = w * sizeof(T);
        
#ifdef CORE_HAVE_CUDA        
        const cudaError err = cudaMallocHost(hostPtr, *pitch * h);        
        if( err != cudaSuccess )  { throw CUDAException(err, "Unable to cudaMallocHost"); }
#else // CORE_HAVE_CUDA
        *hostPtr = malloc(*pitch * h);
        if(*hostPtr == nullptr) { throw std::bad_alloc(); }
#endif // CORE_HAVE_CUDA
    }

    inline static void AllocatePitchedMem(PointerType* hostPtr, size_t *pitch, size_t *img_pitch, size_t w, size_t h, size_t d)
    {

        *pitch = w * sizeof(T);

#ifdef CORE_HAVE_CUDA        
        const cudaError err = cudaMallocHost(hostPtr, *pitch * h * d);
        
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMallocHost"); }
#else // CORE_HAVE_CUDA
        *hostPtr = malloc(*pitch * h * d);
        if(*hostPtr == nullptr) { throw std::bad_alloc(); }
#endif // CORE_HAVE_CUDA
        
        *img_pitch = *pitch * h;
    }

    inline static void DeallocatePitchedMem(PointerType hostPtr) throw()
    {
#ifdef CORE_HAVE_CUDA
        cudaFreeHost(hostPtr);
#else // CORE_HAVE_CUDA
        free(hostPtr);
#endif // CORE_HAVE_CUDA
    }
    
    inline static void memset(PointerType ptr, int value, std::size_t count) 
    {
        std::memset(ptr, value, count);
    }
    
    inline static void memset2D(PointerType ptr, std::size_t pitch, int value, std::size_t width, std::size_t height)
    {
        std::memset(ptr, value, std::max(width,pitch) * height);
    }
    
    inline static void memset3D(PointerType ptr, std::size_t pitch, int value, std::size_t width, std::size_t height, std::size_t depth)
    {
        std::memset(ptr, value, std::max(width,pitch) * height * depth);
    }
};

template<typename T1, typename T2>
struct TargetTransfer<TargetHost<T1>,TargetHost<T2>>
{
    typedef TargetHost<T1> TargetFrom;
    typedef TargetHost<T2> TargetTo;
    
    inline static void memcpy(typename TargetTo::PointerType dst, const typename TargetFrom::PointerType src, std::size_t count)
    {
#ifdef CORE_HAVE_CUDA
        static constexpr cudaMemcpyKind CopyKind = cudaMemcpyHostToHost;
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
#else // CORE_HAVE_CUDA
        std::memcpy(dst, src, count);
#endif // CORE_HAVE_CUDA
    }
    
    inline static void memcpy2D(typename TargetTo::PointerType dst, std::size_t  dpitch, const typename TargetFrom::PointerType src, std::size_t  spitch, std::size_t  width, std::size_t  height)
    {
#ifdef CORE_HAVE_CUDA
        static constexpr cudaMemcpyKind CopyKind = cudaMemcpyHostToHost;
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
#else // CORE_HAVE_CUDA
        std::memcpy(dst, src, width * height);
#endif // CORE_HAVE_CUDA
    }
};

}

#endif // CORE_MEMORY_POLICY_HOST_HPP
