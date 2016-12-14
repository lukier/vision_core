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
 * Memory Policies for CUDA.
 * ****************************************************************************
 */

#ifndef CORE_MEMORY_POLICY_CUDA_HPP
#define CORE_MEMORY_POLICY_CUDA_HPP

#include <cstdlib>

#include <Platform.hpp>

#include <thrust/device_vector.h>

namespace core
{

template<typename T>
struct TargetDeviceCUDA;
    
namespace internal
{
    cudaMemcpyKind TargetCopyKind();
    
    template<typename TargetTo, typename TargetFrom>
    struct TargetCopyKindTraits;
    
    template<typename T>
    struct TargetCopyKindTraits<TargetHost<T>,TargetHost<T>> { static constexpr cudaMemcpyKind Kind = cudaMemcpyHostToHost; };
    template<typename T>
    struct TargetCopyKindTraits<TargetDeviceCUDA<T>,TargetHost<T>> { static constexpr cudaMemcpyKind Kind = cudaMemcpyHostToDevice; };
    template<typename T>
    struct TargetCopyKindTraits<TargetHost<T>,TargetDeviceCUDA<T>> { static constexpr cudaMemcpyKind Kind = cudaMemcpyDeviceToHost; };
    template<typename T>
    struct TargetCopyKindTraits<TargetDeviceCUDA<T>,TargetDeviceCUDA<T>> { static constexpr cudaMemcpyKind Kind = cudaMemcpyDeviceToDevice; };
    
    template<typename TargetTo, typename TargetFrom>
    inline cudaMemcpyKind TargetCopyKind() { return TargetCopyKindTraits<TargetTo,TargetFrom>::Kind; }
    
    template<typename T, typename Target> struct ThrustType;
    template<typename T> struct ThrustType<T,TargetHost<T>> { typedef T* Ptr; };
    template<typename T> struct ThrustType<T,TargetDeviceCUDA<T>> { typedef thrust::device_ptr<T> Ptr; };
}
    
template<typename T>
struct TargetDeviceCUDA
{
    typedef void* PointerType;
    typedef cudaTextureObject_t TextureHandleType;
    
    inline static void AllocateMem(PointerType* devPtr, std::size_t s)
    {
        const cudaError err = cudaMalloc(devPtr, sizeof(T) * s);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMalloc"); }
    }
    
    inline static void AllocatePitchedMem(PointerType* devPtr, std::size_t* pitch, std::size_t w, std::size_t h)
    {
        const cudaError err = cudaMallocPitch(devPtr, pitch, w * sizeof(T), h);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMallocPitch"); }
    }

    inline static void AllocatePitchedMem(PointerType* devPtr, std::size_t* pitch, std::size_t* img_pitch, std::size_t w, std::size_t h, std::size_t d)
    {
        const cudaError err = cudaMallocPitch(devPtr, pitch, w * sizeof(T), h * d);
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMallocPitch"); }
        
        *img_pitch = *pitch * h;
    }

    inline static void DeallocatePitchedMem(PointerType devPtr) throw()
    {
#ifndef CORE_CUDA_KERNEL_SPACE
        cudaFree(devPtr);
#endif // CORE_CUDA_KERNEL_SPACE
    }
    
    inline static void memset(PointerType devPtr, int value, std::size_t count) 
    {
        cudaMemset(devPtr, value, count);
    }
    
    inline static void memset2D(PointerType ptr, std::size_t pitch, int value, std::size_t width, std::size_t height)
    {
        cudaMemset2D(ptr, pitch, value, width, height);
    }
    
    inline static void memset3D(PointerType ptr, std::size_t pitch, int value, std::size_t width, std::size_t height, std::size_t depth)
    {
        cudaMemset3D(make_cudaPitchedPtr(ptr, pitch, width, height), value, make_cudaExtent(width,height,depth));
    }
};

template<typename T1, typename T2>
struct TargetTransfer<TargetHost<T1>,TargetDeviceCUDA<T2>>
{
    typedef TargetHost<T1> TargetFrom;
    typedef TargetDeviceCUDA<T2> TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyHostToDevice;
    
    inline static void memcpy(typename TargetTo::PointerType dst,
                              const typename TargetFrom::PointerType src, std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    inline static void memcpy2D(typename TargetTo::PointerType dst, std::size_t  dpitch, 
                                const typename TargetFrom::PointerType src, 
                                std::size_t  spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

template<typename T1, typename T2>
struct TargetTransfer<TargetDeviceCUDA<T1>,TargetHost<T2>>
{
    typedef TargetDeviceCUDA<T1> TargetFrom;
    typedef TargetHost<T2> TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyDeviceToHost;
    
    inline static void memcpy(typename TargetTo::PointerType dst, 
                              const typename TargetFrom::PointerType src, 
                              std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    inline static void memcpy2D(typename TargetTo::PointerType dst, std::size_t dpitch, 
                                const typename TargetFrom::PointerType src, 
                                std::size_t spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

template<typename T1, typename T2>
struct TargetTransfer<TargetDeviceCUDA<T1>,TargetDeviceCUDA<T2>>
{
    typedef TargetDeviceCUDA<T1> TargetFrom;
    typedef TargetDeviceCUDA<T2> TargetTo;
    static constexpr cudaMemcpyKind CopyKind = cudaMemcpyDeviceToDevice;
    
    inline static void memcpy(typename TargetTo::PointerType dst, 
                              const typename TargetFrom::PointerType src, 
                              std::size_t count)
    {
        const cudaError err = cudaMemcpy(dst, src, count, CopyKind );
        if( err != cudaSuccess ) { throw CUDAException(err, "Unable to cudaMemcpy"); }
    }
    
    inline static void memcpy2D(typename TargetTo::PointerType dst, std::size_t dpitch, 
                                const typename TargetFrom::PointerType src, 
                                std::size_t  spitch, std::size_t  width, std::size_t  height)
    {
        const cudaError err = cudaMemcpy2D(dst,dpitch,src,spitch, width, height, CopyKind );
        if(err != cudaSuccess) { throw CUDAException(err, "Unable to cudaMemcpy2D"); }
    }
};

}

#endif // CORE_MEMORY_POLICY_CUDA_HPP
