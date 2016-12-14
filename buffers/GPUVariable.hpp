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
 * Single Host/Device Variable.
 * ****************************************************************************
 */

#ifndef CORE_GPU_VARIABLE_HPP
#define CORE_GPU_VARIABLE_HPP

#include <Platform.hpp>

#include <MemoryPolicy.hpp>

namespace core
{

template<typename T, template<typename = T> class Target = TargetHost>
class GPUVariableView
{
public:
    typedef T ValueType;
    
#ifndef CORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline GPUVariableView() : memptr(0) { }
#else // CORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline GPUVariableView() { }
#endif // CORE_CUDA_KERNEL_SPACE
    
    EIGEN_DEVICE_FUNC inline ~GPUVariableView()
    {

    }
    
    EIGEN_DEVICE_FUNC inline GPUVariableView( const GPUVariableView<T,Target>& img ) : memptr(img.memptr)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline GPUVariableView( GPUVariableView<T,Target>&& img ) : memptr(img.memptr)
    {
        // This object will take over managing data (if Management = Manage)
        img.memptr = 0;
    }
    
    EIGEN_DEVICE_FUNC inline GPUVariableView(T* optr) : memptr(optr)
    {  
        
    }
    
    EIGEN_DEVICE_FUNC inline GPUVariableView<T,Target>& operator=(const GPUVariableView<T,Target>& other)
    {
        memptr = (void*)other.ptr();
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline GPUVariableView<T,Target>& operator=(GPUVariableView<T,Target>&& img)
    {
        memptr = img.memptr;
        img.memptr = 0;
        return *this;
    }
    
    inline void swap(GPUVariableView<T,Target>& other)
    {
        std::swap(other.memptr, memptr);
    }
    
    EIGEN_DEVICE_FUNC inline bool isValid() const
    {
        return memptr != 0;
    }
    
    EIGEN_DEVICE_FUNC inline T* ptr()
    {
        return static_cast<T*>(memptr);
    }
    
    EIGEN_DEVICE_FUNC inline const T* ptr() const
    {
        return static_cast<T*>(memptr);
    }
    
    EIGEN_PURE_DEVICE_FUNC inline const T& get() const { return *static_cast<T*>(memptr); }
    EIGEN_PURE_DEVICE_FUNC inline T& get() { return *static_cast<T*>(memptr); }
    
    EIGEN_PURE_DEVICE_FUNC inline void set(const T& value) { *static_cast<T*>(memptr) = value; }
    
#ifdef CORE_HAVE_CUDA
    inline T toHost() const
    {
        T ret;
        const cudaError err = cudaMemcpy( (void*)&ret, (void*)memptr, sizeof(T), cudaMemcpyDeviceToHost);
        if( err != cudaSuccess ) 
        {
            throw CUDAException(err, "Unable to cudaMemcpy2D in MemcpyToHost");
        }
        return ret;
    }
    
    inline void fromHost(const T& value)
    {
        const cudaError err = cudaMemcpy( (void*)memptr, (void*)&value, sizeof(T), cudaMemcpyHostToDevice);
        if( err != cudaSuccess ) 
        {
            throw CUDAException(err, "Unable to cudaMemcpy2D in MemcpyToHost");
        }
    }
#endif // CORE_HAVE_CUDA
    
protected:
    typename Target<T>::PointerType memptr;
};

template<typename T, template<typename = T> class Target = TargetHost>
class GPUVariableManaged : public GPUVariableView<T,Target>
{
public:
    typedef GPUVariableView<T,Target> ViewT;
    
    inline GPUVariableManaged()
    {
        typename Target<T>::PointerType ptr = 0;
        Target<T>::AllocateMem(&ptr, 1);
        ViewT::memptr = ptr;
    }
    
    inline GPUVariableManaged(T cpu_value)
    {
        typename Target<T>::PointerType ptr = 0;
        Target<T>::AllocateMem(&ptr, 1);
        ViewT::memptr = ptr;
        ViewT::fromHost(cpu_value);
    }
    
    inline ~GPUVariableManaged()
    {
        Target<T>::DeallocatePitchedMem(ViewT::memptr);
    }
    
    GPUVariableManaged(const GPUVariableManaged<T,Target>& img) = delete;
    
    inline GPUVariableManaged(GPUVariableManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    GPUVariableManaged<T,Target>& operator=(const GPUVariableManaged<T,Target>& img) = delete;
    
    inline GPUVariableManaged<T,Target>& operator=(GPUVariableManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

}

#endif // CORE_GPU_VARIABLE_HPP
