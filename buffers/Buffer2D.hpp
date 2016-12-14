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
 * 2D Host/Device Buffer.
 * ****************************************************************************
 */

#ifndef CORE_BUFFER2D_HPP
#define CORE_BUFFER2D_HPP

#include <Platform.hpp>

// CUDA
#ifdef CORE_HAVE_CUDA
#include <thrust/device_vector.h>
#include <npp.h>
#endif // CORE_HAVE_CUDA

#include <MemoryPolicy.hpp>

namespace core
{
 
// forward
template<typename T, template<typename = T> class Target = TargetHost>
class Image2DView;
    
/**
 * Buffer 2D View - Basics.
 */
template<typename T, template<typename = T> class Target>
class Buffer2DViewBase
{
public:
    typedef T ValueType;
    
#ifndef CORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer2DViewBase() : memptr(nullptr) { }
#else // CORE_CUDA_KERNEL_SPACE
    EIGEN_DEVICE_FUNC inline Buffer2DViewBase() { }
#endif // CORE_CUDA_KERNEL_SPACE
    
    EIGEN_DEVICE_FUNC inline ~Buffer2DViewBase() { }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase( const Buffer2DViewBase<T,Target>& img )  : memptr(img.memptr), line_pitch(img.line_pitch), xsize(img.xsize), ysize(img.ysize)
    {
        
    }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase(Buffer2DViewBase<T,Target>&& img) : memptr(img.memptr), line_pitch(img.line_pitch), xsize(img.xsize), ysize(img.ysize)
    {
        img.memptr = 0;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewBase<T,Target>& operator=(const Buffer2DViewBase<T,Target>& img)
    {        
        line_pitch = img.pitch();
        memptr = (void*)img.rawPtr();
        xsize = img.width();
        ysize = img.height();
        
        return *this;
    }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase<T,Target>& operator=(Buffer2DViewBase<T,Target>&& img)
    {
        line_pitch = img.line_pitch;
        memptr = img.memptr;
        xsize = img.xsize;
        ysize = img.ysize;
        img.memptr = 0;
        
        return *this;
    }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase(typename Target<T>::PointerType optr, std::size_t w) : memptr(optr), line_pitch(w * sizeof(T)), xsize(w), ysize(0)
    {  
        
    }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase(typename Target<T>::PointerType optr, std::size_t w, std::size_t h) : memptr(optr), line_pitch(w * sizeof(T)), xsize(w), ysize(h)
    {
        
    }

    EIGEN_DEVICE_FUNC inline Buffer2DViewBase(typename Target<T>::PointerType optr, std::size_t w, std::size_t h, std::size_t opitch) : memptr(optr), line_pitch(opitch), xsize(w), ysize(h)
    {   
        
    }

    EIGEN_DEVICE_FUNC inline std::size_t width() const { return xsize; }
    EIGEN_DEVICE_FUNC inline std::size_t height() const { return ysize; }
    EIGEN_DEVICE_FUNC inline std::size_t pitch() const { return line_pitch; }
    EIGEN_DEVICE_FUNC inline std::size_t elementPitch() const { return line_pitch / sizeof(T); }
    EIGEN_DEVICE_FUNC inline std::size_t area() const { return xsize * ysize; }
    EIGEN_DEVICE_FUNC inline std::size_t totalElements() const { return elementPitch() * ysize; }
    EIGEN_DEVICE_FUNC inline std::size_t bytes() const { return pitch() * ysize; }
    
    inline void swap(Buffer2DViewBase<T,Target>& img)
    {
        std::swap(img.line_pitch, line_pitch);
        std::swap(img.memptr, memptr);
        std::swap(img.xsize, xsize);
        std::swap(img.ysize, ysize);
    }
    
    EIGEN_DEVICE_FUNC inline bool isValid() const { return rawPtr() != nullptr; }
    
    EIGEN_DEVICE_FUNC inline const typename Target<T>::PointerType rawPtr() const { return memptr; }
    EIGEN_DEVICE_FUNC inline typename Target<T>::PointerType rawPtr() { return memptr; }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(std::size_t x, std::size_t y) const
    {
        return x < xsize && y < ysize;
    }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(int x, int y) const
    {
        return 0 <= x && x < (int)xsize && 0 <= y && y < (int)ysize;
    }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(float x, float y, float border = 0.0f) const
    {
        return border <= x && x < (xsize-border) && border <= y && y < (ysize-border);
    }
    
    EIGEN_DEVICE_FUNC inline bool inBounds(const float2 p, float border) const
    {
        return inBounds(p.x, p.y, border);
    }
    
    template<typename Scalar>
    EIGEN_DEVICE_FUNC inline bool inBounds(const Eigen::Matrix<Scalar,2,1>& p, float border) const
    {
        return inBounds(p(0), p(1), border);
    }
    
    EIGEN_DEVICE_FUNC inline std::size_t indexClampedX(int x) const { return clamp(x, 0, (int)xsize-1); }
    EIGEN_DEVICE_FUNC inline std::size_t indexClampedY(int y) const { return clamp(y, 0, (int)ysize-1); }
    
    EIGEN_DEVICE_FUNC inline std::size_t indexCircularX(int x) const { if(x < 0) { return x + xsize; } else if(x >= xsize) { return x - xsize; } else { return x; } }
    EIGEN_DEVICE_FUNC inline std::size_t indexCircularY(int y) const { if(y < 0) { return y + ysize; } else if(y >= ysize) { return y - ysize; } else { return y; } }
    
    EIGEN_DEVICE_FUNC inline std::size_t indexReflectedX(int x) const { if(x < 0) { return -x-1; } else if(x >= xsize) { return 2 * xsize - x - 1; } else { return x; } }
    EIGEN_DEVICE_FUNC inline std::size_t indexReflectedY(int y) const { if(y < 0) { return -y-1; } else if(y >= ysize) { return 2 * ysize - y - 1; } else { return y; } }
protected:
    typename Target<T>::PointerType memptr;
    std::size_t                     line_pitch; 
    std::size_t                     xsize; 
    std::size_t                     ysize;
};

/**
 * Buffer 2D View - Add contents access methods.
 */
template<typename T, template<typename = T> class Target>
class Buffer2DViewAccessible : public Buffer2DViewBase<T,Target>
{
public:
    typedef Buffer2DViewBase<T,Target> BaseT;
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible() : BaseT() { }
    
    EIGEN_DEVICE_FUNC inline ~Buffer2DViewAccessible() { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible(const Buffer2DViewAccessible<T,Target>& img) : BaseT(img) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible(Buffer2DViewAccessible<T,Target>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible(typename Target<T>::PointerType optr, std::size_t w) : BaseT(optr,w,1, w * sizeof(T)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible(typename Target<T>::PointerType optr, std::size_t w, std::size_t h) : BaseT(optr,w,h, w * sizeof(T)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible(typename Target<T>::PointerType optr, std::size_t w, std::size_t h, std::size_t opitch) : BaseT(optr,w,h,opitch) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target>& operator=(const Buffer2DViewAccessible<T,Target>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target>& operator=(Buffer2DViewAccessible<T,Target>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline T* ptr()
    {
        return (T*)BaseT::rawPtr();
    }
    
    EIGEN_DEVICE_FUNC inline const T* ptr() const
    {
        return (T*)BaseT::rawPtr();
    }
    
    EIGEN_DEVICE_FUNC inline T* ptr(std::size_t x, std::size_t y)
    {
        return (T*)( ((unsigned char*)BaseT::rawPtr()) + y * BaseT::pitch()) + x;
    }
    
    EIGEN_DEVICE_FUNC inline const T* ptr(std::size_t x, std::size_t y) const
    {
        return (const T*)( ((const unsigned char*)BaseT::rawPtr()) + y * BaseT::pitch()) + x;
    }
    
    EIGEN_DEVICE_FUNC inline T* rowPtr(std::size_t y)
    {
        return ptr(0,y);
    }
    
    EIGEN_DEVICE_FUNC inline const T* rowPtr(std::size_t y) const
    {
        return ptr(0,y);
    }
    
    EIGEN_DEVICE_FUNC inline T& operator()(std::size_t x, std::size_t y)
    {
        return *ptr(x,y);
    }
    
    EIGEN_DEVICE_FUNC inline const T& operator()(std::size_t x, std::size_t y) const
    {
        return *ptr(x,y);
    }
    
    EIGEN_DEVICE_FUNC inline T& operator[](std::size_t ix)
    {
        return static_cast<T*>(BaseT::rawPtr())[ix];
    }
    
    EIGEN_DEVICE_FUNC inline const T& operator[](std::size_t ix) const
    {
        return static_cast<T*>(BaseT::rawPtr())[ix];
    }
    
    EIGEN_DEVICE_FUNC inline const T& get(std::size_t x, std::size_t y) const
    {
        return *ptr(x,y);
    }
    
    EIGEN_DEVICE_FUNC inline const T& getWithClampedRange(int x, int y) const
    {
        return rowPtr(BaseT::indexClampedY(y))[BaseT::indexClampedX(x)];
    }
    
    EIGEN_DEVICE_FUNC inline const T& getWithCircularRange(int x, int y) const
    {
        return rowPtr(BaseT::indexCircularY(y))[BaseT::indexCircularX(x)];
    }
    
    EIGEN_DEVICE_FUNC inline const T& getWithReflectedRange(int x, int y) const
    {
        return rowPtr(BaseT::indexReflectedY(y))[BaseT::indexReflectedX(x)];
    }
    
    EIGEN_DEVICE_FUNC inline const T& getConditionNeumann(int x, int y) const
    {
        x = ::abs(x);
        if(x >= BaseT::width()) x = (BaseT::width()-1)-(x-BaseT::width());
        
        y = ::abs(y);
        if(y >= BaseT::height()) y = (BaseT::height()-1)-(y-BaseT::height());
        
        return rowPtr(y)[x];
    }
    
    EIGEN_DEVICE_FUNC inline const Buffer2DViewAccessible<T,Target> subBuffer2DView(std::size_t x, std::size_t y, std::size_t width, std::size_t height) const
    {
        return Buffer2DViewAccessible<T,Target>( rowPtr(y)+x, width, height, BaseT::pitch());
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target> subBuffer2DView(std::size_t x, std::size_t y, std::size_t width, std::size_t height)
    {
        return Buffer2DViewAccessible<T,Target>( rowPtr(y)+x, width, height, BaseT::pitch());
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target> row(std::size_t y) const
    {
        return subBuffer2DView(0,y,BaseT::width(),1);
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target> col(std::size_t x) const
    {
        return subBuffer2DView(x,0,1,BaseT::height());
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<T,Target> subBuffer2DView(std::size_t width, std::size_t height)
    {
        return Buffer2DViewAccessible<T,Target>(BaseT::rawPtr(), width, height, BaseT::pitch());
    }
    
    //! Ignore this images pitch - just return new image of size w x h which uses this memory
    template<typename TP> 
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<TP,Target> packedBuffer2DView(std::size_t width, std::size_t height)
    {
        return Buffer2DViewAccessible<TP,Target>((TP*)BaseT::rawPtr(), width, height, width*sizeof(TP) );
    }
    
    template<typename TP>
    EIGEN_DEVICE_FUNC inline Buffer2DViewAccessible<TP,Target> alignedBuffer2DView(std::size_t width, std::size_t height, std::size_t align_bytes = 16)
    {
        const std::size_t wbytes = width * sizeof(TP);
        const std::size_t npitch = (wbytes % align_bytes) == 0 ? wbytes : align_bytes * (1 + wbytes / align_bytes);
        return Buffer2DViewAccessible<TP,Target>((TP*)BaseT::rawPtr(), width, height, npitch );
    }
};

/**
 * View on a 2D Buffer.
 */    
template<typename T, template<typename = T> class Target = TargetHost>
class Buffer2DView
{
 
};

/**
 * View on a 2D Buffer Specialization - Host.
 */ 
template<typename T>
class Buffer2DView<T,TargetHost> : public Buffer2DViewAccessible<T,TargetHost>
{
public:
    typedef Buffer2DViewAccessible<T,TargetHost> BaseT;
    
    inline Buffer2DView() : BaseT() { }
    
    inline ~Buffer2DView() { }
    
    inline Buffer2DView(const Buffer2DView<T,TargetHost>& img) : BaseT(img) { }
    
    inline Buffer2DView(Buffer2DView<T,TargetHost>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer2DView(typename TargetHost<T>::PointerType optr, std::size_t w) : BaseT(optr,w,1, w * sizeof(T)) { }
    
    inline Buffer2DView(typename TargetHost<T>::PointerType optr, std::size_t w, std::size_t h) : BaseT(optr,w,h, w * sizeof(T)) { }
    
    inline Buffer2DView(typename TargetHost<T>::PointerType optr, std::size_t w, std::size_t h, std::size_t opitch) : BaseT(optr,w,h,opitch) { }
    
    inline Buffer2DView<T,TargetHost>& operator=(const Buffer2DView<T,TargetHost>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer2DView<T,TargetHost>& operator=(Buffer2DView<T,TargetHost>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    inline void copyFrom(const Buffer2DView<T,TargetHost>& img)
    {
        typedef TargetHost<T> TargetFrom;
        core::TargetTransfer<TargetFrom,TargetHost<T>>::template memcpy2D<T>(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom::PointerType)img.ptr(), img.pitch(), std::min(img.width(),BaseT::width())*sizeof(T), std::min(img.height(),BaseT::height()));
    }
    
#ifdef CORE_HAVE_CUDA
    inline void copyFrom(const Buffer2DView<T,TargetDeviceCUDA>& img)
    {
        typedef TargetDeviceCUDA<T> TargetFrom;
        core::TargetTransfer<TargetFrom,TargetHost<T>>::template memcpy2D<T>(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom::PointerType)img.ptr(), img.pitch(), std::min(img.width(),BaseT::width())*sizeof(T), std::min(img.height(),BaseT::height()));
    }
#endif // CORE_HAVE_CUDA
    
#ifdef CORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueReadBuffer(img.clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), BaseT::rawPtr(), events, event);
    }

    inline void copyFrom(const cl::CommandQueue& queue, const Image2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseT::width(), img.width());
        region[1] = std::min(BaseT::height(), img.height());
        region[2] = 1; // API says so
        queue.enqueueReadImage(img.clType(), true, origin, region, 0, 0, BaseT::rawPtr(), events, event);
    }
#endif // CORE_HAVE_OPENCL

    inline void memset(unsigned char v = 0)
    {
        TargetHost<T>::memset2D(BaseT::rawPtr(), BaseT::pitch(),  v, BaseT::width() * sizeof(T), BaseT::height());
    }
    
    inline T* begin() { return BaseT::ptr(); }
    inline const T* begin() const { return BaseT::ptr(); }
    inline T* end() { return (BaseT::ptr() + BaseT::totalElements()); }
    inline const T* end() const { return (BaseT::ptr() + BaseT::totalElements()); }
};

#ifdef CORE_HAVE_CUDA

/**
 * View on a 2D Buffer Specialization - CUDA.
 */ 
template<typename T>
class Buffer2DView<T,TargetDeviceCUDA> : public Buffer2DViewAccessible<T,TargetDeviceCUDA>
{
public:
    typedef Buffer2DViewAccessible<T,TargetDeviceCUDA> BaseT;
    
    EIGEN_DEVICE_FUNC inline Buffer2DView() : BaseT() { }
    
    EIGEN_DEVICE_FUNC inline ~Buffer2DView() { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView(const Buffer2DView<T,TargetDeviceCUDA>& img) : BaseT(img) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView(Buffer2DView<T,TargetDeviceCUDA>&& img) : BaseT(std::move(img)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView(typename TargetDeviceCUDA<T>::PointerType optr, std::size_t w) : BaseT(optr,w,1, w * sizeof(T)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView(typename TargetDeviceCUDA<T>::PointerType optr, std::size_t w, std::size_t h) : BaseT(optr,w,h, w * sizeof(T)) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView(typename TargetDeviceCUDA<T>::PointerType optr, std::size_t w, std::size_t h, std::size_t opitch) : BaseT(optr,w,h,opitch) { }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetDeviceCUDA>& operator=(const Buffer2DView<T,TargetDeviceCUDA>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetDeviceCUDA>& operator=(Buffer2DView<T,TargetDeviceCUDA>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }

    template<template<typename = T> class TargetFrom>
    inline void copyFrom(const Buffer2DView<T,TargetFrom>& img)
    {
        core::TargetTransfer<TargetFrom<T>,TargetDeviceCUDA<T>>::memcpy2D(BaseT::rawPtr(), BaseT::pitch(), (typename TargetFrom<T>::PointerType)img.ptr(), img.pitch(), std::min(img.width(),BaseT::width())*sizeof(T), std::min(img.height(),BaseT::height()));
    }
    
    inline void memset(unsigned char v = 0)
    {
        TargetDeviceCUDA<T>::memset2D(BaseT::rawPtr(), BaseT::pitch(),  v, BaseT::width() * sizeof(T), BaseT::height());
    }

    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetDeviceCUDA> subBuffer2DView(const NppiRect& region)
    {
        return Buffer2DView<T,TargetDeviceCUDA>(BaseT::rowPtr(region.y)+region.x, region.width, region.height, BaseT::pitch());
    }

    EIGEN_DEVICE_FUNC inline Buffer2DView<T,TargetDeviceCUDA> subBuffer2DView(const NppiSize& size)
    {
        return Buffer2DView<T,TargetDeviceCUDA>(BaseT::rawPtr(), size.width,size.height, BaseT::pitch());
    }

    inline const NppiSize size() const
    {
        NppiSize ret = {(int)BaseT::width(),(int)BaseT::height()};
        return ret;
    }

    inline const NppiRect rect() const
    {
        NppiRect ret = {0,0,BaseT::width(),BaseT::height()};
        return ret;
    }

    EIGEN_DEVICE_FUNC inline typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr begin() 
    {
        return (typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr)((T*)BaseT::rawPtr());
    }

    EIGEN_DEVICE_FUNC inline typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr end() 
    {
        return (typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr)( BaseT::rowPtr(BaseT::height()-1) + BaseT::width() );
    }
    
    EIGEN_DEVICE_FUNC inline typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr begin() const
    {
        return (typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr)(const_cast<T*>((T*)BaseT::rawPtr()));
    }
    
    EIGEN_DEVICE_FUNC inline typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr end() const
    {
        return (typename core::internal::ThrustType<T,TargetDeviceCUDA<T>>::Ptr)( const_cast<T*>(BaseT::rowPtr(BaseT::height()-1) + BaseT::width()) );
    }

    inline void fill(T val) 
    {
        thrust::fill(begin(), end(), val);
    }
};

#endif // CORE_HAVE_CUDA

#ifdef CORE_HAVE_OPENCL

/**
 * View on a 2D Buffer Specialization - OpenCL.
 */ 
template<typename T>
class Buffer2DView<T,TargetDeviceOpenCL> : public Buffer2DViewBase<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer2DViewBase<T,TargetDeviceOpenCL> BaseT;
    
    inline Buffer2DView() : BaseT() { }
    
    inline ~Buffer2DView() { }
    
    inline Buffer2DView(const Buffer2DView<T,TargetDeviceOpenCL>& img) : BaseT(img) { }
    
    inline Buffer2DView(Buffer2DView<T,TargetDeviceOpenCL>&& img) : BaseT(std::move(img)) { }
    
    inline Buffer2DView(typename TargetDeviceOpenCL<T>::PointerType optr, std::size_t w) : BaseT(optr,w,1, w * sizeof(T)) { }
    
    inline Buffer2DView(typename TargetDeviceOpenCL<T>::PointerType optr, std::size_t w, std::size_t h) : BaseT(optr,w,h, w * sizeof(T)) { }
    
    inline Buffer2DView(typename TargetDeviceOpenCL<T>::PointerType optr, std::size_t w, std::size_t h, std::size_t opitch) : BaseT(optr,w,h,opitch) { }
    
    inline Buffer2DView<T,TargetDeviceOpenCL>& operator=(const Buffer2DView<T,TargetDeviceOpenCL>& img)
    {
        BaseT::operator=(img);
        return *this;
    }
    
    inline Buffer2DView<T,TargetDeviceOpenCL>& operator=(Buffer2DView<T,TargetDeviceOpenCL>&& img)
    {
        BaseT::operator=(std::move(img));
        return *this;
    }
    
    inline const cl::Buffer& clType() const { return *static_cast<const cl::Buffer*>(BaseT::rawPtr()); }
    inline cl::Buffer& clType() { return *static_cast<cl::Buffer*>(BaseT::rawPtr()); }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer2DView<T,TargetHost>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueWriteBuffer(clType(), true, 0, std::min(BaseT::bytes(), img.bytes()), img.rawPtr(), events, event);
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueCopyBuffer(img.clType(), clType(), 0, 0, std::min(BaseT::bytes(), img.bytes()), events, event);
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Image2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseT::width(), img.width());
        region[1] = std::min(BaseT::height(), img.height());
        region[2] = 1; // API says so
        
        queue.enqueueCopyImageToBuffer(img.clType(), clType(), origin, region, 0, events, event);
    }
    
    inline void memset(const cl::CommandQueue& queue, T v, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueFillBuffer(clType(), v, 0, BaseT::bytes(), events, event);
    }
};

/**
 * OpenCL Buffer to Host Mapping.
 */
struct Buffer2DMapper
{
    template<typename T>
    static inline Buffer2DView<T,TargetHost> map(const cl::CommandQueue& queue, cl_map_flags flags, const Buffer2DView<T,TargetDeviceOpenCL>& buf, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        typename TargetHost<T>::PointerType ptr = queue.enqueueMapBuffer(buf.clType(), true, flags, 0, buf.bytes(), events, event);
        return Buffer2DView<T,TargetHost>(ptr, buf.width(), buf.height());
    }
    
    template<typename T>
    static inline void unmap(const cl::CommandQueue& queue, const Buffer2DView<T,TargetDeviceOpenCL>& buf, const Buffer2DView<T,TargetHost>& bufcpu,  const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueUnmapMemObject(buf.clType(), bufcpu.rawPtr(), events, event);
    }
};

#endif // CORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * CUDA/Host Buffer 2D Creation.
 */
template<typename T, template<typename = T> class Target = TargetHost>
class Buffer2DManaged : public Buffer2DView<T, Target>
{
public:
    typedef Buffer2DView<T, Target> ViewT;
    
    Buffer2DManaged() = delete;
    
    inline Buffer2DManaged(std::size_t w, std::size_t h) : ViewT()
    {        
        typename Target<T>::PointerType ptr = nullptr;
        std::size_t line_pitch = 0;
        ViewT::xsize = w;
        ViewT::ysize = h;
        
        Target<T>::AllocatePitchedMem(&ptr, &line_pitch, w, h);
        
        ViewT::memptr = ptr;
        ViewT::line_pitch = line_pitch;
    }
    
    inline ~Buffer2DManaged()
    {
        if(ViewT::isValid())
        {
            Target<T>::DeallocatePitchedMem(ViewT::memptr);
        }
    }
    
    Buffer2DManaged(const Buffer2DManaged<T,Target>& img) = delete;
    
    inline Buffer2DManaged(Buffer2DManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer2DManaged<T,Target>& operator=(const Buffer2DManaged<T,Target>& img) = delete;
    
    inline Buffer2DManaged<T,Target>& operator=(Buffer2DManaged<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    void resize(std::size_t new_w, std::size_t new_h)
    {
        if(ViewT::isValid())
        {
            Target<T>::DeallocatePitchedMem(ViewT::memptr);
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::memptr = nullptr;
        }
        
        ViewT::memptr = nullptr;
        ViewT::xsize = new_w;
        ViewT::ysize = new_h;
        
        typename Target<T>::PointerType ptr = nullptr;
        std::size_t line_pitch = 0;
        Target<T>::AllocatePitchedMem(&ptr, &line_pitch, new_w, new_h);
        
        ViewT::memptr = ptr;
        ViewT::line_pitch = line_pitch;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef CORE_HAVE_OPENCL

/**
 * OpenCL Buffer 2D Creation.
 */
template<typename T>
class Buffer2DManaged<T,TargetDeviceOpenCL> : public Buffer2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer2DView<T,TargetDeviceOpenCL> ViewT;
    
    Buffer2DManaged() = delete;
    
    inline Buffer2DManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags, typename TargetHost<T>::PointerType hostptr = nullptr) : ViewT()
    {        
        ViewT::memptr = new cl::Buffer(context, flags, w*h*sizeof(T), hostptr);
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::line_pitch = w * sizeof(T);
    }
    
    inline ~Buffer2DManaged()
    {
        if(ViewT::isValid())
        {
            cl::Buffer* clb = static_cast<cl::Buffer*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::line_pitch = 0;
        }
    }
    
    Buffer2DManaged(const Buffer2DManaged<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer2DManaged(Buffer2DManaged<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Buffer2DManaged<T,TargetDeviceOpenCL>& operator=(const Buffer2DManaged<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer2DManaged<T,TargetDeviceOpenCL>& operator=(Buffer2DManaged<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }

    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#endif // CORE_HAVE_OPENCL

}

#endif // CORE_BUFFER2D_HPP
