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
 * Image access to 2D Host/Device Buffer.
 * ****************************************************************************
 */

#ifndef CORE_IMAGE2D_HPP
#define CORE_IMAGE2D_HPP

#include <Platform.hpp>

#include <image/PixelConvert.hpp>

#include <buffers/Buffer2D.hpp>

#ifdef CORE_HAVE_OPENCV
#include <opencv2/core/core.hpp>
#endif // CORE_HAVE_OPENCV

namespace core
{

namespace internal
{
#ifdef CORE_HAVE_OPENCV
// OpenCV traits
template<typename T> struct OpenCVType { };
template<> struct OpenCVType<uint8_t> { static const int TypeCode = CV_8U; };
template<> struct OpenCVType<int8_t> { static const int TypeCode = CV_8S; };
template<> struct OpenCVType<uint16_t> { static const int TypeCode = CV_16U; };
template<> struct OpenCVType<int16_t> { static const int TypeCode = CV_16S; };
template<> struct OpenCVType<int32_t> { static const int TypeCode = CV_32S; };
template<> struct OpenCVType<float> { static const int TypeCode = CV_32F; };
template<> struct OpenCVType<double> { static const int TypeCode = CV_64F; };
#endif // CORE_HAVE_OPENCV
}

/**
 * Image 2D View. CUDA + Host. 
 * Image specific functions. Interpolations, derivatives.
 */
template<typename T, template<typename = T> class Target>
class Image2DView : public Buffer2DView<T,Target>
{
public:
    typedef Buffer2DView<T,Target> BaseType;
    typedef typename core::type_traits<T>::ChannelType ValueType;
    static const int Channels = core::type_traits<T>::ChannelCount;
    
    using BaseType::ptr;
    using BaseType::rowPtr;
    using BaseType::get;
    using BaseType::width;
    using BaseType::height;
    using BaseType::pitch;
    using BaseType::inBounds;
    using BaseType::operator=;
    using BaseType::copyFrom;
    
    EIGEN_DEVICE_FUNC inline Image2DView() : BaseType() { }
    
    EIGEN_DEVICE_FUNC inline ~Image2DView()
    {
        
    }

    EIGEN_DEVICE_FUNC inline Image2DView(const Image2DView<T,Target>& img)  : BaseType(img)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Image2DView(const BaseType& img)  : BaseType(img)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Image2DView(Image2DView<T,Target>&& img) : BaseType(std::move(img))
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Image2DView<T,Target>& operator=(const Image2DView<T,Target>& img)
    {
        BaseType::operator=(img);
        return *this;
    }
        
    EIGEN_DEVICE_FUNC inline Image2DView<T,Target>& operator=(Image2DView<T,Target>&& img)
    {
        BaseType::operator=(std::move(img));
        return *this;
    }
  
    EIGEN_DEVICE_FUNC inline Image2DView(T* ptr, std::size_t w, std::size_t h) : BaseType(ptr, w, h)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline Image2DView(T* ptr, std::size_t w, std::size_t h, std::size_t pitch) : BaseType(ptr, w, h, pitch)
    {
        
    }
    
#ifdef CORE_HAVE_OPENCV
    inline Image2DView( const cv::Mat& img ) : BaseType((T*)img.data, (std::size_t)img.cols, (std::size_t)img.rows, (std::size_t)img.step)
    {
        static_assert(std::is_same<Target<T>,TargetHost<T>>::value == true, "Only possible on TargetHost buffers");
        assert(img.type() == CV_MAKETYPE(internal::OpenCVType<ValueType>::TypeCode, Channels) && "This wont work nicely");
    }
#endif

    EIGEN_DEVICE_FUNC inline const BaseType& buffer() const { return (const BaseType&)*this; }
    EIGEN_DEVICE_FUNC inline BaseType& buffer() { return (BaseType&)*this; }

    EIGEN_DEVICE_FUNC inline T getBilinear(float u, float v) const
    {
        const float ix = floorf(u);
        const float iy = floorf(v);
        const float fx = u - ix;
        const float fy = v - iy;
        
        const T* bl = rowPtr(iy)  + (std::size_t)ix;
        const T* tl = rowPtr(iy+1)+ (std::size_t)ix;
        
        return lerp(
            lerp( bl[0], bl[1], fx ),
                    lerp( tl[0], tl[1], fx ),
                    fy
        );
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getBilinear(const float2 p) const
    {
        return getBilinear<TR>(p.x, p.y);
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getBilinear(const Eigen::Matrix<float,2,1>& p) const
    {
        return getBilinear<TR>(p(0), p(1));
    }

    EIGEN_DEVICE_FUNC inline T getNearestNeighbour(float u, float v) const
    {
        return get(u+0.5, v+0.5);
    }
    
    EIGEN_DEVICE_FUNC inline T getNearestNeighbour(const float2 p) const
    {
        return getNearestNeighbour(p.x,p.y);
    }
    
    EIGEN_DEVICE_FUNC inline T getNearestNeighbour(const Eigen::Matrix<float,2,1>& pt) const
    {
        return getNearestNeighbour(pt(0),pt(1));
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getBackwardDiffDx(std::size_t x, std::size_t y) const
    {
        const T* row = rowPtr(y);
        return ( image::convertPixel<TR,T>(row[x]) - image::convertPixel<TR,T>(row[x-1]) );
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getBackwardDiffDy(std::size_t x, std::size_t y) const
    {
        return ( image::convertPixel<TR,T>(get(x,y)) - image::convertPixel<TR,T>(get(x,y-1)) );
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getCentralDiffDx(std::size_t x, std::size_t y) const
    {
        const T* row = rowPtr(y);
        return ( image::convertPixel<TR,T>(row[x+1]) - image::convertPixel<TR,T>(row[x-1]) ) / 2;
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline TR getCentralDiffDy(std::size_t x, std::size_t y) const
    {
        return ( image::convertPixel<TR,T>(get(x,y+1)) - image::convertPixel<TR,T>(get(x,y-1)) ) / 2;
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<TR,1,2> getCentralDiff(std::size_t px, std::size_t py) const
    {
        Eigen::Matrix<TR,1,2> res;
        res(0,0) = getCentralDiffDx<TR>(px,py);
        res(0,1) = getCentralDiffDy<TR>(px,py);
        return res;
    }

    template<typename TR>
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<TR,1,2> getCentralDiff(float px, float py) const
    {
        // TODO: Make more efficient by expanding GetCentralDiff calls
        const int ix = floorf(px);
        const int iy = floorf(py);
        const float fx = px - ix;
        const float fy = py - iy;
        
        const int b = py;   const int l = px;
        const int t = py+1; const int r = px+1;
        
        const TR tldx = getCentralDiffDx<TR>(l,t);
        const TR trdx = getCentralDiffDx<TR>(r,t);
        const TR bldx = getCentralDiffDx<TR>(l,b);
        const TR brdx = getCentralDiffDx<TR>(r,b);
        const TR tldy = getCentralDiffDy<TR>(l,t);
        const TR trdy = getCentralDiffDy<TR>(r,t);
        const TR bldy = getCentralDiffDy<TR>(l,b);
        const TR brdy = getCentralDiffDy<TR>(r,b);
        
        Eigen::Matrix<TR,1,2> res;
        res(0,0) = lerp(lerp(bldx,brdx,fx), lerp(tldx,trdx,fx), fy);
        res(0,1) = lerp(lerp(bldy,brdy,fx), lerp(tldy,trdy,fx), fy);
        return res;
    }
    
    template<typename TR>
    EIGEN_DEVICE_FUNC inline Eigen::Matrix<TR,1,2> getCentralDiff(const Eigen::Matrix<float,2,1>& pt) const
    {
        return getCentralDiff<TR>(pt(0), pt(1));
    }
    
    template<typename TJET>
    EIGEN_DEVICE_FUNC inline TJET getJet(const Eigen::Matrix<TJET,2,1>& pt) const
    {
        return getJet<TJET>(pt(0), pt(1));
    }

    template<typename TJET>
    EIGEN_DEVICE_FUNC inline TJET getJet(const TJET& x, const TJET& y) const
    {
        typename core::ADTraits<TJET>::Scalar scalar_x = core::ADTraits<TJET>::getScalar(x);
        typename core::ADTraits<TJET>::Scalar scalar_y = core::ADTraits<TJET>::getScalar(y);
        
        typename core::ADTraits<TJET>::Scalar sample[3];
        if(core::ADTraits<TJET>::isScalar()) 
        {
            // For the scalar case, only sample the image.
            if(inBounds(scalar_x, scalar_y, 1.0f))
            {
                sample[0] = getNearestNeighbour(scalar_x, scalar_y);
            }
            else
            {
                sample[0] = 0.0f;
            }
        }
        else 
        {
            Eigen::Map<Eigen::Matrix<typename core::ADTraits<TJET>::Scalar,1,2> > tmp(sample + 1,2);
            
            sample[0] = 0.0f;
            // For the derivative case, sample the gradient as well.
            if(inBounds(scalar_x, scalar_y, 1.0f))
            {
                sample[0] = getNearestNeighbour(scalar_x, scalar_y); // pixel value
            }
            
            tmp << 0.0f , 0.0f;
            if(inBounds(scalar_x, scalar_y, 2.0f))
            {
                tmp = getCentralDiff<typename core::ADTraits<TJET>::Scalar>(scalar_x, scalar_y);
            }   
        }
        
        TJET xy[2] = { x, y };
        return core::Chain<typename core::ADTraits<TJET>::Scalar, 2, TJET>::Rule(sample[0], sample + 1, xy);
    }
    
#ifdef CORE_HAVE_OPENCV
    inline cv::Mat getOpenCV() const
    {
        static_assert(std::is_same<Target<T>,TargetHost<T>>::value == true, "Only possible on TargetHost buffers");
        return cv::Mat(height(), width(), CV_MAKETYPE(internal::OpenCVType<ValueType>::TypeCode, Channels), (void*)BaseType::ptr(), BaseType::pitch());
    }

    inline void copyFrom(const cv::Mat& img)
    {
        assert(img.cols == (int)width());
        assert(img.rows == (int)height());
        assert(img.type() == CV_MAKETYPE(internal::OpenCVType<ValueType>::TypeCode, Channels) && "This wont work nicely");
        
        Image2DView<T,TargetHost> proxy((T*)img.data, (std::size_t)img.cols, (std::size_t)img.rows, (std::size_t)img.step);
        BaseType::copyFrom(proxy);
    }
#endif // CORE_HAVE_OPENCV

#ifdef CORE_HAVE_OPENCL
    inline void copyFrom(const cl::CommandQueue& queue, const Image2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        static_assert(std::is_same<Target<T>,TargetHost<T>>::value == true, "Only possible on TargetHost buffers");
        
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseType::width(), img.width());
        region[1] = std::min(BaseType::height(), img.height());
        region[2] = 1; // API says so
        queue.enqueueReadImage(img.clType(), true, origin, region, 0, 0, BaseType::rawPtr(), events, event);
    }
#endif // CORE_HAVE_OPENCL
};

#ifdef CORE_HAVE_OPENCL

/**
 * Image 2D View. OpenCL Specialization.
 * @note Image2D object here.
 */
template<typename T>
class Image2DView<T,TargetDeviceOpenCL> : public Buffer2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer2DView<T,TargetDeviceOpenCL> BaseType;
    typedef typename core::type_traits<T>::ChannelType ValueType;
    static const int Channels = core::type_traits<T>::ChannelCount;
    
    using BaseType::width;
    using BaseType::height;
    using BaseType::pitch;
    using BaseType::inBounds;
    using BaseType::operator=;
    
    inline Image2DView() : BaseType() { }
    
    inline ~Image2DView()
    {
        
    }
    
    inline Image2DView(const Image2DView<T,TargetDeviceOpenCL>& img)  : BaseType(img)
    {
        
    }
    
    inline Image2DView(const BaseType& img)  : BaseType(img)
    {
        
    }
    
    inline Image2DView(Image2DView<T,TargetDeviceOpenCL>&& img) : BaseType(std::move(img))
    {
        
    }
    
    inline Image2DView<T,TargetDeviceOpenCL>& operator=(const Image2DView<T,TargetDeviceOpenCL>& img)
    {
        BaseType::operator=(img);
        return *this;
    }
    
    inline Image2DView<T,TargetDeviceOpenCL>& operator=(Image2DView<T,TargetDeviceOpenCL>&& img)
    {
        BaseType::operator=(std::move(img));
        return *this;
    }
    
    inline Image2DView(typename TargetDeviceOpenCL<T>::PointerType ptr, std::size_t w, std::size_t h) : BaseType(ptr, w, h)
    {
        
    }
    
    inline Image2DView(typename TargetDeviceOpenCL<T>::PointerType ptr, std::size_t w, std::size_t h, std::size_t pitch) : BaseType(ptr, w, h, pitch)
    {
        
    }
    
    inline const cl::Image2D& clType() const { return *static_cast<const cl::Image2D*>(BaseType::rawPtr()); }
    inline cl::Image2D& clType() { return *static_cast<cl::Image2D*>(BaseType::rawPtr()); }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Image2DView<T,TargetHost>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseType::width(), img.width());
        region[1] = std::min(BaseType::height(), img.height());
        region[2] = 1; // API says so
        queue.enqueueWriteImage(clType(), true, origin, region, 0, 0, img.rawPtr(), events, event);
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Image2DView<T,TargetDeviceOpenCL>& img, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseType::width(), img.width());
        region[1] = std::min(BaseType::height(), img.height());
        region[2] = 1; // API says so
        queue.enqueueCopyImage(img.clType(), clType(), origin, origin, region, events, event);
    }
    
    inline void copyFrom(const cl::CommandQueue& queue, const Buffer2DView<T, TargetDeviceOpenCL>& buf, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = std::min(BaseType::width(), buf.width());
        region[1] = std::min(BaseType::height(), buf.height());
        region[2] = 1; // API says so
        
        queue.enqueueCopyBufferToImage(buf.clType(), clType(), 0, origin, region, events, event);
    }
    
    inline void memset(const cl::CommandQueue& queue, cl_float4 v, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = BaseType::width();
        region[1] = BaseType::height();
        region[2] = 1; // API says so
        
        queue.enqueueFillImage(clType(), v, origin, region, events, event);
    }
    
#ifdef CORE_HAVE_OPENCV
    inline void copyFrom(const cl::CommandQueue& queue, const cv::Mat& img)
    {
        Image2DView<T,TargetHost> proxy((T*)img.data, (std::size_t)img.cols, (std::size_t)img.rows, (std::size_t)img.step);
        copyFrom(proxy);
    }
#endif // CORE_HAVE_OPENCV
};

#endif // CORE_HAVE_OPENCL

/// ***********************************************************************

/**
 * CUDA/Host Image 2D Creation.
 */
template<typename T, template<typename = T> class Target = TargetHost>
class Image2DManaged : public Image2DView<T,Target>
{
public:
    typedef Image2DView<T,Target> ViewT;
    
    Image2DManaged() = delete;
    
    inline Image2DManaged(std::size_t w, std::size_t h) : ViewT()
    {
        typename Target<T>::PointerType ptr = nullptr;
        std::size_t line_pitch = 0;
        ViewT::xsize = w;
        ViewT::ysize = h;
        
        Target<T>::AllocatePitchedMem(&ptr, &line_pitch, w, h);
        
        ViewT::memptr = ptr;
        ViewT::line_pitch = line_pitch;
    }
    
    inline ~Image2DManaged()
    {
        if(ViewT::isValid())
        {
            Target<T>::DeallocatePitchedMem(ViewT::memptr);
        }
    }
    
    Image2DManaged(const Image2DManaged<T,Target>& img) = delete;
    
    inline Image2DManaged(Image2DManaged<T,Target>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Image2DManaged<T,Target>& operator=(const Image2DManaged<T,Target>& img) = delete;
    
    inline Image2DManaged<T,Target>& operator=(Image2DManaged<T,Target>&& img)
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

namespace internal
{
    
    template<typename T>
    struct TypeToOpenCLChannelType;
    
    template<> struct TypeToOpenCLChannelType<uint8_t> { static const cl_channel_type ChannelType = CL_UNORM_INT8; };
    template<> struct TypeToOpenCLChannelType<int8_t> { static const cl_channel_type ChannelType = CL_SNORM_INT8; };
    template<> struct TypeToOpenCLChannelType<uint16_t> { static const cl_channel_type ChannelType = CL_UNORM_INT16; };
    template<> struct TypeToOpenCLChannelType<int16_t> { static const cl_channel_type ChannelType = CL_SNORM_INT16; };
    template<> struct TypeToOpenCLChannelType<int32_t> { static const cl_channel_type ChannelType = CL_SIGNED_INT32; };
    template<> struct TypeToOpenCLChannelType<float> { static const cl_channel_type ChannelType = CL_FLOAT; };
    
    template<int chn>
    struct ChannelCountToOpenCLChannelOrder;
    
    template<> struct ChannelCountToOpenCLChannelOrder<1> { static const cl_channel_order ChannelOrder = CL_R; };
    template<> struct ChannelCountToOpenCLChannelOrder<2> { static const cl_channel_order ChannelOrder = CL_RG; };
    template<> struct ChannelCountToOpenCLChannelOrder<3> { static const cl_channel_order ChannelOrder = CL_RGB; };
    template<> struct ChannelCountToOpenCLChannelOrder<4> { static const cl_channel_order ChannelOrder = CL_RGBA; };
    
}

/**
 * OpenCL Image 2D Creation.
 */
template<typename T>
class Image2DManaged<T,TargetDeviceOpenCL> : public Image2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Image2DView<T,TargetDeviceOpenCL> ViewT;
    
    Image2DManaged() = delete;
    
    inline Image2DManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags, const cl::ImageFormat& fmt, typename TargetHost<T>::PointerType hostptr = nullptr) : ViewT()
    {
        ViewT::line_pitch = w * sizeof(T);
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::memptr = new cl::Image2D(context, flags, fmt, w, h, 0, hostptr);   
    }
    
    inline Image2DManaged(std::size_t w, std::size_t h, const cl::Context& context, cl_mem_flags flags, typename TargetHost<T>::PointerType hostptr = nullptr) : ViewT()
    {
        ViewT::line_pitch = w * sizeof(T);
        ViewT::xsize = w;
        ViewT::ysize = h;
        ViewT::memptr = new cl::Image2D(context, flags, 
                                        cl::ImageFormat(internal::ChannelCountToOpenCLChannelOrder<ViewT::Channels>::ChannelOrder,
                                                        internal::TypeToOpenCLChannelType<typename ViewT::ValueType>::ChannelType)
                                        , w, h, 0, hostptr);   
    }
    
    inline ~Image2DManaged()
    {
        if(ViewT::isValid())
        {
            cl::Image2D* clb = static_cast<cl::Image2D*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::line_pitch = 0;
        }
    }
    
    Image2DManaged(const Image2DManaged<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Image2DManaged(Image2DManaged<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img))
    {
        
    }
    
    Image2DManaged<T,TargetDeviceOpenCL>& operator=(const Image2DManaged<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Image2DManaged<T,TargetDeviceOpenCL>& operator=(Image2DManaged<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

struct Image2DMapper
{
    template<typename T>
    static inline Image2DView<T,TargetHost> map(const cl::CommandQueue& queue, cl_map_flags flags, const Image2DView<T,TargetDeviceOpenCL>& buf, const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        std::array<std::size_t,3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = buf.width();
        region[1] = buf.height();
        region[2] = 1; // API says so
        
        std::size_t line_pitch = 0;
        
        typename TargetHost<T>::PointerType ptr = queue.enqueueMapImage(buf.clType(), true, flags, origin, region, &line_pitch, nullptr, events, event);
        return Image2DView<T,TargetHost>(ptr, buf.width(), buf.height(), line_pitch);
    }
    
    template<typename T>
    static inline void unmap(const cl::CommandQueue& queue, const Image2DView<T,TargetDeviceOpenCL>& buf, const Image2DView<T,TargetHost>& bufcpu,  const std::vector<cl::Event>* events = nullptr, cl::Event* event = nullptr)
    {
        queue.enqueueUnmapMemObject(buf.clType(), bufcpu.rawPtr(), events, event);
    }
};

#endif // CORE_HAVE_OPENCL
    
}

#endif // CORE_IMAGE2D_HPP
