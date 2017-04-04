/**
 * ****************************************************************************
 * Copyright (c) 2016, Robert Lukierski.
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
 * Image IO.
 * ****************************************************************************
 */

#include <buffers/Buffer2D.hpp>
#include <buffers/Image2D.hpp>

#include <io/ImageIO.hpp>

#ifdef CORE_HAVE_CAMERA_DRIVERS
#include <CameraDrivers.hpp>
#endif // CORE_HAVE_CAMERA_DRIVERS

#ifdef CORE_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif // CORE_HAVE_OPENCV

struct OpenCVBackend { };


template<typename T, typename Backend>
struct ImageIOProxy
{
    static inline T load(const std::string& fn) { throw std::runtime_error("No IO backend"); }
    static inline void save(const std::string& fn, const T& input) { throw std::runtime_error("No IO backend"); }
};

/**
 * OpenCV Backend.
 */
#ifdef CORE_HAVE_OPENCV
template<>
struct ImageIOProxy<cv::Mat,OpenCVBackend>
{
    static inline cv::Mat load(const std::string& fn)
    {
        cv::Mat ret = cv::imread(fn);
        
        if(!ret.data)
        {
            throw std::runtime_error("File not found");
        }
        
        return ret;
    }
    
    static inline void save(const std::string& fn, const cv::Mat& input)
    {
        cv::imwrite(fn, input);
    }
};

#ifdef CORE_HAVE_CAMERA_DRIVERS

static inline drivers::camera::EPixelFormat OpenCVType2PixelFormat(int ocv_type)
{
    switch(ocv_type)
    {
        case CV_8UC1:  return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8;
        case CV_16UC1: return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16;
        case CV_8UC3:  return drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8;
        case CV_32FC1: return drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO32F;
        case CV_32FC3: return drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB32F;
        default: throw std::runtime_error("Pixel Format not supported");
    }
}

static inline int PixelFomat2OpenCVType(drivers::camera::EPixelFormat pf)
{
    switch(pf)
    {
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO8: return CV_8UC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO16: return CV_16UC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8: return CV_8UC3;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_MONO32F: return CV_32FC1;
        case drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB32F: return CV_32FC3;
        default: throw std::runtime_error("Pixel Format not supported");
    }
}

template<>
struct ImageIOProxy<drivers::camera::FrameBuffer, OpenCVBackend>
{
    static inline drivers::camera::FrameBuffer load(const std::string& fn)
    {
        cv::Mat cv_tmp = cv::imread(fn);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        drivers::camera::FrameBuffer ret(cv_tmp.cols, cv_tmp.rows, OpenCVType2PixelFormat(cv_tmp.type()));
        cv::Mat cv_wrapper(ret.getHeight(), ret.getWidth(), cv_tmp.type(), ret.getData());
        
        cv_tmp.copyTo(cv_wrapper);
        
        return ret;
    }
    
    static inline void save(const std::string& fn, const drivers::camera::FrameBuffer& input)
    {
        cv::Mat cv_wrapper(input.getHeight(), input.getWidth(), PixelFomat2OpenCVType(input.getPixelFormat()), (void*)input.getData());
        cv::imwrite(fn, cv_wrapper);
    }
};
#endif // IMAGE_UTILS_HAVE_CAMERA_DRIVERS

template<typename T>
struct ImageIOProxy<core::Image2DView<T, core::TargetHost>,OpenCVBackend>
{
    static inline void save(const std::string& fn, const core::Image2DView<T, core::TargetHost>& input)
    {
        cv::Mat cv_proxy(input.height(), input.width(), CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount), (void*)input.ptr());
        cv::imwrite(fn, cv_proxy);
    }
};

template<typename T>
struct ImageIOProxy<core::Image2DManaged<T, core::TargetHost>,OpenCVBackend>
{
    static inline core::Image2DManaged<T, core::TargetHost> load(const std::string& fn)
    {
        int flag = 0;
        if(core::type_traits<T>::ChannelCount == 1)
        {
            flag = CV_LOAD_IMAGE_GRAYSCALE;
        }
        else
        {
            flag = CV_LOAD_IMAGE_COLOR;
        }
        
        cv::Mat cv_tmp = cv::imread(fn, flag | cv::IMREAD_ANYDEPTH);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        if((core::type_traits<T>::ChannelCount != cv_tmp.channels()) || (CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount) != cv_tmp.type()))
        {
            throw std::runtime_error("Wrong channel count / pixel type");
        }
        
        core::Image2DManaged<T, core::TargetHost> ret(cv_tmp.cols, cv_tmp.rows);
        cv::Mat cv_proxy(ret.height(), ret.width(), CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount), (void*)ret.ptr());
        
        // copy
        cv_tmp.copyTo(cv_proxy);
        
        return ret;
    }
};

template<typename T>
struct ImageIOProxy<core::Buffer2DView<T, core::TargetHost>,OpenCVBackend>
{
    static inline void save(const std::string& fn, const core::Buffer2DView<T, core::TargetHost>& input)
    {
        cv::Mat cv_proxy(input.height(), input.width(), CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount), (void*)input.ptr());
        cv::imwrite(fn, cv_proxy);
    }
};

template<typename T>
struct ImageIOProxy<core::Buffer2DManaged<T, core::TargetHost>,OpenCVBackend>
{
    static inline core::Buffer2DManaged<T, core::TargetHost> load(const std::string& fn)
    {
        int flag = 0;
        if(core::type_traits<T>::ChannelCount == 1)
        {
            flag = CV_LOAD_IMAGE_GRAYSCALE;
        }
        else
        {
            flag = CV_LOAD_IMAGE_COLOR;
        }
        cv::Mat cv_tmp = cv::imread(fn, flag | cv::IMREAD_ANYDEPTH);
        
        if(!cv_tmp.data)
        {
            throw std::runtime_error("File not found");
        }
        
        if((core::type_traits<T>::ChannelCount != cv_tmp.channels()) || (CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount) != cv_tmp.type()))
        {
            throw std::runtime_error("Wrong channel count / pixel type");
        }
        
        core::Buffer2DManaged<T, core::TargetHost> ret(cv_tmp.cols, cv_tmp.rows);
        cv::Mat cv_proxy(ret.height(), ret.width(), CV_MAKETYPE(core::internal::OpenCVType<typename core::type_traits<T>::ChannelType>::TypeCode, core::type_traits<T>::ChannelCount), (void*)ret.ptr());
        
        // copy
        cv_tmp.copyTo(cv_proxy);
        
        return ret;
    }
};
#endif //CORE_HAVE_OPENCV

template<typename T>
T core::io::loadImage(const std::string& fn)
{
    return ImageIOProxy<T,OpenCVBackend>::load(fn);
}

template<typename T>
void core::io::saveImage(const std::string& fn, const T& input)
{
    ImageIOProxy<T,OpenCVBackend>::save(fn, input);
}

#ifdef CORE_HAVE_OPENCV
template cv::Mat core::io::loadImage<cv::Mat>(const std::string& fn);
template void core::io::saveImage<cv::Mat>(const std::string& fn, const cv::Mat& input);
#endif // CORE_HAVE_OPENCV

template core::Image2DManaged<float, core::TargetHost> core::io::loadImage<core::Image2DManaged<float, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Image2DView<float, core::TargetHost>>(const std::string& fn, const core::Image2DView<float, core::TargetHost>& input);

template core::Image2DManaged<uint8_t, core::TargetHost> core::io::loadImage<core::Image2DManaged<uint8_t, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Image2DView<uint8_t, core::TargetHost>>(const std::string& fn, const core::Image2DView<uint8_t, core::TargetHost>& input);

template core::Image2DManaged<uint16_t, core::TargetHost> core::io::loadImage<core::Image2DManaged<uint16_t, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Image2DView<uint16_t, core::TargetHost>>(const std::string& fn, const core::Image2DView<uint16_t, core::TargetHost>& input);

template core::Image2DManaged<uchar3, core::TargetHost> core::io::loadImage<core::Image2DManaged<uchar3, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Image2DView<uchar3, core::TargetHost>>(const std::string& fn, const core::Image2DView<uchar3, core::TargetHost>& input);

template core::Buffer2DManaged<float, core::TargetHost> core::io::loadImage<core::Buffer2DManaged<float, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Buffer2DView<float, core::TargetHost>>(const std::string& fn, const core::Buffer2DView<float, core::TargetHost>& input);

template core::Buffer2DManaged<uint8_t, core::TargetHost> core::io::loadImage<core::Buffer2DManaged<uint8_t, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Buffer2DView<uint8_t, core::TargetHost>>(const std::string& fn, const core::Buffer2DView<uint8_t, core::TargetHost>& input);

template core::Buffer2DManaged<uint16_t, core::TargetHost> core::io::loadImage<core::Buffer2DManaged<uint16_t, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Buffer2DView<uint16_t, core::TargetHost>>(const std::string& fn, const core::Buffer2DView<uint16_t, core::TargetHost>& input);

template core::Buffer2DManaged<uchar3, core::TargetHost> core::io::loadImage<core::Buffer2DManaged<uchar3, core::TargetHost>>(const std::string& fn);
template void core::io::saveImage<core::Buffer2DView<uchar3, core::TargetHost>>(const std::string& fn, const core::Buffer2DView<uchar3, core::TargetHost>& input);

#ifdef CORE_HAVE_CAMERA_DRIVERS
template drivers::camera::FrameBuffer core::io::loadImage<drivers::camera::FrameBuffer>(const std::string& fn);
template void core::io::saveImage<drivers::camera::FrameBuffer>(const std::string& fn, const drivers::camera::FrameBuffer& input);
#endif // CORE_HAVE_CAMERA_DRIVERS
