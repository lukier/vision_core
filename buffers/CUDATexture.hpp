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
 * CUDA Texture Access Wrapper.
 * ****************************************************************************
 */

#ifndef CORE_GPU_TEXTURE_HPP
#define CORE_GPU_TEXTURE_HPP

#include <Platform.hpp>

#include <image/PixelConvert.hpp>

#include <buffers/Buffer2D.hpp>
#include <buffers/Image2D.hpp>

namespace core
{
    
namespace internal
{

template<typename T, template<typename = T> class Target>
struct TextureFetchHelper
{
    EIGEN_DEVICE_FUNC static inline T get(typename Target<T>::TextureHandleType tex, float x, float y)
    {
        return core::zero<T>();
    }
};

}
    
}

#ifdef CORE_HAVE_CUDA

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<Eigen::Vector2f>(void)
{
    int e = (int)sizeof(float) * 8;
    
    return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat);
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<Eigen::Vector4f>(void)
{
    int e = (int)sizeof(float) * 8;
    
    return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat);
}

namespace core
{

namespace internal
{

template<typename T>
struct TextureFetchHelper<T,TargetDeviceCUDA>
{
    EIGEN_DEVICE_FUNC static inline T get(typename TargetDeviceCUDA<T>::TextureHandleType tex, float x, float y)
    {
#ifdef CORE_CUDA_KERNEL_SPACE    
        return ::tex2D<T>(tex,x,y);
#endif // CORE_CUDA_KERNEL_SPACE  
    }
};

#ifdef CORE_CUDA_KERNEL_SPACE    

template<> struct TextureFetchHelper<Eigen::Vector2f,TargetDeviceCUDA>
{
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f get(typename TargetDeviceCUDA<Eigen::Vector2f>::TextureHandleType tex, float x, float y)
    {
        float4 tmp;
        // TODO FIXME __tex_2d_v4f32_f32(tex, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
        return Eigen::Vector2f(tmp.x,tmp.y);
    }
};

template<> struct TextureFetchHelper<Eigen::Vector4f,TargetDeviceCUDA>
{
    EIGEN_DEVICE_FUNC static inline Eigen::Vector4f get(typename TargetDeviceCUDA<Eigen::Vector4f>::TextureHandleType tex, float x, float y)
    {
        float4 tmp;
        // TODO FIXME __tex_2d_v4f32_f32(tex, x, y, &tmp.x, &tmp.y, &tmp.z, &tmp.w);
        return Eigen::Vector4f(tmp.x,tmp.y,tmp.z,tmp.w);
    }
};

#endif // CORE_CUDA_KERNEL_SPACE 

}

}

#endif // CORE_HAVE_CUDA

namespace core
{

template<typename T, template<typename = T> class Target = TargetHost>
class GPUTexture2DView 
{
public:
    typedef typename core::type_traits<T>::ChannelType ValueType;
    static const int Channels = core::type_traits<T>::ChannelCount;
    
    EIGEN_DEVICE_FUNC inline GPUTexture2DView() : tex(0) { }
    
    EIGEN_DEVICE_FUNC inline ~GPUTexture2DView()
    {
        
    }

    EIGEN_DEVICE_FUNC inline GPUTexture2DView(const GPUTexture2DView<T,Target>& img)  : tex(img.tex)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline GPUTexture2DView(GPUTexture2DView<T,Target>&& img) : tex(img.tex)
    {
        img.tex = 0;   
    }
    
    EIGEN_DEVICE_FUNC inline GPUTexture2DView<T,Target>& operator=(const GPUTexture2DView<T,Target>& img)
    {
        tex = img.tex;
        return *this;
    }
        
    EIGEN_DEVICE_FUNC inline GPUTexture2DView<T,Target>& operator=(GPUTexture2DView<T,Target>&& img)
    {
        tex = img.tex;
        img.tex = 0;
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline bool isValid() const
    {
        return tex != 0;
    }
    
    EIGEN_DEVICE_FUNC inline T operator()(float x, float y) const
    {
        return internal::TextureFetchHelper<T,Target>::get(tex,x,y);
    }
    
    EIGEN_DEVICE_FUNC inline typename Target<T>::TextureHandleType& handle() { return tex; }
    EIGEN_DEVICE_FUNC inline const typename Target<T>::TextureHandleType& handle() const { return tex; }
    
protected:
    typename Target<T>::TextureHandleType tex;
};

#ifdef CORE_HAVE_CUDA

template<typename T, template<typename = T> class Target = TargetDeviceCUDA>
class GPUTexture2DFromBuffer2D : public GPUTexture2DView<T,Target>
{
public:
    typedef GPUTexture2DView<T,Target> ViewT;
    
    GPUTexture2DFromBuffer2D() = delete;
    
    inline GPUTexture2DFromBuffer2D(const Buffer2DView<T,Target>& img) : ViewT()
    {
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = img.ptr();
        resDesc.res.pitch2D.pitchInBytes =  img.pitch();
        resDesc.res.pitch2D.width = img.width();
        resDesc.res.pitch2D.height = img.height();
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.normalizedCoords = 0;
        
        cudaError_t err = cudaCreateTextureObject(&(ViewT::handle()), &resDesc, &texDesc, NULL);
        if(err != cudaSuccess)
        {
            throw CUDAException(err, "Cannot create texture object");
        }
    }
    
    inline ~GPUTexture2DFromBuffer2D()
    {
        cudaDestroyTextureObject(ViewT::tex);
    }
    
    inline GPUTexture2DFromBuffer2D(const GPUTexture2DFromBuffer2D<T,Target>& img) = delete;
    
    inline GPUTexture2DFromBuffer2D(GPUTexture2DFromBuffer2D<T,Target>&& img) : ViewT(std::move(img)), resDesc(img.resDesc), texDesc(img.texDesc)
    {
        img.texref = 0;
    }
    
    inline GPUTexture2DFromBuffer2D<T,Target>& operator=(const GPUTexture2DFromBuffer2D<T,Target>& img) = delete;
    
    inline GPUTexture2DFromBuffer2D<T,Target>& operator=(GPUTexture2DFromBuffer2D<T,Target>&& img)
    {
        ViewT::operator=(std::move(img));
        resDesc = img.resDesc;
        texDesc = img.texDesc;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }

private:
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
};

#endif // CORE_HAVE_CUDA
    
}

#endif // CORE_IMAGE_HPP
