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
 * Texture.
 * ****************************************************************************
 */

#ifndef WRAPGL_TEXTURE_HPP
#define WRAPGL_TEXTURE_HPP

#include <wrapgl/WrapGLCommon.hpp>

#include <buffers/Buffer1D.hpp>
#include <buffers/Buffer2D.hpp>
#include <buffers/Buffer3D.hpp>
#include <buffers/Image2D.hpp>

/**
 * TODO:
 * 
 * This is only 2D. There can be also: glTexImage1D / glTexImage3D / GL_TEXTURE_CUBE_MAP etc
 * 
 * Add Pyramid-MultiLevel support.
 * 
 * More TexParameters (get/set)
 * 
 * Texture with Buffer data - GL_TEXTURE_BUFFER
 * 
 * glTexStorage2D  glTextureStorage2D ??
 * 
 * glTextureView — initialize a texture as a data alias of another texture's data store
 * 
 * glGenerateTextureMipmap
 * 
 * Image: glBindImageTexture​.
 */

namespace wrapgl
{
    
class TextureBase
{
public:
    TextureBase() : texid(0), internal_format((GLenum)0)
    {
        
    }
    
    virtual ~TextureBase()
    {
        destroy();
    }
    
    void create(GLenum int_format = GL_RGBA)
    {
        if(isValid()) { destroy(); }
        
        internal_format = int_format;
        glGenTextures(1,&texid);
    }
    
    void destroy()
    {
        if(isValid()) 
        {
            glDeleteTextures(1,&texid);
            internal_format = (GLenum)0;
            texid = 0;
        }
    }
    
    inline GLuint id() const { return texid; }
    inline GLenum intFormat() const { return internal_format; }
    
    inline bool isValid() const { return texid != 0; }
    
    static void bind(const GLenum unit)
    {
        glActiveTexture(unit);
    }
    
protected:
    GLuint texid;
    GLenum internal_format;
};

class Texture2DBase : public TextureBase
{
public:
    Texture2DBase() : TextureBase(), texw(0), texh(0)
    {
        
    }
    
    template<typename T>
    void upload(const core::Buffer2DView<T,core::TargetHost>& buf, GLenum data_format = GL_LUMINANCE)
    {
        upload(buf.ptr(), data_format, internal::GLTypeTraits<typename core::type_traits<T>::ChannelType>::opengl_data_type);
    }
    
    void upload(const GLvoid* data, GLenum data_format = GL_LUMINANCE, GLenum data_type = GL_FLOAT)
    {
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,texw,texh,data_format,data_type,data);
    }
    
    template<typename T>
    void download(core::Buffer2DView<T,core::TargetHost>& buf, GLenum data_format = GL_LUMINANCE)
    {
        download(buf.ptr(), data_format, internal::GLTypeTraits<typename core::type_traits<T>::ChannelType>::opengl_data_type);
    }
    
    void download(GLvoid* data, GLenum data_format = GL_LUMINANCE, GLenum data_type = GL_FLOAT)
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glGetTexImage(GL_TEXTURE_2D, 0, data_format, data_type, data);
    }
    
    inline void setSamplingLinear()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_LINEAR);
    }
    
    inline void setSamplingNearestNeighbour()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_NEAREST);
    }
    
    inline void setWrapClamp()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP);
    }
    
    inline void setWrapClampToEdge()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP_TO_EDGE);
    }
    
    inline void setWrapClampToBorder()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_CLAMP_TO_BORDER);
    }
    
    inline void setWrapRepeat()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_REPEAT);
    }
    
    inline void setWrapMirroredRepeat()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_MIRRORED_REPEAT);
    }
    
    inline void setDepthParameters()
    {
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, (GLint)GL_INTENSITY);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, (GLint)GL_COMPARE_R_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, (GLint)GL_LEQUAL);
    }
    
    inline void setBorderColor(float3 color)
    {
        setBorderColor(color.x, color.y, color.z, 1.0f);
    }
    
    inline void setBorderColor(float4 color)
    {
        setBorderColor(color.x, color.y, color.z, color.w);
    }
    
    inline void setBorderColor(const Eigen::Matrix<float,3,1>& color)
    {
        setBorderColor(color(0), color(1), color(2), 1.0f);
    }
    
    inline void setBorderColor(const Eigen::Matrix<float,4,1>& color)
    {
        setBorderColor(color(0), color(1), color(2), color(3));
    }
    
    inline void setBorderColor(float r, float g, float b, float a)
    {
        GLfloat params[4] = {r,g,b,a};
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, params);
    }
    
    void bind() const
    {
        glBindTexture(GL_TEXTURE_2D, texid);
    }
    
    void unbind() const
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    inline GLint width() const { return texw; }
    inline GLint height() const { return texh; }
    
protected:
    GLint texw;
    GLint texh;
};
    
class Texture2D : public Texture2DBase
{
public:
    Texture2D() : Texture2DBase()
    {
        
    }
    
    Texture2D(GLint w, GLint h, GLenum int_format = GL_RGBA32F, GLvoid* data = nullptr, int border = 0) : Texture2DBase()
    {
        create(w, h, int_format, data, border);
    }
    
    template<typename T>
    Texture2D(const core::Buffer2DView<T,core::TargetHost>& buf, GLenum int_format = GL_RGBA32F, int border = 0) : Texture2DBase()
    {
        create(buf, int_format, border);
    }
    
    virtual ~Texture2D()
    {
        destroy();
    }
    
    template<typename T>
    void create(const core::Buffer2DView<T,core::TargetHost>& buf, GLenum int_format = GL_RGBA32F, int border = 0) 
    {
        create(buf.width() * buf.height(), int_format, buf.ptr(), border);
    }
    
    void create(GLint w, GLint h, GLenum int_format = GL_RGBA32F,  GLvoid* data = nullptr, int border = 0)
    {
        if(isValid()) { destroy(); }
        
        TextureBase::create(int_format);
        
        texw = w;
        texh = h;

        bind();
        
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (GLint)GL_REPEAT);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, (GLint)GL_REPEAT);
        
        glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, texw, texh );
        
        unbind();
    }
    
    void destroy()
    {
        TextureBase::destroy();
    }
 
};

}

namespace core
{    

template<typename T, typename Target = TargetDeviceCUDA>
class GPUTexture2DFromOpenGL { };

}

#ifdef CORE_HAVE_CUDA

#include <CUDAException.hpp>

#include <buffers/CUDATexture.hpp>

namespace core
{
    
// some functions wrapper here as cuda_gl_interop header leaks horrible stuff
namespace internal
{
    cudaGraphicsResource* registerOpenGLTexture(GLenum textype, GLuint id, unsigned int flags = cudaGraphicsMapFlagsNone);
}

template<typename T>
class GPUTexture2DFromOpenGL<T,TargetDeviceCUDA> : public GPUTexture2DView<T,TargetDeviceCUDA>
{
public:
    typedef GPUTexture2DView<T,TargetDeviceCUDA> ViewT;
    
    GPUTexture2DFromOpenGL() = delete;
    
    inline GPUTexture2DFromOpenGL(const wrapgl::Texture2D& gltex) : ViewT(), cuda_res(0)
    {
        cuda_res = internal::registerOpenGLTexture(GL_TEXTURE_2D, gltex.id());
        
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
        if(err != cudaSuccess) { throw core::CUDAException(err, "Error mapping OpenGL texture"); }

        err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res, 0, 0);
        if(err != cudaSuccess) { throw core::CUDAException(err, "Error getting cudaArray from OpenGL texture"); }
        
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;
        
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.normalizedCoords = 0;
        
        err = cudaCreateTextureObject(&(ViewT::handle()), &resDesc, &texDesc, NULL);
        if(err != cudaSuccess)
        {
            throw CUDAException(err, "Cannot create texture object");
        }
    }
    
    inline ~GPUTexture2DFromOpenGL()
    {
        if(cuda_res) 
        {
            cudaError_t err;
            
            err = cudaDestroyTextureObject(ViewT::handle());
            assert(err == cudaSuccess);
            
            err = cudaGraphicsUnmapResources(1, &cuda_res);
            assert(err == cudaSuccess);

            err = cudaGraphicsUnregisterResource(cuda_res);
            assert(err == cudaSuccess);
        }
    }
    
    inline GPUTexture2DFromOpenGL(const GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline GPUTexture2DFromOpenGL(GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img)), resDesc(img.resDesc), texDesc(img.texDesc)
    {
        img.texref = 0;
    }
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& operator=(const GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>& operator=(GPUTexture2DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        resDesc = img.resDesc;
        texDesc = img.texDesc;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
    
private:
    cudaGraphicsResource* cuda_res;
    cudaArray* array;
    cudaResourceDesc resDesc;
    cudaTextureDesc texDesc;
};

}

#endif // CORE_HAVE_CUDA

#ifdef CORE_HAVE_OPENCL

namespace core
{
    
template<typename T>
class GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL> : public Image2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Image2DView<T,TargetDeviceOpenCL> ViewT;
    
    GPUTexture2DFromOpenGL() = delete;
    
    inline GPUTexture2DFromOpenGL(const wrapgl::Texture2D& gltex) : ViewT()
    {
        
    }
    
    inline ~GPUTexture2DFromOpenGL()
    {
        if(ViewT::isValid()) 
        {
            
        }
    }
    
    inline GPUTexture2DFromOpenGL(const GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline GPUTexture2DFromOpenGL(GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(GPUTexture2DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};
    
}

#endif // CORE_HAVE_OPENCL

#endif // WRAPGL_TEXTURE_HPP
