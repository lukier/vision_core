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
 * OpenGL Buffers.
 * ****************************************************************************
 */
#ifndef VISIONCORE_WRAPGL_BUFFER_HPP
#define VISIONCORE_WRAPGL_BUFFER_HPP

#include <array>
#include <vector>

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

#include <VisionCore/Buffers/Buffer1D.hpp>
#include <VisionCore/Buffers/Buffer2D.hpp>
#include <VisionCore/Buffers/Buffer3D.hpp>

/**
 * TODO:
 * 
 * glTexBuffer / glTexBufferRange - texture as buffer?
 * 
 * 
 */

namespace vc
{

namespace wrapgl
{
       
class Buffer
{
public:
   
    Buffer() : 
        bufid(0), 
        buffer_type(GL_ARRAY_BUFFER), 
        num_elements(0), 
        datatype(GL_BYTE), 
        count_per_element(1), 
        bufuse(GL_DYNAMIC_DRAW)
    {
        
    }
    
    Buffer(GLenum bt, GLuint numel, GLenum dtype, GLuint cpe, GLenum gluse = GL_DYNAMIC_DRAW, const GLvoid* data = nullptr) : bufid(0)
    {
        create(bt, numel, dtype, cpe, gluse, data);
    }
    
    template<typename T, typename AllocatorT>
    Buffer(GLenum bt, const std::vector<T,AllocatorT>& vec, GLenum gluse = GL_DYNAMIC_DRAW) : bufid(0)
    {
        create(bt, vec, gluse);
    }
    
    template<typename T>
    Buffer(GLenum bt, GLuint numel, GLenum gluse = GL_DYNAMIC_DRAW, const T* data = nullptr) : bufid(0)
    {
        create<T>(bt, numel, gluse, data);
    }
    
    template<typename T>
    Buffer(GLenum bt, const Buffer1DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) : bufid(0)
    {
        create(bt, buf, gluse);
    }
    
    template<typename T>
    Buffer(GLenum bt, const Buffer2DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) : bufid(0)
    {
        create(bt, buf, gluse);
    }
    
    template<typename T>
    Buffer(GLenum bt, const Buffer3DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) : bufid(0)
    {
        create(bt, buf, gluse);
    }
    
    virtual ~Buffer()
    {
        destroy();
    }
    
    template<typename T, typename AllocatorT>
    void create(GLenum bt, const std::vector<T,AllocatorT>& vec, GLenum gluse = GL_DYNAMIC_DRAW)
    {
        create(bt, vec.size(), internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, static_cast<const GLvoid*>(vec.data()));
    }
    
    template<typename T>
    void create(GLenum bt, GLuint numel, GLenum gluse = GL_DYNAMIC_DRAW, const T* data = nullptr) 
    {
        create(bt, numel, internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, static_cast<const GLvoid*>(data));
    }
    
    template<typename T>
    void create(GLenum bt, const Buffer1DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) 
    {
        create(bt, buf.size(), internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, buf.ptr());
    }
    
    template<typename T>
    void create(GLenum bt, const Buffer2DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) 
    {
        create(bt, buf.width() * buf.height(), internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, buf.ptr());
    }
    
    template<typename T>
    void create(GLenum bt, const Buffer3DView<T,TargetHost>& buf, GLenum gluse = GL_DYNAMIC_DRAW) 
    {
        create(bt, buf.width() * buf.height() * buf.depth(), internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, buf.ptr());
    }
    
    void create(GLenum bt, GLuint numel, GLenum dtype, GLuint cpe, GLenum gluse = GL_DYNAMIC_DRAW, const GLvoid* data = nullptr)
    {
        if(isValid()) { destroy(); }
        
        buffer_type = bt;
        num_elements = numel;
        datatype = dtype;
        count_per_element = cpe;
        bufuse = gluse;
        
        glGenBuffers(1, &bufid);
        
        bind();
        glBufferData(buffer_type, num_elements*internal::getByteSize(datatype)*count_per_element, data, gluse);
        unbind();
    }
    
    void destroy()
    {
        if(bufid != 0) 
        {
            glDeleteBuffers(1, &bufid);
            bufid = 0;
        }
    }
    
    inline bool isValid() const { return bufid != 0; }
    
    void resize(GLuint new_num_elements, GLenum gluse = GL_DYNAMIC_DRAW)
    {
        bind();
        glBufferData(buffer_type, new_num_elements*internal::getByteSize(datatype)*count_per_element, 0, gluse);
        unbind();
        num_elements = new_num_elements;
    }
    
    void bind() const
    {
        glBindBuffer(buffer_type, bufid);
    }
    
    void unbind() const
    {
        glBindBuffer(buffer_type, 0);
    }
    
    void upload(const GLvoid* data, GLsizeiptr size_bytes, GLintptr offset = 0)
    {
        bind();
        glBufferSubData(buffer_type,offset,size_bytes, data);
        unbind();
    }
    
    template<typename T, typename AllocatorT>
    void upload(const std::vector<T,AllocatorT>& vec, GLintptr offset = 0)
    {
        upload(vec.data(), vec.size() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void upload(const Buffer1DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        upload(buf.ptr(), buf.size() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void upload(const Buffer2DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        upload(buf.ptr(), buf.width() * buf.height() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void upload(const Buffer3DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        upload(buf.ptr(), buf.width() * buf.height() * buf.depth() * sizeof(T), offset * sizeof(T));
    }
    
    void download(GLvoid* data, GLsizeiptr size_bytes, GLintptr offset = 0)
    {
        bind();
        glGetBufferSubData(buffer_type,offset,size_bytes, data);
        unbind();
    }
    
    template<typename T, typename AllocatorT>
    void download(std::vector<T,AllocatorT>& vec, GLintptr offset = 0)
    {
        download(vec.data(), vec.size() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void download(Buffer1DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        download(buf.ptr(), buf.size() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void download(Buffer2DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        download(buf.ptr(), buf.width() * buf.height() * sizeof(T), offset * sizeof(T));
    }
    
    template<typename T>
    void download(Buffer3DView<T,TargetHost>& buf, GLintptr offset = 0)
    {
        download(buf.ptr(), buf.width() * buf.height() * buf.depth() * sizeof(T), offset * sizeof(T));
    }
    
    inline GLuint id() const { return bufid; }
    inline GLenum type() const { return buffer_type; }
    inline GLuint size() const { return num_elements; }
    inline GLuint sizeBytes() const { return num_elements * count_per_element * internal::getByteSize(datatype); }
    inline GLenum dataType() const { return datatype; }
    inline GLuint countPerElement() const { return count_per_element; }
    
    struct MapPtr
    {
        MapPtr(Buffer& b, GLenum acc) : buf(b)
        {
            ptr = glMapBuffer(buf.type(), acc);
        }
        
        ~MapPtr()
        {
            glUnmapBuffer(buf.type());
        }
        
        inline GLvoid* operator*() { return ptr; }
        inline GLvoid* get() { return ptr; }
    private:
        Buffer& buf;
        GLvoid* ptr;
    };
    
    MapPtr map(GLenum acc = GL_READ_ONLY) { return MapPtr(*this, acc); }
private:
    GLuint bufid;
    GLenum buffer_type;
    GLuint num_elements;
    GLenum datatype;
    GLuint count_per_element;
    GLenum bufuse;
};
    
}

template<typename T, typename Target>
class Buffer1DFromOpenGL;
template<typename T, typename Target>
class Buffer2DFromOpenGL;
template<typename T, typename Target>
class Buffer3DFromOpenGL;

}

#ifdef VISIONCORE_HAVE_CUDA

#include <cuda_runtime_api.h>

#include <VisionCore/CUDAException.hpp>

namespace vc
{
 
// some functions wrapper here as cuda_gl_interop header leaks horrible stuff
namespace internal
{
    cudaGraphicsResource* registerOpenGLBuffer(GLuint id, unsigned int flags = cudaGraphicsMapFlagsNone);
}
    
template<typename T>
class Buffer1DFromOpenGL<T,TargetDeviceCUDA> : public Buffer1DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer1DView<T,TargetDeviceCUDA> ViewT;
    
    Buffer1DFromOpenGL() = delete;
    
    Buffer1DFromOpenGL(wrapgl::Buffer& glbuf) : cuda_res(0)
    {
        if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
        {
            throw std::runtime_error("Buffer format error");
        }
        
        cuda_res = internal::registerOpenGLBuffer(glbuf.id());
        
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error mapping OpenGL buffer"); }
        
        err = cudaGraphicsResourceGetMappedPointer(&(ViewT::memptr), &(ViewT::xsize), cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error getting OpenGL buffer ptr"); }
    }
    
    ~Buffer1DFromOpenGL()
    {
        if(cuda_res) 
        {   
            cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
            assert(err == cudaSuccess);
            
            err = cudaGraphicsUnregisterResource(cuda_res);
            assert(err == cudaSuccess);
        }
    }
    
    inline Buffer1DFromOpenGL(const Buffer1DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    
    inline Buffer1DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer1DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer1DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer1DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
    
private:
    cudaGraphicsResource* cuda_res;
};

template<typename T>
class Buffer2DFromOpenGL<T,TargetDeviceCUDA> : public Buffer2DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer2DView<T,TargetDeviceCUDA> ViewT;
    
    Buffer2DFromOpenGL() = delete;
    
    Buffer2DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height) : cuda_res(0)
    {
        if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
        {
            throw std::runtime_error("Buffer format error");
        }
        
        cuda_res = internal::registerOpenGLBuffer(glbuf.id());
        
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error mapping OpenGL buffer"); }
        
        std::size_t totalSize;
        err = cudaGraphicsResourceGetMappedPointer(&(ViewT::memptr), &(totalSize), cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error getting OpenGL buffer ptr"); }
        
        ViewT::xsize = totalSize / height;
        ViewT::ysize = height;
        ViewT::line_pitch = (ViewT::xsize * sizeof(T));
    }
    
    ~Buffer2DFromOpenGL()
    {
        if(cuda_res) 
        {   
            cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
            assert(err == cudaSuccess);
            
            err = cudaGraphicsUnregisterResource(cuda_res);
            assert(err == cudaSuccess);
        }
    }
    
    inline Buffer2DFromOpenGL(const Buffer2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    
    inline Buffer2DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer2DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer2DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
    
private:
    cudaGraphicsResource* cuda_res;
};

template<typename T>
class Buffer3DFromOpenGL<T,TargetDeviceCUDA> : public Buffer3DView<T,TargetDeviceCUDA>
{
public:
    typedef Buffer3DView<T,TargetDeviceCUDA> ViewT;
    
    Buffer3DFromOpenGL() = delete;
    
    Buffer3DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth) : cuda_res(0)
    {
        if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
        {
            throw std::runtime_error("Buffer format error");
        }
        
        cuda_res = internal::registerOpenGLBuffer(glbuf.id());
        
        cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error mapping OpenGL buffer"); }
        
        std::size_t totalSize;
        err = cudaGraphicsResourceGetMappedPointer(&(ViewT::memptr), &(totalSize), cuda_res);
        if(err != cudaSuccess) { throw CUDAException(err, "Error getting OpenGL buffer ptr"); }
        
        ViewT::xsize = totalSize / (height * depth);
        ViewT::ysize = height;
        ViewT::zsize = depth;
        ViewT::line_pitch = (ViewT::xsize * sizeof(T));
        ViewT::plane_pitch = (ViewT::ysize * ViewT::xsize * sizeof(T));
    }
    
    ~Buffer3DFromOpenGL()
    {
        if(cuda_res) 
        {   
            cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
            assert(err == cudaSuccess);
            
            err = cudaGraphicsUnregisterResource(cuda_res);
            assert(err == cudaSuccess);
        }
    }
    
    inline Buffer3DFromOpenGL(const Buffer3DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img) : ViewT(std::move(img)), cuda_res(img.cuda_res)
    {
        img.cuda_res = 0;
    }
    
    inline Buffer3DFromOpenGL<T,TargetDeviceCUDA>& operator=(const Buffer3DFromOpenGL<T,TargetDeviceCUDA>& img) = delete;
    
    inline Buffer3DFromOpenGL<T,TargetDeviceCUDA>& operator=(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img)
    {
        ViewT::operator=(std::move(img));
        cuda_res = img.cuda_res;
        img.cuda_res = 0;
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
    
private:
    cudaGraphicsResource* cuda_res;
};
    
}

#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL

namespace vc
{
    
template<typename T>
class Buffer1DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer1DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer1DView<T,TargetDeviceOpenCL> ViewT;
    
    Buffer1DFromOpenGL() = delete;
    
    Buffer1DFromOpenGL(const cl::Context& context, cl_mem_flags flags, wrapgl::Buffer& glbuf) 
    {
        ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
        ViewT::xsize = glbuf.size();
    }
    
    ~Buffer1DFromOpenGL()
    {
        if(ViewT::isValid()) 
        {   
            cl::BufferGL* clb = static_cast<cl::BufferGL*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
        }
    }
    
    inline Buffer1DFromOpenGL(const Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    
    inline Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer1DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer1DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

template<typename T>
class Buffer2DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer2DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer2DView<T,TargetDeviceOpenCL> ViewT;
    
    Buffer2DFromOpenGL() = delete;
    
    Buffer2DFromOpenGL(const cl::Context& context, cl_mem_flags flags, wrapgl::Buffer& glbuf, std::size_t height) 
    {
        ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
        ViewT::xsize = glbuf.size() / height;
        ViewT::ysize = height;
        ViewT::line_pitch = (ViewT::xsize * sizeof(T));
    }
    
    ~Buffer2DFromOpenGL()
    {
        if(ViewT::isValid()) 
        {   
            cl::BufferGL* clb = static_cast<cl::BufferGL*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::line_pitch = 0;
        }
    }
    
    inline Buffer2DFromOpenGL(const Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    
    inline Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer2DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer2DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

template<typename T>
class Buffer3DFromOpenGL<T,TargetDeviceOpenCL> : public Buffer3DView<T,TargetDeviceOpenCL>
{
public:
    typedef Buffer3DView<T,TargetDeviceOpenCL> ViewT;
    
    Buffer3DFromOpenGL() = delete;
    
    Buffer3DFromOpenGL(const cl::Context& context, cl_mem_flags flags, wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth) 
    {
        ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
        ViewT::xsize = glbuf.size() / (height * depth);
        ViewT::ysize = height;
        ViewT::zsize = depth;
        ViewT::line_pitch = (ViewT::xsize * sizeof(T));
        ViewT::plane_pitch = (ViewT::ysize * ViewT::xsize * sizeof(T));
    }
    
    ~Buffer3DFromOpenGL()
    {
        if(ViewT::isValid()) 
        {   
            cl::BufferGL* clb = static_cast<cl::BufferGL*>(ViewT::memptr);
            delete clb;
            ViewT::memptr = nullptr;
            ViewT::xsize = 0;
            ViewT::ysize = 0;
            ViewT::zsize = 0;
            ViewT::line_pitch = 0;
            ViewT::plane_pitch = 0;
        }
    }
    
    inline Buffer3DFromOpenGL(const Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetDeviceOpenCL>&& img) : ViewT(std::move(img)) { }
    
    inline Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& operator=(const Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& img) = delete;
    
    inline Buffer3DFromOpenGL<T,TargetDeviceOpenCL>& operator=(Buffer3DFromOpenGL<T,TargetDeviceOpenCL>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};  
    
}

#endif // VISIONCORE_HAVE_OPENCL

#endif // VISIONCORE_WRAPGL_BUFFER_HPP
