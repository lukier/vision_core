/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
#ifndef VISIONCORE_WRAPGL_BUFFER_IMPL_HPP
#define VISIONCORE_WRAPGL_BUFFER_IMPL_HPP

vc::wrapgl::Buffer::Buffer() : 
    bufid(0), 
    buffer_type(GL_ARRAY_BUFFER), 
    num_elements(0), 
    datatype(GL_BYTE), 
    count_per_element(1), 
    bufuse(GL_DYNAMIC_DRAW)
{
    
}

vc::wrapgl::Buffer::Buffer(GLenum bt, GLuint numel, GLenum dtype, GLuint cpe, 
                           GLenum gluse, const GLvoid* data) : bufid(0)
{
    create(bt, numel, dtype, cpe, gluse, data);
}

template<typename T, typename AllocatorT>
vc::wrapgl::Buffer::Buffer(GLenum bt, const std::vector<T,AllocatorT>& vec, GLenum gluse) : bufid(0)
{
    create(bt, vec, gluse);
}

template<typename T>
vc::wrapgl::Buffer::Buffer(GLenum bt, GLuint numel, GLenum gluse, const T* data) : bufid(0)
{
    create<T>(bt, numel, gluse, data);
}

template<typename T>
vc::wrapgl::Buffer::Buffer(GLenum bt, const Buffer1DView<T,TargetHost>& buf, GLenum gluse) : bufid(0)
{
    create(bt, buf, gluse);
}

template<typename T>
vc::wrapgl::Buffer::Buffer(GLenum bt, const Buffer2DView<T,TargetHost>& buf, GLenum gluse) : bufid(0)
{
    create(bt, buf, gluse);
}

template<typename T>
vc::wrapgl::Buffer::Buffer(GLenum bt, const Buffer3DView<T,TargetHost>& buf, GLenum gluse) : bufid(0)
{
    create(bt, buf, gluse);
}

template<typename T, typename AllocatorT>
void vc::wrapgl::Buffer::create(GLenum bt, const std::vector<T,AllocatorT>& vec, GLenum gluse)
{
    create(bt, vec.size(), 
           internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, 
           type_traits<T>::ChannelCount, gluse, static_cast<const GLvoid*>(vec.data()));
}

template<typename T>
void vc::wrapgl::Buffer::create(GLenum bt, GLuint numel, GLenum gluse, const T* data) 
{
    create(bt, numel, internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, 
           type_traits<T>::ChannelCount, gluse, static_cast<const GLvoid*>(data));
}

template<typename T>
void vc::wrapgl::Buffer::create(GLenum bt, const Buffer1DView<T,TargetHost>& buf, GLenum gluse) 
{
    create(bt, buf.size(), 
           internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, 
           type_traits<T>::ChannelCount, gluse, buf.ptr());
}

template<typename T>
void vc::wrapgl::Buffer::create(GLenum bt, const Buffer2DView<T,TargetHost>& buf, GLenum gluse) 
{
    create(bt, buf.width() * buf.height(), 
           internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, 
           type_traits<T>::ChannelCount, gluse, buf.ptr());
}

template<typename T>
void vc::wrapgl::Buffer::create(GLenum bt, const Buffer3DView<T,TargetHost>& buf, GLenum gluse) 
{
    create(bt, buf.width() * buf.height() * buf.depth(), internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type, type_traits<T>::ChannelCount, gluse, buf.ptr());
}

void vc::wrapgl::Buffer::create(GLenum bt, GLuint numel, GLenum dtype, GLuint cpe, GLenum gluse, const GLvoid* data)
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

void vc::wrapgl::Buffer::destroy()
{
    if(bufid != 0) 
    {
        glDeleteBuffers(1, &bufid);
        bufid = 0;
    }
}

bool vc::wrapgl::Buffer::isValid() const 
{ 
    return bufid != 0; 
}

void vc::wrapgl::Buffer::resize(GLuint new_num_elements, GLenum gluse)
{
    bind();
    glBufferData(buffer_type, new_num_elements*internal::getByteSize(datatype)*count_per_element, 0, gluse);
    unbind();
    num_elements = new_num_elements;
}

void vc::wrapgl::Buffer::bind() const
{
    glBindBuffer(buffer_type, bufid);
}

void vc::wrapgl::Buffer::unbind() const
{
    glBindBuffer(buffer_type, 0);
}

void vc::wrapgl::Buffer::upload(const GLvoid* data, GLsizeiptr size_bytes, GLintptr offset)
{
    bind();
    glBufferSubData(buffer_type,offset,size_bytes, data);
    unbind();
}

template<typename T, typename AllocatorT>
void vc::wrapgl::Buffer::upload(const std::vector<T,AllocatorT>& vec, GLintptr offset)
{
    upload(vec.data(), vec.size() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::upload(const Buffer1DView<T,TargetHost>& buf, GLintptr offset)
{
    upload(buf.ptr(), buf.size() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::upload(const Buffer2DView<T,TargetHost>& buf, GLintptr offset)
{
    upload(buf.ptr(), buf.width() * buf.height() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::upload(const Buffer3DView<T,TargetHost>& buf, GLintptr offset)
{
    upload(buf.ptr(), buf.width() * buf.height() * buf.depth() * sizeof(T), offset * sizeof(T));
}

void vc::wrapgl::Buffer::download(GLvoid* data, GLsizeiptr size_bytes, GLintptr offset)
{
    bind();
    glGetBufferSubData(buffer_type,offset,size_bytes, data);
    unbind();
}

template<typename T, typename AllocatorT>
void vc::wrapgl::Buffer::download(std::vector<T,AllocatorT>& vec, GLintptr offset)
{
    download(vec.data(), vec.size() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::download(Buffer1DView<T,TargetHost>& buf, GLintptr offset)
{
    download(buf.ptr(), buf.size() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::download(Buffer2DView<T,TargetHost>& buf, GLintptr offset)
{
    download(buf.ptr(), buf.width() * buf.height() * sizeof(T), offset * sizeof(T));
}

template<typename T>
void vc::wrapgl::Buffer::download(Buffer3DView<T,TargetHost>& buf, GLintptr offset)
{
    download(buf.ptr(), buf.width() * buf.height() * buf.depth() * sizeof(T), offset * sizeof(T));
}

GLuint vc::wrapgl::Buffer::id() const 
{ 
    return bufid; 
}

GLenum vc::wrapgl::Buffer::type() const 
{ 
    return buffer_type; 
}

GLuint vc::wrapgl::Buffer::size() const 
{   
    return num_elements; 
}

GLuint vc::wrapgl::Buffer::sizeBytes() const 
{ 
    return num_elements * count_per_element * internal::getByteSize(datatype); 
}

GLenum vc::wrapgl::Buffer::dataType() const 
{ 
    return datatype; 
}

GLuint vc::wrapgl::Buffer::countPerElement() const 
{ 
    return count_per_element; 
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::Buffer1DFromOpenGL(wrapgl::Buffer& glbuf, GLenum acc) : buf(glbuf)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
      || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
    {
        throw std::runtime_error("Buffer format error");
    }
  
    ViewT::xsize = buf.size();
    ViewT::memptr = glMapNamedBuffer(buf.id(), acc);
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::~Buffer1DFromOpenGL()
{
    if(ViewT::ptr != nullptr) 
    {   
        glUnmapNamedBuffer(buf.id());
    }
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,vc::TargetHost>&& img) 
    : ViewT(std::move(img)), buf(std::move(img.buf))
{
  
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetHost>& 
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::operator=(Buffer1DFromOpenGL<T,TargetHost>&& img)
{
    ViewT::operator=(std::move(img));
    buf = std::move(img.buf);
    return *this;
}

template<typename T>
const typename vc::Buffer1DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer1DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetHost>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::Buffer2DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, GLenum acc) : buf(glbuf)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
      || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
    {
        throw std::runtime_error("Buffer format error");
    }
    
    ViewT::xsize = buf.size() / height;
    ViewT::ysize = height;
    ViewT::line_pitch = (ViewT::xsize * sizeof(T));
    ViewT::memptr = glMapNamedBuffer(buf.id(), acc);
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::~Buffer2DFromOpenGL()
{
    if(ViewT::ptr != nullptr) 
    {   
        glUnmapNamedBuffer(buf.id());
    }
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetHost>&& img)
    : ViewT(std::move(img)), buf(std::move(img.buf))
{
  
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetHost>& 
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::operator=(Buffer2DFromOpenGL<T,TargetHost>&& img)
{
  ViewT::operator=(std::move(img));
  buf = std::move(img.buf);
  
  return *this;
}

template<typename T>
const typename vc::Buffer2DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer2DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetHost>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::Buffer3DFromOpenGL(wrapgl::Buffer& glbuf, 
                                                             std::size_t height, std::size_t depth, GLenum acc) : buf(glbuf)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
    || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
    {
        throw std::runtime_error("Buffer format error");
    }

    ViewT::xsize = buf.size() / (height * depth);
    ViewT::ysize = height;
    ViewT::zsize = depth;
    ViewT::line_pitch = (ViewT::xsize * sizeof(T));
    ViewT::plane_pitch = (ViewT::ysize * ViewT::xsize * sizeof(T));
    ViewT::memptr = glMapNamedBuffer(buf.id(), acc);
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::~Buffer3DFromOpenGL()
{
    if(ViewT::ptr != nullptr) 
    {   
        glUnmapNamedBuffer(buf.id());
    }
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetHost>&& img)
    : ViewT(std::move(img)), buf(std::move(img.buf))
{

}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetHost>& 
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::operator=(Buffer3DFromOpenGL<T,TargetHost>&& img)
{
    ViewT::operator=(std::move(img));
    buf = std::move(img.buf);
    return *this;
}

template<typename T>
const typename vc::Buffer3DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer3DFromOpenGL<T,vc::TargetHost>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetHost>::view() 
{ 
    return (ViewT&)*this; 
}

#ifdef VISIONCORE_HAVE_CUDA
template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer1DFromOpenGL(wrapgl::Buffer& glbuf) : cuda_res(0)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
      || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
    {
        throw std::runtime_error("Buffer format error");
    }
    
    cuda_res = internal::registerOpenGLBuffer(glbuf.id());
    
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_res);
    if(err != cudaSuccess) { throw CUDAException(err, "Error mapping OpenGL buffer"); }
    
    err = cudaGraphicsResourceGetMappedPointer(&(ViewT::memptr), &(ViewT::xsize), cuda_res);
    if(err != cudaSuccess) { throw CUDAException(err, "Error getting OpenGL buffer ptr"); }
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::~Buffer1DFromOpenGL()
{
    if(cuda_res) 
    {   
        cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
        assert(err == cudaSuccess);
        
        err = cudaGraphicsUnregisterResource(cuda_res);
        assert(err == cudaSuccess);
    }
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>&& img) 
  : ViewT(std::move(img)), cuda_res(img.cuda_res)
{
    img.cuda_res = 0;
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>& vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::operator=(Buffer1DFromOpenGL<T,TargetDeviceCUDA>&& img)
{
    ViewT::operator=(std::move(img));
    cuda_res = img.cuda_res;
    img.cuda_res = 0;
    return *this;
}

template<typename T>
const typename vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceCUDA>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer2DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height) : cuda_res(0)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
      || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
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

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::~Buffer2DFromOpenGL()
{
    if(cuda_res) 
    {   
        cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
        assert(err == cudaSuccess);
        
        err = cudaGraphicsUnregisterResource(cuda_res);
        assert(err == cudaSuccess);
    }
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img)
    : ViewT(std::move(img)), cuda_res(img.cuda_res)
{
    img.cuda_res = 0;
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>& vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::operator=(Buffer2DFromOpenGL<T,TargetDeviceCUDA>&& img)
{
    ViewT::operator=(std::move(img));
    cuda_res = img.cuda_res;
    img.cuda_res = 0;
    return *this;
}

template<typename T>
const typename vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceCUDA>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer3DFromOpenGL(wrapgl::Buffer& glbuf, std::size_t height, 
                                                                   std::size_t depth) : cuda_res(0)
{
    if((wrapgl::internal::GLTypeTraits<typename type_traits<T>::ChannelType>::opengl_data_type != glbuf.dataType()) 
      || (type_traits<T>::ChannelCount != glbuf.countPerElement()))
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

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::~Buffer3DFromOpenGL()
{
    if(cuda_res) 
    {   
        cudaError_t err = cudaGraphicsUnmapResources(1, &cuda_res);
        assert(err == cudaSuccess);
        
        err = cudaGraphicsUnregisterResource(cuda_res);
        assert(err == cudaSuccess);
    }
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img)
    : ViewT(std::move(img)), cuda_res(img.cuda_res)
{
    img.cuda_res = 0;
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>& vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::operator=(Buffer3DFromOpenGL<T,TargetDeviceCUDA>&& img)
{
    ViewT::operator=(std::move(img));
    cuda_res = img.cuda_res;
    img.cuda_res = 0;
    return *this;
}

template<typename T>
const typename vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceCUDA>::view() 
{ 
    return (ViewT&)*this; 
}
#endif // VISIONCORE_HAVE_CUDA

#ifdef VISIONCORE_HAVE_OPENCL
template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer1DFromOpenGL(const cl::Context& context, cl_mem_flags flags, 
                                                                     wrapgl::Buffer& glbuf) 
{
    ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
    ViewT::xsize = glbuf.size();
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::~Buffer1DFromOpenGL()
{
    if(ViewT::isValid()) 
    {   
        cl::BufferGL* clb = static_cast<cl::BufferGL*>(ViewT::memptr);
        delete clb;
        ViewT::memptr = nullptr;
        ViewT::xsize = 0;
    }
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer1DFromOpenGL(Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
    : ViewT(std::move(img)) 
{
    
}

template<typename T>
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>& vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::operator=(Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
{
    ViewT::operator=(std::move(img));
    return *this;
}

template<typename T>
const typename vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer1DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer2DFromOpenGL(const cl::Context& context, cl_mem_flags flags, 
                                                                     wrapgl::Buffer& glbuf, std::size_t height) 
{
    ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
    ViewT::xsize = glbuf.size() / height;
    ViewT::ysize = height;
    ViewT::line_pitch = (ViewT::xsize * sizeof(T));
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::~Buffer2DFromOpenGL()
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

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer2DFromOpenGL(Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
    : ViewT(std::move(img)) 
{ 
  
}

template<typename T>
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>& vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::operator=(Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
{
    ViewT::operator=(std::move(img));
    return *this;
}

template<typename T>
const typename vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() const 
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer2DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() 
{ 
    return (ViewT&)*this; 
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer3DFromOpenGL(const cl::Context& context, cl_mem_flags flags, 
                                                                     wrapgl::Buffer& glbuf, std::size_t height, std::size_t depth) 
{
    ViewT::memptr = new cl::BufferGL(context, flags, glbuf.id());
    ViewT::xsize = glbuf.size() / (height * depth);
    ViewT::ysize = height;
    ViewT::zsize = depth;
    ViewT::line_pitch = (ViewT::xsize * sizeof(T));
    ViewT::plane_pitch = (ViewT::ysize * ViewT::xsize * sizeof(T));
}

template<typename T>    
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::~Buffer3DFromOpenGL()
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
    
template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::Buffer3DFromOpenGL(Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
    : ViewT(std::move(img)) 
{ 
  
}

template<typename T>
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>& vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::operator=(Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>&& img)
{
    ViewT::operator=(std::move(img));
    return *this;
}
    
template<typename T>
const typename vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() const
{ 
    return (const ViewT&)*this; 
}

template<typename T>
typename vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::ViewT& 
vc::Buffer3DFromOpenGL<T,vc::TargetDeviceOpenCL>::view() 
{ 
    return (ViewT&)*this; 
}
#endif // VISIONCORE_HAVE_OPENCL

#endif // VISIONCORE_WRAPGL_BUFFER_IMPL_HPP
