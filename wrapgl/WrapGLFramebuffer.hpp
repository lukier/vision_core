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
 * Frame/Render buffer objects.
 * ****************************************************************************
 */

#ifndef WRAPGL_FRAME_BUFFER_HPP
#define WRAPGL_FRAME_BUFFER_HPP

#include <array>

#include <wrapgl/WrapGLCommon.hpp>
#include <wrapgl/WrapGLTexture.hpp>

namespace wrapgl
{
    
class RenderBuffer
{
public:    
    RenderBuffer() : rbid(0), rbw(0), rbh(0)
    {
        
    }
    
    RenderBuffer(GLint w, GLint h, GLenum internal_format = GL_DEPTH_COMPONENT24) : rbid(0)
    {
        create(w,h,internal_format);
    }
    
    virtual ~RenderBuffer()
    {
        destroy();
    }
    
    void create(GLint w, GLint h, GLenum internal_format = GL_DEPTH_COMPONENT24)
    {
        destroy();
        
        rbw = w;
        rbh = h;
        
        glGenRenderbuffers(1, &rbid);
        bind();
        glRenderbufferStorage(GL_RENDERBUFFER, internal_format, rbw, rbh);
        unbind();
    }
    
    void destroy()
    {
        if(rbid != 0)
        {
            glDeleteRenderbuffers(1, &rbid);
            rbid = 0;
        }
    }
    
    inline bool isValid() const { return rbid != 0; }
    
    void bind() const
    {
        glBindRenderbuffer(GL_RENDERBUFFER, rbid);
    }
    
    void unbind() const
    {
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    
    template<typename T>
    void download(core::Buffer2DView<T,core::TargetHost>& buf, GLenum data_format = GL_DEPTH_COMPONENT)
    {
        download(buf.ptr(), data_format, internal::GLTypeTraits<typename core::type_traits<T>::ChannelType>::opengl_data_type);
    }
    
    void download(GLvoid* data, GLenum data_format = GL_DEPTH_COMPONENT, GLenum data_type = GL_FLOAT)
    {
        glReadPixels(0,0,width(),height(),data_format,data_type,data);
    }
    
    inline GLuint id() const { return rbid; }
    inline GLint width() const { return rbw; }
    inline GLint height() const { return rbh; }
private:
    GLuint rbid;
    GLint rbw;
    GLint rbh;
};

class FrameBuffer
{
    constexpr static std::size_t MAX_ATTACHMENTS = 8;
    static std::array<GLenum,MAX_ATTACHMENTS> attachment_buffers;
public:    
    FrameBuffer() : attachments(0)
    {
        glGenFramebuffers(1, &fbid);
    }
    
    virtual ~FrameBuffer()
    {
        glDeleteFramebuffers(1, &fbid);
        fbid = 0;
    }
    
    inline GLenum attach(Texture2D& tex)
    {
        const GLenum color_attachment = GL_COLOR_ATTACHMENT0 + attachments;
        glFramebufferTexture2D(GL_FRAMEBUFFER, color_attachment, GL_TEXTURE_2D, tex.id(), 0);
        attachments++;
        return color_attachment;
    }
    
    /**
     * This needs GL_DEPTH_COMPONENT24 / GL_DEPTH texture.
     */
    inline GLenum attachDepth(Texture2D& tex)
    {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex.id(), 0);
        return GL_DEPTH_ATTACHMENT;
    }
    
    inline GLenum attachDepth(RenderBuffer& rb )
    {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb.id());
        return GL_DEPTH_ATTACHMENT;
    }
    
    inline bool isValid() const { return fbid != 0; }
    
    void bind() const
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbid);   
    }
    
    void unbind() const
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    void drawInto() const
    {
        glDrawBuffers( attachments, attachment_buffers.data() );
    }
    
    static GLenum checkComplete() 
    {
        return glCheckFramebufferStatus(GL_FRAMEBUFFER);
    }
    
    void clearBuffer(unsigned int idx, float* val)
    {
        glClearBufferfv(GL_COLOR, idx, val);
    }
    
    void clearDepthBuffer(float val)
    {
        glClearBufferfv(GL_DEPTH, 0, &val);
    }
    
    inline GLuint id() const { return fbid; }
    inline unsigned int colorAttachmentCount() const { return attachments; }
private:
    GLuint fbid;
    unsigned int attachments;
};
    
}

#endif // WRAPGL_FRAME_BUFFER_HPP
