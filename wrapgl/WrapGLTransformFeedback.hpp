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
 * Transform feedback.
 * ****************************************************************************
 */

#ifndef WRAPGL_TRANSFORM_FEEDBACK_HPP
#define WRAPGL_TRANSFORM_FEEDBACK_HPP

#include <wrapgl/WrapGLCommon.hpp>

namespace wrapgl
{
    
class TransformFeedback
{
public:    
    TransformFeedback() : tbid(0)
    {
        create();
    }
        
    virtual ~TransformFeedback()
    {
        destroy();
    }
    
    void create()
    {
        destroy();
        
        glGenTransformFeedbacks(1, &tbid);
    }
    
    void destroy()
    {
        if(tbid != 0)
        {
            glDeleteTransformFeedbacks(1, &tbid);
            tbid = 0;
        }
    }
    
    inline bool isValid() const { return tbid != 0; }
    
    void bind() const
    {
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tbid);
    }
    
    void unbind() const
    {
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tbid);
    }
    
    void draw(GLenum mode = GL_POINTS) const
    {
        glDrawTransformFeedback(mode, tbid);
    }
    
    void draw(GLenum mode, GLsizei instcnt) const
    {
        glDrawTransformFeedbackInstanced(mode, tbid, instcnt);
    }
    
    static void begin(GLenum primode)
    {
        glBeginTransformFeedback(primode);
    }
    
    static void end()
    {
        glEndTransformFeedback();
    }
    
    static void pause()
    {
        glPauseTransformFeedback();
    }
    
    static void resume()
    {
        glPauseTransformFeedback();
    }
    
    inline GLuint id() const { return tbid; }
private:
    GLuint tbid;
};
    
}

#endif // WRAPGL_TRANSFORM_FEEDBACK_HPP
