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
 * Queries.
 * ****************************************************************************
 */

#ifndef VISIONCORE_WRAPGL_QUERY_HPP
#define VISIONCORE_WRAPGL_QUERY_HPP

#include <VisionCore/WrapGL/WrapGLCommon.hpp>

/**
 * TODO:
 * 
 * glGetQueryObject
 * 
 * glGetQueryiv
 * glGetQueryIndexed
 */

namespace vc
{

namespace wrapgl
{
    
class Query
{
public:    
    Query() : qid(0)
    {
        create();
    }
    
    virtual ~Query()
    {
        destroy();
    }
    
    void create()
    {
        destroy();
        
        glGenQueries(1, &qid);
    }
    
    void destroy()
    {
        if(qid != 0)
        {
            glDeleteQueries(1, &qid);
            qid = 0;
        }
    }
    
    inline bool isValid() const { return qid != 0; }
    
    void begin(GLenum target) const
    {
        glBeginQuery(target, qid);
    }
    
    void end(GLenum target) const
    {
        glEndQuery(target);
    }
    
    void begin(GLenum target, GLuint idx) const
    {
        glBeginQueryIndexed(target, idx, qid);
    }
    
    void end(GLenum target, GLuint idx) const
    {
        glEndQueryIndexed(target, idx);
    }
    
    GLint get(GLenum target, GLenum pname)
    {
        GLint ret = 0;
        glGetQueryiv(target, pname, &ret);
        return ret;
    }
    
    GLint get(GLenum target, GLuint index, GLenum pname)
    {
        GLint ret = 0;
        glGetQueryIndexediv(target, index, pname, &ret);
        return ret;
    }
    
    void queryTimestamp()
    {
        glQueryCounter(qid, GL_TIMESTAMP);
    }
    
    inline GLuint id() const { return qid; }
private:
    GLuint qid;
};

}
    
}

#endif // VISIONCORE_WRAPGL_QUERY_HPP
