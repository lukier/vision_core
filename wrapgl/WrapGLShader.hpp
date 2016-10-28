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
 * Shaders.
 * ****************************************************************************
 */

#ifndef WRAPGL_SHADER_HPP
#define WRAPGL_SHADER_HPP

#include <string>
#include <sstream>
#include <fstream>

#include <wrapgl/WrapGLCommon.hpp>
#include <wrapgl/WrapGLTexture.hpp>
#include <wrapgl/WrapGLBuffer.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

/**
 * TODO:
 * 
 * - glValidateProgram
 * - glTransformFeedbackVaryings​
 * 
 * Add class: ProgramPipelines
 */

namespace wrapgl
{
    
class Shader
{
public:    
    enum class Type : GLuint
    {
        Vertex = (GLuint)GL_VERTEX_SHADER,
        Fragment = (GLuint)GL_FRAGMENT_SHADER,
        Geometry = (GLuint)GL_GEOMETRY_SHADER,
        TessellationControl = (GLuint)GL_TESS_CONTROL_SHADER,
        TessellationEvaluation = (GLuint)GL_TESS_EVALUATION_SHADER,
        Compute = (GLuint)GL_COMPUTE_SHADER,
    };
    
    Shader() : progid(0), linked(false)
    {
        create();
    }
    
    virtual ~Shader()
    {
        destroy();
    }
    
    std::pair<bool,std::string> addShaderFromSourceCode(Type type, const char* source)
    {
        bool was_ok = false;
        std::string err_log;
        
        GLhandleARB shader = glCreateShader((GLenum)type);
        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);
        
        GLint status;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if(status != (GLint)GL_TRUE) 
        {
            constexpr std::size_t SHADER_LOG_MAX_LEN = 10240;
            err_log.resize(SHADER_LOG_MAX_LEN);
            GLsizei len;
            glGetShaderInfoLog(shader, SHADER_LOG_MAX_LEN, &len, const_cast<char*>(err_log.data()));
            err_log.resize(len);
        }
        else
        {
            was_ok = true;
            glAttachShader(progid, shader);
        }
        
        return std::make_pair(was_ok,err_log);
    }
    
    std::pair<bool,std::string> addShaderFromSourceFile(Type type, const char* fn)
    {
        std::ifstream ifs(fn);
        if(ifs)
        {
            std::ostringstream contents;
            contents << ifs.rdbuf();
            ifs.close();
            
            return addShaderFromSourceCode(type, contents.str().c_str());
        }
        else
        {
            return std::make_pair(false,"No such file");
        }
    }
    
    void removeAllShaders()
    {
        if(progid != 0)
        {
            for(auto& sh : shaders)
            {
                glDetachShader(progid, sh);
                glDeleteShader(sh);
            }
            
            linked = false;
        }
    }
    
    std::pair<bool,std::string> link()
    {
        bool was_ok = false;
        std::string err_log;
        
        glLinkProgram(progid);
        
        GLint status;
        glGetProgramiv(progid, GL_LINK_STATUS, &status);
        if(status != (GLint)GL_TRUE) 
        {
            constexpr std::size_t PROGRAM_LOG_MAX_LEN = 10240;
            err_log.resize(PROGRAM_LOG_MAX_LEN);
            GLsizei len;
            glGetProgramInfoLog(progid, PROGRAM_LOG_MAX_LEN, &len, const_cast<char*>(err_log.data()));
            err_log.resize(len);
        }
        else
        {
            was_ok = true;
            linked = true;
        }
        
        return std::make_pair(was_ok,err_log);
    }
    
    bool isLinked() const { return linked; }
    
    bool isValid() const { return progid != 0; }
        
    void bind() const
    {
        glUseProgram(progid);
    }
    
    void unbind() const
    {
        glUseProgram(0);
    }
    
    void create()
    {
        if(isValid()) { destroy(); }
        
        progid = glCreateProgram();
        linked = false;
    }
    
    void destroy()
    {
        if(progid != 0) 
        {
            removeAllShaders();
            glDeleteProgram(progid);
            progid = 0;
        }
    }
    
    void dispatchCompute(GLuint num_groups_x, GLuint num_groups_y = 0, GLuint num_groups_z = 0) const
    {
        glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    }
    
    void memoryBarrier(MemoryBarrierMask mbm =  GL_ALL_BARRIER_BITS)
    {
        glMemoryBarrier(mbm);
    }
    
    inline GLuint id() const { return progid; }

    void bindAttributeLocation(const char* name, int location)
    {
        glBindAttribLocation(progid, location, name);
        linked = false;
    }
    
    inline GLint attributeLocation(const char* name) const { return glGetAttribLocation(progid, name); }
   
    void setAttributeValue(int location, GLfloat value)
    {
        glVertexAttrib1f(location, value);
    }
    
    void setAttributeValue(int location, GLfloat x, GLfloat y)
    {
        glVertexAttrib2f(location, x, y);
    }
    
    void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z)
    {
        glVertexAttrib3f(location, x, y, z);
    }
    
    void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
    {
        glVertexAttrib4f(location, x, y, z, w);
    }
    
    void setAttributeValue(int location, const Eigen::Matrix<float,2,1>& value) { setAttributeValue(location, value(0), value(1)); }
    void setAttributeValue(int location, const Eigen::Matrix<float,3,1>& value) { setAttributeValue(location, value(0), value(1), value(2)); }
    void setAttributeValue(int location, const Eigen::Matrix<float,4,1>& value) { setAttributeValue(location, value(0), value(1), value(2), value(3)); }
    void setAttributeValue(int location, const float2& value) { setAttributeValue(location, value.x, value.y); }
    void setAttributeValue(int location, const float3& value) { setAttributeValue(location, value.x, value.y, value.z); }
    void setAttributeValue(int location, const float4& value) { setAttributeValue(location, value.x, value.y, value.z, value.w); }
    
    void setAttributeValue(const char* name, GLfloat value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, GLfloat x, GLfloat y) { setAttributeValue(attributeLocation(name), x, y); }
    void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z) { setAttributeValue(attributeLocation(name), x, y, z); }
    void setAttributeValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { setAttributeValue(attributeLocation(name), x, y, z, w); }
    void setAttributeValue(const char* name, const Eigen::Matrix<float,2,1>& value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, const Eigen::Matrix<float,3,1>& value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, const Eigen::Matrix<float,4,1>& value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, const float2& value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, const float3& value) { setAttributeValue(attributeLocation(name), value); }
    void setAttributeValue(const char* name, const float4& value) { setAttributeValue(attributeLocation(name), value); }
     
    void setAttributeArray(int location, const GLfloat* values, int tupleSize, int stride = 0)
    {
        glVertexAttribPointer(location, tupleSize, GL_FLOAT, GL_FALSE, stride, values);
    }
    
    void setAttributeArray(int location, const Eigen::Matrix<float,2,1>& value, int stride = 0) { setAttributeArray(location, value.data(), 2, stride); }
    void setAttributeArray(int location, const Eigen::Matrix<float,3,1>& value, int stride = 0) { setAttributeArray(location, value.data(), 3, stride); }
    void setAttributeArray(int location, const Eigen::Matrix<float,4,1>& value, int stride = 0) { setAttributeArray(location, value.data(), 4, stride); }
    
    void setAttributeArray(int location, GLenum type, const void* values, int tupleSize, int stride = 0) 
    {
        glVertexAttribPointer(location, tupleSize, type, GL_TRUE, stride, values); // NOTE
    }
    
    void setAttributeArray(const char* name, const GLfloat* values, int tupleSize, int stride = 0) { setAttributeArray(attributeLocation(name), values, tupleSize, stride); }
    void setAttributeArray(const char* name, const Eigen::Matrix<float,2,1>& value, int stride = 0) { setAttributeArray(attributeLocation(name), value, stride); }
    void setAttributeArray(const char* name, const Eigen::Matrix<float,3,1>& value, int stride = 0) { setAttributeArray(attributeLocation(name), value, stride); }
    void setAttributeArray(const char* name, const Eigen::Matrix<float,4,1>& value, int stride = 0) { setAttributeArray(attributeLocation(name), value, stride); }
    void setAttributeArray(const char* name, GLenum type, const void* values, int tupleSize, int stride = 0) { setAttributeArray(attributeLocation(name), type, values, stride); }
    
    void setAttributeBuffer(int location, GLenum type, int offset, int tupleSize, int stride = 0)
    {
        glVertexAttribPointer(location, tupleSize, type, GL_TRUE, stride, reinterpret_cast<const void *>(offset));
    }
    void setAttributeBuffer(const char* name, GLenum type, int offset, int tupleSize, int stride = 0) { setAttributeBuffer(attributeLocation(name), type, offset, tupleSize, stride); }
    
    void enableAttributeArray(int location)
    {
        glEnableVertexAttribArray(location);
    }
    void enableAttributeArray(const char* name) { enableAttributeArray(attributeLocation(name)); }
    
    void disableAttributeArray(int location)
    {
        glDisableVertexAttribArray(location);
    }
    void disableAttributeArray(const char* name) { disableAttributeArray(attributeLocation(name)); }
    
    inline GLint uniformLocation(const char* name) const { return glGetUniformLocation(progid, name); }
    
    void setUniformValue(int location, GLfloat value)
    {
        glUniform1f(location, value);
    }
    
    void setUniformValue(int location, GLint value)
    {
        glUniform1i(location, value);
    }
    
    void setUniformValue(int location, GLuint value)
    {
        glUniform1ui(location, value);
    }
    
    void setUniformValue(int location, GLfloat x, GLfloat y)
    {
        glUniform2f(location, x, y);
    }
    
    void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z)
    {
        glUniform3f(location, x, y, z);
    }
    
    void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
    {
        glUniform4f(location, x, y, z, w);
    }
    
    void setUniformValue(int location, const Eigen::Matrix<float,2,1>& value) { setUniformValue(location, value(0), value(1)); }
    void setUniformValue(int location, const Eigen::Matrix<float,3,1>& value) { setUniformValue(location, value(0), value(1), value(2)); }
    void setUniformValue(int location, const Eigen::Matrix<float,4,1>& value) { setUniformValue(location, value(0), value(1), value(2), value(3)); }
    void setUniformValue(int location, const float2& value) { setUniformValue(location, value.x, value.y); }
    void setUniformValue(int location, const float3& value) { setUniformValue(location, value.x, value.y, value.z); }
    void setUniformValue(int location, const float4& value) { setUniformValue(location, value.x, value.y, value.z, value.w); }
    
    void setUniformValue(int location, const Eigen::Matrix<float,2,2>& value) { glUniformMatrix2fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,2,3>& value) { glUniformMatrix2x3fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,2,4>& value) { glUniformMatrix2x4fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,3,2>& value) { glUniformMatrix3x2fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,3,3>& value) { glUniformMatrix3fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,3,4>& value) { glUniformMatrix3x4fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,4,2>& value) { glUniformMatrix4x2fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,4,3>& value) { glUniformMatrix4x3fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const Eigen::Matrix<float,4,4>& value) { glUniformMatrix4fv(location, 1, GL_FALSE, value.data()); }
    void setUniformValue(int location, const GLfloat value[2][2]) { glUniformMatrix2fv(location, 1, GL_FALSE, value[0]); }
    void setUniformValue(int location, const GLfloat value[3][3]) { glUniformMatrix3fv(location, 1, GL_FALSE, value[0]); }
    void setUniformValue(int location, const GLfloat value[4][4]) { glUniformMatrix4fv(location, 1, GL_FALSE, value[0]); }
    void setUniformValue(int location, const Sophus::SE3f& value)
    {
        const Sophus::SE3f::Transformation m = value.matrix();
        glUniformMatrix4fv(location, 1, GL_FALSE, m.data());
    }
    
    void setUniformValue(const char* name, GLfloat value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, GLint value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, GLuint value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, GLfloat x, GLfloat y) { setUniformValue(uniformLocation(name), x, y); }
    void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z) { setUniformValue(uniformLocation(name), x, y, z); }
    void setUniformValue(const char* name, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { setUniformValue(uniformLocation(name), x, y, z, w); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,2,1>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,3,1>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,4,1>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const float2& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const float3& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const float4& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,2,2>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,2,3>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,2,4>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,3,2>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,3,3>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,3,4>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,4,2>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,4,3>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Eigen::Matrix<float,4,4>& value) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const GLfloat value[2][2]) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const GLfloat value[3][3]) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const GLfloat value[4][4]) { setUniformValue(uniformLocation(name), value); }
    void setUniformValue(const char* name, const Sophus::SE3f& value) { setUniformValue(uniformLocation(name), value); }
    
    void setUniformValueArray(int location, const GLfloat* values, int count, int tupleSize)
    {
        if (tupleSize == 1)
        {
            glUniform1fv(location, count, values);
        }
        else if (tupleSize == 2)
        {
            glUniform2fv(location, count, values);
        }
        else if (tupleSize == 3)
        {
            glUniform3fv(location, count, values);
        }
        else if (tupleSize == 4)
        {
            glUniform4fv(location, count, values);
        }
    }
    
    void setUniformValueArray(int location, const GLint* values, int count) { glUniform1iv(location, count, values); }
    void setUniformValueArray(int location, const GLuint* values, int count) { glUniform1uiv(location, count, values); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,2,1>* values, int count) { glUniform2fv(location, count, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,3,1>* values, int count) { glUniform3fv(location, count, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,4,1>* values, int count) { glUniform4fv(location, count, values[0].data()); }
    
    void setUniformValueArray(int location, const Eigen::Matrix<float,2,2>* values, int count) { glUniformMatrix2fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,2,3>* values, int count) { glUniformMatrix2x3fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,2,4>* values, int count) { glUniformMatrix2x4fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,3,2>* values, int count) { glUniformMatrix3x2fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,3,3>* values, int count) { glUniformMatrix3fv(location, count, GL_FALSE, values[0].data()); } 
    void setUniformValueArray(int location, const Eigen::Matrix<float,3,4>* values, int count) { glUniformMatrix3x4fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,4,2>* values, int count) { glUniformMatrix4x2fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,4,3>* values, int count) { glUniformMatrix4x3fv(location, count, GL_FALSE, values[0].data()); }
    void setUniformValueArray(int location, const Eigen::Matrix<float,4,4>* values, int count) { glUniformMatrix4fv(location, count, GL_FALSE, values[0].data()); }
    
    void setUniformValueArray(const char* name, const GLfloat* values, int count, int tupleSize) { setUniformValueArray(uniformLocation(name), values, count, tupleSize); }
    void setUniformValueArray(const char* name, const GLint* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const GLuint* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,1>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,1>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,1>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,2>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,3>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,2,4>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,2>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,3>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,3,4>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,2>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,3>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    void setUniformValueArray(const char* name, const Eigen::Matrix<float,4,4>* values, int count) { setUniformValueArray(uniformLocation(name), values, count); }
    
    void bindImage(const wrapgl::Texture2D& tex, GLuint unit, GLenum access = GL_READ_ONLY, GLenum intfmt = GL_R32F) const 
    {
        glBindImageTexture(unit, /* unit, note that we're not offseting GL_TEXTURE0 */
                               tex.id(), /* a 2D texture for example */
                               0, /* miplevel */
                               GL_FALSE, /* we cannot use layered */
                               0, /* this is ignored */
                               access, /* we're only writing to it */
                               intfmt/* interpret format as 32-bit float */);
    }
    
    void unbindImage(GLuint unit)
    {
        glBindImageTexture(unit, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
    }
    
    GLuint getMaxImageUnits() const { return internal::getParameter<GLint>(GL_MAX_IMAGE_UNITS); }
    
    GLint getFragmentDataLocation(const char* name) const { return glGetFragDataLocation(progid, name); }
    void bindFragmentDataLocation(const char* name, GLuint color) { glBindFragDataLocation(progid, color, name); }
    
    void setTransformFeedbackVaryings(GLsizei count, const char **varyings, GLenum bufmode = GL_INTERLEAVED_ATTRIBS)
    {
        glTransformFeedbackVaryings(progid, count, varyings, bufmode);
    }
    
    inline GLuint uniformBlockLocation(const char* name) const { return glGetUniformBlockIndex(progid, name); }
    
    void bindUniformBuffer(GLuint location, const Buffer& buf)
    {
        glBindBufferBase(buf.type(), location, buf.id());
    }
    
    void bindUniformBuffer(const char* name, const Buffer& buf)
    {
        bindUniformBuffer(uniformBlockLocation(name),buf);
    }
    
    void bindUniformBufferRange(GLuint location, const Buffer& buf, GLintptr offset, GLsizeiptr size)
    {
        glBindBufferRange(buf.type(), location, buf.id(), offset, size);
    }
    
    void bindUniformBufferRange(const char* name, const Buffer& buf, GLintptr offset, GLsizeiptr size)
    {
        bindUniformBufferRange(uniformBlockLocation(name), buf, offset, size);
    }
private:
    GLuint progid;
    std::vector<GLhandleARB> shaders;
    bool linked;
};


}

#endif // WRAPGL_SHADER_HPP
