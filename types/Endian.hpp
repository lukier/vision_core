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
 * Wrapper for endianess.
 * ****************************************************************************
 */

#ifndef CORE_TYPES_ENDIAN_HPP
#define CORE_TYPES_ENDIAN_HPP

#include <Platform.hpp>

namespace core
{
    
namespace types
{

enum class Endianness
{
    Little = 0,
    Big
};
    
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
static constexpr Endianness PlatformEndian = Endianness::Little;
#else 
static constexpr Endianness PlatformEndian = Endianness::Big;
#endif

namespace internal
{
    template<class T, Endianness esrc, Endianness edst> 
    struct endian_converter
    {
        static inline T convert(const T& other)
        {
            T ret_value = 0;
            for(int  i = 0 ; i < sizeof(T) ; ++i) 
            {
                reinterpret_cast<char *>(&ret_value)[i] = reinterpret_cast<const char*>(&other)[sizeof(T)-i-1];
            }
            return ret_value;
        }
    };
    
    template<class T, Endianness esrc> 
    struct endian_converter<T,esrc,esrc>
    {
        static inline T convert(const T& other)
        {
            return other;
        }
    };
}

#pragma pack(push, 1)
template<class T, Endianness esrc = PlatformEndian> 
class EndianWrapper
{
public:
    typedef T value_type;
    static constexpr Endianness value_endian = esrc;
    
    inline EndianWrapper() = default;
    inline EndianWrapper(const T& native) { value = native; }
    template<Endianness eother>
    inline EndianWrapper(const EndianWrapper<T,eother>& e) : value( internal::endian_converter<T,eother,value_endian>::convert(e.getNative()) ) { }
    
    inline operator T () const { return get(); }
    
    template<Endianness eother = PlatformEndian>
    inline T get() const { return internal::endian_converter<T,value_endian,eother>::convert(value); }
    inline T getNative() const { return value; }
    
    template<Endianness eother = PlatformEndian>
    inline void set(const T& other) { value = internal::endian_converter<T,eother,value_endian>::convert(other); }
    inline void setNative(const T& other) { value = other; }
private:
    T value;
};
#pragma pack(pop)
    
}

}

#endif // CORE_TYPES_ENDIAN_HPP
