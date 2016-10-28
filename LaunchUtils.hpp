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
 * Launching parallel processing.
 * ****************************************************************************
 */

#ifndef CORE_LAUNCH_UTILS_HPP
#define CORE_LAUNCH_UTILS_HPP

#include <Platform.hpp>

#ifdef CORE_HAVE_TBB
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#endif // CORE_HAVE_TBB

namespace core
{
    
namespace detail
{
    template<typename T>
    inline int Gcd(T a, T b)
    {
        const T amodb = a%b;
        return amodb ? Gcd(b, amodb) : b;
    }
}
    
/// @note: From Kangaroo, @todo fix
template<typename T>
inline void InitDimFromOutputImage(dim3& blockDim, dim3& gridDim, const T& image, unsigned blockx = 32, unsigned blocky = 32)
{
    blockDim = dim3( detail::Gcd<unsigned>(image.width(), blockx), detail::Gcd<unsigned>(image.height(), blocky), 1);
    gridDim =  dim3( image.width() / blockDim.x, image.height() / blockDim.y, 1);
}

template<typename T>
inline void InitDimFromOutputImageOver(dim3& blockDim, dim3& gridDim, const T& image, int blockx = 32, int blocky = 32)
{
    blockDim = dim3(blockx, blocky);
    gridDim =  dim3( ceil(image.width() / (double)blockDim.x), ceil(image.height() / (double)blockDim.y) );
}

template<typename T>
inline void InitDimFromLinearBuffer(dim3& blockDim, dim3& gridDim, const T& lin_buffer, int blockx = 32)
{
    blockDim = dim3(detail::Gcd<unsigned>(lin_buffer.size(), blockx), 1, 1);
    gridDim =  dim3( lin_buffer.size() / blockDim.x, 1, 1);
}

template<typename T>
inline void InitDimFromLinearBufferOver(dim3& blockDim, dim3& gridDim, const T& lin_buffer, int blockx = 32)
{
    blockDim = dim3(blockx);
    gridDim =  dim3( ceil(lin_buffer.size() / (double)blockDim.x));
}

inline void InitDimFromDimensionsOver(dim3& blockDim, dim3& gridDim, int dimx, int dimy, int blockx = 32, int blocky = 32)
{
    blockDim = dim3(blockx, blocky);
    gridDim =  dim3( ceil(dimx / (double)blockDim.x), ceil(dimy / (double)blockDim.y) );
}

inline dim3 calculateBlockDim(int blockx, int blocky)
{
    return dim3(blockx, blocky);    
}

inline dim3 calculateGridDim(dim3 blockDim, int dimx, int dimy)
{
    return dim3( ceil(dimx / (double)blockDim.x), ceil(dimy / (double)blockDim.y) );
}


/// TBB!
#ifdef CORE_HAVE_TBB
template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dim, PerItemFunction pif)
{
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, dim), [&](const tbb::blocked_range<std::size_t>& r)
    {
        for(std::size_t i = r.begin() ; i != r.end() ; ++i )
        {
            pif(i);
        }
    });
}

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dimx, std::size_t dimy, PerItemFunction pif)
{
    tbb::parallel_for(tbb::blocked_range2d<std::size_t>(0, dimy, 0, dimx), [&](const tbb::blocked_range2d<std::size_t>& r)
    {
        for(std::size_t y = r.rows().begin() ; y != r.rows().end() ; ++y )
        {
            for(std::size_t x = r.cols().begin() ; x != r.cols().end() ; ++x ) 
            {
                pif(x,y);
            }
        }
    });
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dim, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    return tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, dim), initial,
    [&](const tbb::blocked_range<std::size_t>& r, const VT& v)
    {
        VT ret = v;
        for(std::size_t i = r.begin() ; i != r.end() ; ++i )
        {
            pif(i, ret);
        }
        return ret;
    },
    [&](const VT& v1, const VT& v2)
    {
        return jf(v1, v2);
    }
    );
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dimx, std::size_t dimy, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    return tbb::parallel_reduce(tbb::blocked_range2d<std::size_t>(0, dimy, 0, dimx), initial,
    [&](const tbb::blocked_range2d<std::size_t>& r, const VT& v)
    {
        VT ret = v;
        
        for(std::size_t y = r.rows().begin() ; y != r.rows().end() ; ++y )
        {
            for(std::size_t x = r.cols().begin() ; x != r.cols().end() ; ++x ) 
            {
                pif(x,y,ret);
            }
        }
        return ret;
    },
    [&](const VT& v1, const VT& v2)
    {
        return jf(v1, v2);
    }
    );
}

#else // CORE_HAVE_TBB

/// no TBB = no parallelism at all

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dim, PerItemFunction pif)
{
    for(std::size_t i = 0 ; i < dim ; ++i )
    {
        pif(i);
    }
}

template<typename PerItemFunction>
static inline void launchParallelFor(std::size_t dimx, std::size_t dimy, PerItemFunction pif)
{
    for(std::size_t y = 0 ; y < dimy ; ++y)
    {
        for(std::size_t x = 0 ; x < dimy ; ++x)
        {
            pif(x,y);
        }
    }
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dim, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    VT ret = initial;
    
    for(std::size_t i = 0 ; i < dim ; ++i )
    {
        pif(i, ret);
    }
    
    return ret;
}

template<typename VT, typename PerItemFunction, typename JoinFunction>
static inline VT launchParallelReduce(std::size_t dimx, std::size_t dimy, const VT& initial, PerItemFunction pif, JoinFunction jf)
{
    VT ret = initial;
    
    for(std::size_t y = 0 ; y < dimy ; ++y)
    {
        for(std::size_t x = 0 ; x < dimy ; ++x)
        {
            pif(x,y, ret);
        }
    }
    
    return ret;
}

#endif // CORE_HAVE_TBB

}
#endif // CORE_LAUNCH_UTILS_HPP
