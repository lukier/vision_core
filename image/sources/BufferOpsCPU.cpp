/**
 * ****************************************************************************
 * Copyright (c) 2016, Robert Lukierski.
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
 * Various operations on buffers.
 * ****************************************************************************
 */

#include <Platform.hpp>
#include <LaunchUtils.hpp>
#include <math/LossFunctions.hpp>

#include <image/BufferOps.hpp>
#include <image/sources/JoinSplitHelpers.hpp>

#include <numeric>

template<typename T, template<typename> class Target>
void core::image::rescaleBufferInplace(core::Buffer1DView< T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    std::transform(buf_in.ptr(), buf_in.ptr() + buf_in.size(), buf_in.ptr(), [&](T val) -> T 
    {
        return clamp(val * alpha + beta, clamp_min, clamp_max); 
    });
}

template<typename T, template<typename> class Target>
void core::image::rescaleBufferInplace(core::Buffer2DView<T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, alpha, beta, clamp_min, clamp_max);
}

template<typename T, template<typename> class Target>
void core::image::rescaleBufferInplaceMinMax(core::Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, T(1.0f) / (vmax - vmin), -vmin * (T(1.0)/(vmax - vmin)), clamp_min, clamp_max);
}

template<typename T1, typename T2, template<typename> class Target>
void core::image::rescaleBuffer(const core::Buffer2DView<T1, Target>& buf_in, core::Buffer2DView<T2, Target>& buf_out, float alpha, float beta, float clamp_min, float clamp_max)
{
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y) && buf_out.inBounds(x,y))
        {
            const T2 val = core::image::convertPixel<T1,T2>(buf_in(x,y));
            buf_out(x,y) = clamp(val * alpha + beta, clamp_min, clamp_max); 
        }
    });
}

template<typename T, template<typename> class Target>
void core::image::normalizeBufferInplace(core::Buffer2DView< T, Target >& buf_in)
{
    const T min_val = calcBufferMin(buf_in);
    const T max_val = calcBufferMax(buf_in);

    rescaleBufferInplace(buf_in, T(1.0f) / (max_val - min_val), -min_val * (T(1.0)/(max_val - min_val)));
}

template<typename T, template<typename> class Target>
void core::image::clampBuffer(core::Buffer1DView<T, Target>& buf_io, T a, T b)
{
    core::launchParallelFor(buf_io.size(), [&](std::size_t idx)
    {
        if(buf_io.inBounds(idx))
        {
            buf_io(idx) = clamp(buf_io(idx), a, b); 
        }
    });
}

template<typename T, template<typename> class Target>
void core::image::clampBuffer(core::Buffer2DView<T, Target>& buf_io, T a, T b)
{
    core::launchParallelFor(buf_io.width(), buf_io.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_io.inBounds(x,y))
        {
            buf_io(x,y) = clamp(buf_io(x,y), a, b); 
        }
    });
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMin(const core::Buffer1DView< T, Target >& buf_in)
{
    const T* ret = std::min_element(buf_in.ptr(), buf_in.ptr() + buf_in.size());
    return *ret;
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMax(const core::Buffer1DView< T, Target >& buf_in)
{
    const T* ret = std::max_element(buf_in.ptr(), buf_in.ptr() + buf_in.size());
    return *ret;
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMean(const core::Buffer1DView< T, Target >& buf_in)
{
    T sum = std::accumulate(buf_in.ptr(), buf_in.ptr() + buf_in.size(), core::zero<T>());
    return sum / (T)buf_in.size();
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMin(const core::Buffer2DView< T, Target >& buf_in)
{
    T minval = core::numeric_limits<T>::max();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            minval = std::min(minval, buf_in(x,y));
        }
    }
    
    return minval;
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMax(const core::Buffer2DView< T, Target >& buf_in)
{
    T maxval = core::numeric_limits<T>::lowest();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            maxval = std::max(maxval, buf_in(x,y));
        }
    }
    
    return maxval;
}

template<typename T, template<typename> class Target>
T core::image::calcBufferMean(const core::Buffer2DView< T, Target >& buf_in)
{
    T sum = core::zero<T>();
    
    for(std::size_t y = 0 ; y < buf_in.height() ; ++y)
    {
        for(std::size_t x = 0 ; x < buf_in.width() ; ++x)
        {
            sum += buf_in(x,y);
        }
    }
    
    return sum / (T)buf_in.area();
}

template<typename T, template<typename> class Target>
void core::image::downsampleHalf(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
            const T* tl = buf_in.ptr(2*x,2*y);
            const T* bl = buf_in.ptr(2*x,2*y+1);
            
            buf_out(x,y) = (T)(*tl + *(tl+1) + *bl + *(bl+1)) / 4;
    });
}

template<typename T, template<typename> class Target>
void core::image::downsampleHalfNoInvalid(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        const T v1 = *tl;
        const T v2 = *(tl+1);
        const T v3 = *bl;
        const T v4 = *(bl+1);
        
        int n = 0;
        T sum = 0;
        
        if(core::isvalid(v1)) { sum += v1; n++; }
        if(core::isvalid(v2)) { sum += v2; n++; }
        if(core::isvalid(v3)) { sum += v3; n++; }
        if(core::isvalid(v4)) { sum += v4; n++; }
        
        buf_out(x,y) = n > 0 ? (T)(sum / (T)n) : core::getInvalid<T>();
    });
}

template<typename T, template<typename> class Target>
void core::image::leaveQuarter(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        buf_out(x,y) = buf_in(2*x,2*y);
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in4, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    assert((buf_in3.width() == buf_in4.width()) && (buf_in3.height() == buf_in4.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_in4(x,y), buf_out(x,y));
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y));
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y));
    });
}

template<typename TCOMP, template<typename> class Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out4)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    assert((buf_out3.width() == buf_out4.width()) && (buf_out3.height() == buf_out4.height()));
    
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y), buf_out4(x,y));
    });
}

/**
 * fillBuffer
 */
template<typename T, template<typename> class Target>
void core::image::fillBuffer(core::Buffer1DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v)
{
    std::transform(buf_in.ptr(), buf_in.ptr() + buf_in.size(), buf_in.ptr(), [&](T val) -> T 
    {
        return core::internal::type_dispatcher_helper<T>::fill(v);
    });
}

/**
 * fillBuffer
 */
template<typename T, template<typename> class Target>
void core::image::fillBuffer(core::Buffer2DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v)
{
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        buf_in(x,y) = core::internal::type_dispatcher_helper<T>::fill(v);
    });
}

/**
 * Invert Buffer
 */
template<typename T, template<typename> class Target>
void core::image::invertBuffer(core::Buffer1DView<T, Target>& buf_io)
{
    //typedef typename core::type_traits<T>::ChannelType Scalar;
    
    std::transform(buf_io.ptr(), buf_io.ptr() + buf_io.size(), buf_io.ptr(), [&](T val) -> T 
    {
        return ::internal::JoinSplitHelper<T>::invertedValue(val);
    });
}

/**
 * Invert Buffer
 */
template<typename T, template<typename> class Target>
void core::image::invertBuffer(core::Buffer2DView<T, Target>& buf_io)
{
    core::launchParallelFor(buf_io.width(), buf_io.height(), [&](std::size_t x, std::size_t y)
    {
        buf_io(x,y) = ::internal::JoinSplitHelper<T>::invertedValue(buf_io(x,y));
    });
}

/**
 * Threshold Buffer
 */
template<typename T, template<typename> class Target>
void core::image::thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above)
{
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y)) // is valid
        {
            const T& val = buf_in(x,y);
            if(val < thr)
            {
                buf_out(x,y) = val_below;
            }
            else
            {
                buf_out(x,y) = val_above;
            }
        }
    });
}

/**
 * Threshold Buffer
 */
template<typename T, template<typename> class Target>
void core::image::thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation)
{
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_in.inBounds(x,y)) // is valid
        {
            T val = buf_in(x,y);
            
            if(saturation)
            {
                val = clamp(val, minval, maxval);
            }
            
            const T relative_val = (val - minval) / (maxval - minval);
            
            if(relative_val < thr)
            {
                buf_out(x,y) = val_below;
            }
            else
            {
                buf_out(x,y) = val_above;
            }
        }
    });
}

/**
 * Flip X.
 */
template<typename T, template<typename> class Target>
void core::image::flipXBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            const std::size_t nx = ((buf_in.width() - 1) - x);
            if(buf_in.inBounds(nx,y))
            {
                buf_out(x,y) = buf_in(nx,y);
            }
        }
    });
}

/**
 * Flip Y.
 */
template<typename T, template<typename> class Target>
void core::image::flipYBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            const std::size_t ny = ((buf_in.height() - 1) - y);
            if(buf_in.inBounds(x,ny))
            {
                buf_out(x,y) = buf_in(x,ny);
            }
        }
    });
}

template<typename T, template<typename> class Target>
void core::image::bufferSubstract(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = buf_in1(x,y) - buf_in2(x,y);
        }
    });
}

template<typename T, template<typename> class Target>
void core::image::bufferSubstractL1(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = core::math::lossL1(buf_in1(x,y) - buf_in2(x,y));
        }
    });
}

template<typename T, template<typename> class Target>
void core::image::bufferSubstractL2(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        if(buf_out.inBounds(x,y)) // is valid
        {
            buf_out(x,y) = core::math::lossL2(buf_in1(x,y) - buf_in2(x,y));
        }
    });
}

template<typename T, template<typename> class Target>
T core::image::bufferSum(const core::Buffer1DView<T, Target>& buf_in, const T& initial)
{
    return core::launchParallelReduce(buf_in.size(), initial,
    [&](const std::size_t i, T& v)
    {
        v += buf_in(i);
    },
    [&](const T& v1, const T& v2)
    {
        return v1 + v2;
    });
}

template<typename T, template<typename> class Target>
T core::image::bufferSum(const core::Buffer2DView<T, Target>& buf_in, const T& initial)
{
    return core::launchParallelReduce(buf_in.width(), buf_in.height(), initial,
    [&](const std::size_t x, const std::size_t y, T& v)
    {
        v += buf_in(x,y);
    },
    [&](const T& v1, const T& v2)
    {
        return v1 + v2;
    });
}

#define JOIN_SPLIT_FUNCTIONS2(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in2, core::Buffer2DView<TCOMP, core::TargetHost>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetHost>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out2); 

#define JOIN_SPLIT_FUNCTIONS3(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in3, core::Buffer2DView<TCOMP, core::TargetHost>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetHost>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out3); 

#define JOIN_SPLIT_FUNCTIONS4(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_in4, core::Buffer2DView<TCOMP, core::TargetHost>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetHost>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetHost>& buf_out4); \

// instantiations

JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2f)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3f)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4f)
JOIN_SPLIT_FUNCTIONS2(Eigen::Vector2d)
JOIN_SPLIT_FUNCTIONS3(Eigen::Vector3d)
JOIN_SPLIT_FUNCTIONS4(Eigen::Vector4d)

JOIN_SPLIT_FUNCTIONS2(float2)
JOIN_SPLIT_FUNCTIONS3(float3)
JOIN_SPLIT_FUNCTIONS4(float4)

// statistics

#define MIN_MAX_MEAN_THR_FUNCS(BUF_TYPE) \
template BUF_TYPE core::image::calcBufferMin<BUF_TYPE, core::TargetHost>(const core::Buffer1DView< BUF_TYPE, core::TargetHost >& buf_in); \
template BUF_TYPE core::image::calcBufferMax<BUF_TYPE, core::TargetHost>(const core::Buffer1DView< BUF_TYPE, core::TargetHost >& buf_in); \
template BUF_TYPE core::image::calcBufferMean<BUF_TYPE, core::TargetHost>(const core::Buffer1DView< BUF_TYPE, core::TargetHost >& buf_in); \
template BUF_TYPE core::image::calcBufferMin<BUF_TYPE, core::TargetHost>(const core::Buffer2DView< BUF_TYPE, core::TargetHost >& buf_in); \
template BUF_TYPE core::image::calcBufferMax<BUF_TYPE, core::TargetHost>(const core::Buffer2DView< BUF_TYPE, core::TargetHost >& buf_in); \
template BUF_TYPE core::image::calcBufferMean<BUF_TYPE, core::TargetHost>(const core::Buffer2DView< BUF_TYPE, core::TargetHost >& buf_in); \
template void core::image::thresholdBuffer<BUF_TYPE, core::TargetHost>(const core::Buffer2DView< BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above); \
template void core::image::thresholdBuffer<BUF_TYPE, core::TargetHost>(const core::Buffer2DView< BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above, BUF_TYPE minval, BUF_TYPE maxval, bool saturation);

MIN_MAX_MEAN_THR_FUNCS(float)
MIN_MAX_MEAN_THR_FUNCS(uint8_t)
MIN_MAX_MEAN_THR_FUNCS(uint16_t)

// various

#define SIMPLE_TYPE_FUNCS(BUF_TYPE) \
template void core::image::leaveQuarter<BUF_TYPE, core::TargetHost>(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out); \
template void core::image::downsampleHalf<BUF_TYPE, core::TargetHost>(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out); \
template void core::image::fillBuffer(core::Buffer1DView<BUF_TYPE, core::TargetHost>& buf_in, const typename core::type_traits<BUF_TYPE>::ChannelType& v); \
template void core::image::fillBuffer(core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, const typename core::type_traits<BUF_TYPE>::ChannelType& v); \
template void core::image::invertBuffer(core::Buffer1DView<BUF_TYPE, core::TargetHost>& buf_io); \
template void core::image::invertBuffer(core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_io);  \
template void core::image::flipXBuffer(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out); \
template void core::image::flipYBuffer(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out); \
template BUF_TYPE core::image::bufferSum(const core::Buffer1DView<BUF_TYPE, core::TargetHost>& buf_in, const BUF_TYPE& initial);\
template BUF_TYPE core::image::bufferSum(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in, const BUF_TYPE& initial);

SIMPLE_TYPE_FUNCS(uint8_t)
SIMPLE_TYPE_FUNCS(uint16_t)
SIMPLE_TYPE_FUNCS(uchar3)
SIMPLE_TYPE_FUNCS(uchar4)
SIMPLE_TYPE_FUNCS(float)
SIMPLE_TYPE_FUNCS(float3)
SIMPLE_TYPE_FUNCS(float4)
SIMPLE_TYPE_FUNCS(Eigen::Vector3f)
SIMPLE_TYPE_FUNCS(Eigen::Vector4f)

#define CLAMP_FUNC_TYPES(BUF_TYPE) \
template void core::image::clampBuffer(core::Buffer1DView<BUF_TYPE, core::TargetHost>& buf_io, BUF_TYPE a, BUF_TYPE b); \
template void core::image::clampBuffer(core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_io, BUF_TYPE a, BUF_TYPE b);

CLAMP_FUNC_TYPES(uint8_t)
CLAMP_FUNC_TYPES(uint16_t)
CLAMP_FUNC_TYPES(float)
CLAMP_FUNC_TYPES(float2)
CLAMP_FUNC_TYPES(float3)
CLAMP_FUNC_TYPES(float4)

#define STUPID_FUNC_TYPES(BUF_TYPE)\
template void core::image::bufferSubstract(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out);\
template void core::image::bufferSubstractL1(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out);\
template void core::image::bufferSubstractL2(const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetHost>& buf_out);
STUPID_FUNC_TYPES(float)

template void core::image::rescaleBufferInplace<float, core::TargetHost>(core::Buffer1DView< float, core::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBufferInplace<float, core::TargetHost>(core::Buffer2DView< float, core::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBufferInplaceMinMax<float, core::TargetHost>(core::Buffer2DView< float, core::TargetHost >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::normalizeBufferInplace<float, core::TargetHost>(core::Buffer2DView< float, core::TargetHost >& buf_in);

template void core::image::rescaleBuffer<uint8_t, float, core::TargetHost>(const core::Buffer2DView<uint8_t, core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<uint16_t, float, core::TargetHost>(const core::Buffer2DView<uint16_t, core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<uint32_t, float, core::TargetHost>(const core::Buffer2DView<uint32_t, core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, float, core::TargetHost>(const core::Buffer2DView<float, core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void core::image::rescaleBuffer<float, uint8_t, core::TargetHost>(const core::Buffer2DView<float, core::TargetHost>& buf_in, core::Buffer2DView<uint8_t, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, uint16_t, core::TargetHost>(const core::Buffer2DView<float, core::TargetHost>& buf_in, core::Buffer2DView<uint16_t, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, uint32_t, core::TargetHost>(const core::Buffer2DView<float, core::TargetHost>& buf_in, core::Buffer2DView<uint32_t, core::TargetHost>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void core::image::downsampleHalfNoInvalid<uint8_t, core::TargetHost>(const core::Buffer2DView<uint8_t, core::TargetHost>& buf_in, core::Buffer2DView<uint8_t, core::TargetHost>& buf_out);
template void core::image::downsampleHalfNoInvalid<uint16_t, core::TargetHost>(const core::Buffer2DView<uint16_t, core::TargetHost>& buf_in, core::Buffer2DView<uint16_t, core::TargetHost>& buf_out);
template void core::image::downsampleHalfNoInvalid<float, core::TargetHost>(const core::Buffer2DView<float, core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out);
