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
#include <CUDAException.hpp>
#include <image/PixelConvert.hpp>

#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>

#include <math/LossFunctions.hpp>
#include <buffers/ReductionSum2D.hpp>

#include <image/BufferOps.hpp>
#include <image/sources/JoinSplitHelpers.hpp>

template<typename T>
struct rescale_functor : public thrust::unary_function<T, T>
{    
    rescale_functor(T al, T be, T cmin, T cmax) : alpha(al), beta(be), clamp_min(cmin), clamp_max(cmax) { }
    
    EIGEN_DEVICE_FUNC T operator()(T val)
    {
        return clamp(val * alpha + beta, clamp_min, clamp_max); 
    }
    
    T alpha, beta, clamp_min, clamp_max;
};

template<typename T, typename Target>
void core::image::rescaleBufferInplace(core::Buffer1DView< T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    thrust::transform(buf_in.begin(), buf_in.end(), buf_in.begin(), rescale_functor<T>(alpha, beta, clamp_min, clamp_max) );
}

template<typename T, typename Target>
void core::image::rescaleBufferInplace(core::Buffer2DView<T, Target>& buf_in, T alpha, T beta, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, alpha, beta, clamp_min, clamp_max);
}

template<typename T, typename Target>
void core::image::rescaleBufferInplaceMinMax(core::Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min, T clamp_max)
{
    rescaleBuffer(buf_in, buf_in, T(1.0f) / (vmax - vmin), -vmin * (T(1.0)/(vmax - vmin)), clamp_min, clamp_max);
}

template<typename T1, typename T2, typename Target>
__global__ void Kernel_rescaleBuffer(const core::Buffer2DView<T1, Target> buf_in, 
                                     core::Buffer2DView<T2, Target> buf_out, float alpha, float beta, float clamp_min, float clamp_max)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        const T2 val = core::image::convertPixel<T1,T2>(buf_in(x,y));
        buf_out(x,y) = clamp(val * alpha + beta, clamp_min, clamp_max); 
    }
}

template<typename T1, typename T2, typename Target>
void core::image::rescaleBuffer(const core::Buffer2DView<T1, Target>& buf_in, core::Buffer2DView<T2, Target>& buf_out, float alpha, float beta, float clamp_min, float clamp_max)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_rescaleBuffer<T1,T2><<<gridDim,blockDim>>>(buf_in, buf_out, alpha, beta, clamp_min, clamp_max);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
void core::image::normalizeBufferInplace(core::Buffer2DView< T, Target >& buf_in)
{
    const T min_val = calcBufferMin(buf_in);
    const T max_val = calcBufferMax(buf_in);

    rescaleBufferInplace(buf_in, T(1.0f) / (max_val - min_val), -min_val * (T(1.0)/(max_val - min_val)));
}

template<typename T>
struct clamp_functor : public thrust::unary_function<T, T>
{    
    clamp_functor(T al, T be) : alpha(al), beta(be) { }
    
    EIGEN_DEVICE_FUNC T operator()(T val)
    {
        return clamp(val, alpha, beta); 
    }
    
    T alpha, beta;
};

template<typename T, typename Target>
void core::image::clampBuffer(core::Buffer1DView<T, Target>& buf_io, T a, T b)
{
    thrust::transform(buf_io.begin(), buf_io.end(), buf_io.begin(), clamp_functor<T>(a, b) );
}

template<typename T, typename Target>
__global__ void Kernel_clampBuffer(core::Buffer2DView<T, Target> buf_io, T a, T b)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_io.inBounds((int)x,(int)y)) // is valid
    {
        buf_io(x,y) = clamp(buf_io(x,y),a,b);
    }
}

template<typename T, typename Target>
void core::image::clampBuffer(core::Buffer2DView<T, Target>& buf_io, T a, T b)
{
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_io);
    
    // run kernel
    Kernel_clampBuffer<T,Target><<<gridDim,blockDim>>>(buf_io, a, b);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
T core::image::calcBufferMin(const core::Buffer1DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::min_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T core::image::calcBufferMax(const core::Buffer1DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::max_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T core::image::calcBufferMean(const core::Buffer1DView< T, Target >& buf_in)
{
    T sum = thrust::reduce(buf_in.begin(), buf_in.end());
    return sum / buf_in.size();
}

template<typename T, typename Target>
T core::image::calcBufferMin(const core::Buffer2DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::min_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T core::image::calcBufferMax(const core::Buffer2DView< T, Target >& buf_in)
{
    thrust::device_ptr<T> iter = thrust::max_element(buf_in.begin(), buf_in.end());
    return iter[0];
}

template<typename T, typename Target>
T core::image::calcBufferMean(const core::Buffer2DView< T, Target >& buf_in)
{
    T sum = thrust::reduce(buf_in.begin(), buf_in.end());
    return sum / buf_in.area();
}

template<typename T, typename Target>
__global__ void Kernel_leaveQuarter(const core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        buf_out(x,y) = buf_in(2*x + 1,2*y + 1);
    }
}

template<typename T, typename Target>
void core::image::leaveQuarter(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_leaveQuarter<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_downsampleHalf(const core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        
        buf_out(x,y) = (T)(*tl + *(tl+1) + *bl + *(bl+1)) / 4;
    }
}

template<typename T, typename Target>
void core::image::downsampleHalf(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_downsampleHalf<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_downsampleHalfNoInvalid(const core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        const T* tl = buf_in.ptr(2*x,2*y);
        const T* bl = buf_in.ptr(2*x,2*y+1);
        const T v1 = *tl;
        const T v2 = *(tl+1);
        const T v3 = *bl;
        const T v4 = *(bl+1);
        
        int n = 0;
        T sum = core::zero<T>();
        
        if(core::isvalid(v1)) { sum += v1; n++; }
        if(core::isvalid(v2)) { sum += v2; n++; }
        if(core::isvalid(v3)) { sum += v3; n++; }
        if(core::isvalid(v4)) { sum += v4; n++; }     
        
        buf_out(x,y) = n > 0 ? (T)(sum / n) : core::getInvalid<T>();
    }
}

template<typename T, typename Target>
void core::image::downsampleHalfNoInvalid(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    dim3 gridDim, blockDim;
    
    if(!( (buf_in.width()/2 == buf_out.width()) && (buf_in.height()/2 == buf_out.height())))
    {
        throw std::runtime_error("In/Out dimensions don't match");
    }
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_downsampleHalfNoInvalid<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join2(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in2, core::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join3(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in3, core::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_join4(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_in4, core::Buffer2DView<TCOMP, Target> buf_out)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::join(buf_in1(x,y), buf_in2(x,y), buf_in3(x,y), buf_in4(x,y), buf_out(x,y));
    }
}

template<typename TCOMP, typename Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join2<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join3<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_in3, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in4, core::Buffer2DView<TCOMP, Target>& buf_out)
{
    assert((buf_out.width() == buf_in1.width()) && (buf_out.height() == buf_in1.height()));
    assert((buf_in1.width() == buf_in2.width()) && (buf_in1.height() == buf_in2.height()));
    assert((buf_in2.width() == buf_in3.width()) && (buf_in2.height() == buf_in3.height()));
    assert((buf_in3.width() == buf_in4.width()) && (buf_in3.height() == buf_in4.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_join4<TCOMP,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_in3, buf_in4, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split2(const core::Buffer2DView<TCOMP, Target> buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out2)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split3(const core::Buffer2DView<TCOMP, Target> buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out3)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y));
    }
}

template<typename TCOMP, typename Target>
__global__ void Kernel_split4(const core::Buffer2DView<TCOMP, Target> buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target> buf_out4)
{
    // current point
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds((int)x,(int)y)) // is valid
    {
        ::internal::JoinSplitHelper<TCOMP>::split(buf_in(x,y), buf_out1(x,y), buf_out2(x,y), buf_out3(x,y), buf_out4(x,y));
    }
}

template<typename TCOMP, typename Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split2<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split3<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2, buf_out3);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename TCOMP, typename Target>
void core::image::split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out4)
{
    assert((buf_in.width() == buf_out1.width()) && (buf_in.height() == buf_out1.height()));
    assert((buf_out1.width() == buf_out2.width()) && (buf_out1.height() == buf_out2.height()));
    assert((buf_out2.width() == buf_out3.width()) && (buf_out2.height() == buf_out3.height()));
    assert((buf_out3.width() == buf_out4.width()) && (buf_out3.height() == buf_out4.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_split4<TCOMP,Target><<<gridDim,blockDim>>>(buf_in, buf_out1, buf_out2, buf_out3, buf_out4);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_fillBuffer1D(core::Buffer1DView<T, Target> buf_in, const typename core::type_traits<T>::ChannelType v)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(buf_in.inBounds(x)) // is valid
    {
        buf_in(x) = core::internal::type_dispatcher_helper<T>::fill(v);
    }
}

template<typename T, typename Target>
__global__ void Kernel_fillBuffer2D(core::Buffer2DView<T, Target> buf_in, const typename core::type_traits<T>::ChannelType v)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        buf_in(x,y) = core::internal::type_dispatcher_helper<T>::fill(v);
    }
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void core::image::fillBuffer(core::Buffer1DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v)
{
    dim3 gridDim, blockDim;

    core::InitDimFromLinearBuffer(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_fillBuffer1D<T,Target><<<gridDim,blockDim>>>(buf_in, v);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

/**
 * fillBuffer
 */
template<typename T, typename Target>
void core::image::fillBuffer(core::Buffer2DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_fillBuffer2D<T,Target><<<gridDim,blockDim>>>(buf_in, v);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_invertBuffer2D(core::Buffer2DView<T, Target> buf_io)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_io.inBounds(x,y)) // is valid
    {
        buf_io(x,y) = ::internal::JoinSplitHelper<T>::invertedValue(buf_io(x,y));
    }
}

/**
 * Invert Buffer
 */
template<typename T, typename Target>
void core::image::invertBuffer(core::Buffer2DView<T, Target>& buf_io)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_io);
    
    // run kernel
    Kernel_invertBuffer2D<T,Target><<<gridDim,blockDim>>>(buf_io);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_thresholdBufferSimple(core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out, T thr, T val_below, T val_above )
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
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
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void core::image::thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_thresholdBufferSimple<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out, thr, val_below, val_above);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_thresholdBufferAdvanced(core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation )
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
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
}

/**
 * Threshold Buffer
 */
template<typename T, typename Target>
void core::image::thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_thresholdBufferAdvanced<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out, thr, val_below, val_above, minval, maxval, saturation);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T, typename Target>
__global__ void Kernel_flipXBuffer(const core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        const std::size_t nx = ((buf_in.width() - 1) - x);
        if(buf_in.inBounds(nx,y))
        {
            buf_out(x,y) = buf_in(nx,y);
        }
    }
}

/**
 * Flip X.
 */
template<typename T, typename Target>
void core::image::flipXBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_flipXBuffer<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_flipYBuffer(const core::Buffer2DView<T, Target> buf_in, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        const std::size_t ny = ((buf_in.height() - 1) - y);
        if(buf_in.inBounds(x,ny))
        {
            buf_out(x,y) = buf_in(x,ny);
        }
    }
}

/**
 * Flip Y.
 */
template<typename T, typename Target>
void core::image::flipYBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in.width() == buf_out.width()) && (buf_in.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_flipYBuffer<T,Target><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstract(const core::Buffer2DView<T, Target> buf_in1, const core::Buffer2DView<T, Target> buf_in2, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = buf_in1(x,y) - buf_in2(x,y);
    }
}

template<typename T, typename Target>
void core::image::bufferSubstract(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstract<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstractL1(const core::Buffer2DView<T, Target> buf_in1, const core::Buffer2DView<T, Target> buf_in2, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = core::math::lossL1(buf_in1(x,y) - buf_in2(x,y));
    }
}

template<typename T, typename Target>
void core::image::bufferSubstractL1(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstractL1<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
__global__ void Kernel_bufferSubstractL2(const core::Buffer2DView<T, Target> buf_in1, const core::Buffer2DView<T, Target> buf_in2, core::Buffer2DView<T, Target> buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = core::math::lossL2(buf_in1(x,y) - buf_in2(x,y));
    }
}

template<typename T, typename Target>
void core::image::bufferSubstractL2(const core::Buffer2DView<T, Target>& buf_in1, const core::Buffer2DView<T, Target>& buf_in2, core::Buffer2DView<T, Target>& buf_out)
{
    assert((buf_in1.width() == buf_out.width()) && (buf_in1.height() == buf_out.height()));
    assert((buf_in2.width() == buf_out.width()) && (buf_in2.height() == buf_out.height()));
    
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_bufferSubstractL2<T,Target><<<gridDim,blockDim>>>(buf_in1, buf_in2, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
}

template<typename T, typename Target>
T core::image::bufferSum(const core::Buffer1DView<T, Target>& buf_in, const T& initial)
{
    return thrust::reduce(buf_in.begin(), buf_in.end(), initial);
}

template<typename T, typename Target>
__global__ void Kernel_bufferSum2D(const core::Buffer2DView<T, Target> buf_in, core::HostReductionSum2DView<T> reductor)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    typedef core::DeviceReductionSum2DDynamic<T> ReductionT;
    
    ReductionT sumIt;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        sumIt.getThisBlock() = buf_in(x,y);
    }
    
    sumIt.reduceBlock(reductor);
}

template<typename T, typename Target>
T core::image::bufferSum(const core::Buffer2DView<T, Target>& buf_in, const T& initial)
{
    dim3 gridDim, blockDim;
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    core::HostReductionSum2DManaged<T> reductor(gridDim);
    
    // run kernel
    Kernel_bufferSum2D<T,Target><<<gridDim,blockDim,reductor.getSharedMemorySize(blockDim)>>>(buf_in, reductor);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
    
    T final_sum;
    
    reductor.getFinalSum(final_sum);
    
    return initial + final_sum;
}

#define JOIN_SPLIT_FUNCTIONS2(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in2, core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out2); 

#define JOIN_SPLIT_FUNCTIONS3(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in3, core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out3); 

#define JOIN_SPLIT_FUNCTIONS4(TCOMP) \
template void core::image::join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_in4, core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_out); \
template void core::image::split(const core::Buffer2DView<TCOMP, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, core::TargetDeviceCUDA>& buf_out4); \

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

// instantiations
template void core::image::rescaleBufferInplace<float, core::TargetDeviceCUDA>(core::Buffer1DView< float, core::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBufferInplace<float, core::TargetDeviceCUDA>(core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBufferInplaceMinMax<float, core::TargetDeviceCUDA>(core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_in, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::normalizeBufferInplace<float, core::TargetDeviceCUDA>(core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_in);

template void core::image::rescaleBuffer<uint8_t, float, core::TargetDeviceCUDA>(const core::Buffer2DView<uint8_t, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<uint16_t, float, core::TargetDeviceCUDA>(const core::Buffer2DView<uint16_t, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<uint32_t, float, core::TargetDeviceCUDA>(const core::Buffer2DView<uint32_t, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, float, core::TargetDeviceCUDA>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

template void core::image::rescaleBuffer<float, uint8_t, core::TargetDeviceCUDA>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<uint8_t, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, uint16_t, core::TargetDeviceCUDA>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<uint16_t, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);
template void core::image::rescaleBuffer<float, uint32_t, core::TargetDeviceCUDA>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<uint32_t, core::TargetDeviceCUDA>& buf_out, float alpha, float beta, float clamp_min, float clamp_max);

// statistics

#define MIN_MAX_MEAN_THR_FUNCS(BUF_TYPE) \
template BUF_TYPE core::image::calcBufferMin<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer1DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE core::image::calcBufferMax<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer1DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE core::image::calcBufferMean<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer1DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE core::image::calcBufferMin<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE core::image::calcBufferMax<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template BUF_TYPE core::image::calcBufferMean<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView< BUF_TYPE, core::TargetDeviceCUDA >& buf_in); \
template void core::image::thresholdBuffer<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView< BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above); \
template void core::image::thresholdBuffer<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView< BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out, BUF_TYPE thr, BUF_TYPE val_below, BUF_TYPE val_above, BUF_TYPE minval, BUF_TYPE maxval, bool saturation);

MIN_MAX_MEAN_THR_FUNCS(float)
MIN_MAX_MEAN_THR_FUNCS(uint8_t)
MIN_MAX_MEAN_THR_FUNCS(uint16_t)

// various

#define SIMPLE_TYPE_FUNCS(BUF_TYPE) \
template void core::image::leaveQuarter<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out); \
template void core::image::downsampleHalf<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out); \
template void core::image::fillBuffer<BUF_TYPE, core::TargetDeviceCUDA>(core::Buffer1DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, const typename core::type_traits<BUF_TYPE>::ChannelType& v); \
template void core::image::fillBuffer<BUF_TYPE, core::TargetDeviceCUDA>(core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, const typename core::type_traits<BUF_TYPE>::ChannelType& v); \
template void core::image::invertBuffer<BUF_TYPE, core::TargetDeviceCUDA>(core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_io); \
template void core::image::flipXBuffer(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out); \
template void core::image::flipYBuffer(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out); \
template BUF_TYPE core::image::bufferSum(const core::Buffer1DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, const BUF_TYPE& initial);\
template BUF_TYPE core::image::bufferSum(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, const BUF_TYPE& initial);

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
template void core::image::clampBuffer(core::Buffer1DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_io, BUF_TYPE a, BUF_TYPE b); \
template void core::image::clampBuffer(core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_io, BUF_TYPE a, BUF_TYPE b);

CLAMP_FUNC_TYPES(uint8_t)
CLAMP_FUNC_TYPES(uint16_t)
CLAMP_FUNC_TYPES(float)
CLAMP_FUNC_TYPES(float2)
CLAMP_FUNC_TYPES(float3)
CLAMP_FUNC_TYPES(float4)

#define STUPID_FUNC_TYPES(BUF_TYPE)\
template void core::image::bufferSubstract(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out);\
template void core::image::bufferSubstractL1(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out);\
template void core::image::bufferSubstractL2(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in1, const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in2, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out);
STUPID_FUNC_TYPES(float)

//template void core::image::downsampleHalfNoInvalid<BUF_TYPE, core::TargetDeviceCUDA>(const core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<BUF_TYPE, core::TargetDeviceCUDA>& buf_out); 
