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
 * Convolution.
 * ****************************************************************************
 */

#include <Platform.hpp>

#include <LaunchUtils.hpp>
#include <CUDAException.hpp>

#include <image/Convolution.hpp>

template<typename _Scalar, typename Target, typename KernelT>
__global__ void Kernel_convolveBuffer(core::Buffer2DView<_Scalar,Target> img_in, core::Buffer2DView<_Scalar,Target> img_out, KernelT kern)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(img_in.inBounds(x,y) && img_out.inBounds(x,y)) // is valid
    {
        const int split_x = KernelT::RowsAtCompileTime/2;
        const int split_y = KernelT::ColsAtCompileTime/2;
        
        _Scalar sum = core::zero<_Scalar>();
        _Scalar kernsum = core::zero<_Scalar>();
        
        for(int py = -split_y ; py <= split_y ; ++py)
        {
            for(int px = -split_x ; px <= split_x ; ++px)
            {
                const _Scalar& pix = img_in.getWithClampedRange((int)x + px, (int)y + py);
                const _Scalar& kv = kern( split_x + px , split_y + py );
                sum += (pix * kv);
                kernsum += kv;
            }
        }
        
        img_out(x,y) = sum / kernsum;
    }
}

template<typename T, typename Target, typename T2>
struct ConvolutionDispatcherGPU;

template<typename Target, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ConvolutionDispatcherGPU<_Scalar, Target, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> KernelT;
    
    static void convolve(const core::Buffer2DView<_Scalar,Target>& img_in, core::Buffer2DView<_Scalar,Target>& img_out, const KernelT& kern)
    {
        dim3 gridDim, blockDim;
        
        core::InitDimFromOutputImageOver(blockDim, gridDim, img_in);
        
        // run kernel
        Kernel_convolveBuffer<_Scalar,Target,KernelT><<<gridDim,blockDim>>>(img_in, img_out, kern);
        
        // wait for it
        const cudaError err = cudaDeviceSynchronize();
        if(err != cudaSuccess)
        {
            throw core::CUDAException(err, "Error launching the kernel");
        }
    }
};

template<typename T, typename Target, typename T2>
void core::image::convolve(const core::Buffer2DView<T,Target>& img_in, core::Buffer2DView<T,Target>& img_out, const T2& kern)
{
    return ConvolutionDispatcherGPU<T,Target,T2>::convolve(img_in, img_out, kern);
}

// CPU float
template void core::image::convolve<float,core::TargetDeviceCUDA, Eigen::Matrix<float,3,3> >(const core::Buffer2DView<float,core::TargetDeviceCUDA>& img_in, core::Buffer2DView<float,core::TargetDeviceCUDA>& img_out, const Eigen::Matrix<float,3,3>& kern);
template void core::image::convolve<float,core::TargetDeviceCUDA, Eigen::Matrix<float,5,5> >(const core::Buffer2DView<float,core::TargetDeviceCUDA>& img_in, core::Buffer2DView<float,core::TargetDeviceCUDA>& img_out, const Eigen::Matrix<float,5,5>& kern);
template void core::image::convolve<float,core::TargetDeviceCUDA, Eigen::Matrix<float,7,7> >(const core::Buffer2DView<float,core::TargetDeviceCUDA>& img_in, core::Buffer2DView<float,core::TargetDeviceCUDA>& img_out, const Eigen::Matrix<float,7,7>& kern);
template void core::image::convolve<float,core::TargetDeviceCUDA, Eigen::Matrix<float,9,9> >(const core::Buffer2DView<float,core::TargetDeviceCUDA>& img_in, core::Buffer2DView<float,core::TargetDeviceCUDA>& img_out, const Eigen::Matrix<float,9,9>& kern);
