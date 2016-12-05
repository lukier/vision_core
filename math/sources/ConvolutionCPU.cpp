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

#include <math/Convolution.hpp>

#include <LaunchUtils.hpp>

template<typename T, typename Target, typename T2>
struct ConvolutionDispatcher;

template<typename Target, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
struct ConvolutionDispatcher<_Scalar, Target, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> >
{
    typedef Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> KernelT;
    
    static void convolve1D(const core::Buffer1DView<_Scalar,Target>& img_in, core::Buffer1DView<_Scalar,Target>& img_out, const KernelT& kern)
    {
        int split_x = _Rows/2;
        
        core::launchParallelFor(img_in.size(), [&](std::size_t x)
        {
            _Scalar sum = core::zero<_Scalar>();
            _Scalar kernsum = core::zero<_Scalar>();

            for(int px = -split_x ; px <= split_x ; ++px)
            {
                const _Scalar& pix = img_in.getWithClampedRange((int)x + px);
                const _Scalar& kv = kern( split_x + px , 0 );
                sum += (pix * kv);
                kernsum += kv;
            }
            
            img_out(x) = sum / kernsum;
        });
    }
    
    static void convolve2D(const core::Buffer2DView<_Scalar,Target>& img_in, core::Buffer2DView<_Scalar,Target>& img_out, const KernelT& kern)
    {
        int split_x = _Rows/2, split_y = _Cols/2;
        
        core::launchParallelFor(img_in.width(), img_in.height(), [&](std::size_t x, std::size_t y)
        {
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
        });
    }
};

template<typename T, typename Target, typename T2>
void core::math::convolve(const core::Buffer1DView<T,Target>& img_in, core::Buffer1DView<T,Target>& img_out, const T2& kern)
{
    return ConvolutionDispatcher<T,Target,T2>::convolve1D(img_in, img_out, kern);
}

template<typename T, typename Target, typename T2>
void core::math::convolve(const core::Buffer2DView<T,Target>& img_in, core::Buffer2DView<T,Target>& img_out, const T2& kern)
{
    return ConvolutionDispatcher<T,Target,T2>::convolve2D(img_in, img_out, kern);
}

// 1D CPU float
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,3,1> >(const core::Buffer1DView<float,core::TargetHost>& img_in, core::Buffer1DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,3,1>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,5,1> >(const core::Buffer1DView<float,core::TargetHost>& img_in, core::Buffer1DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,5,1>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,7,1> >(const core::Buffer1DView<float,core::TargetHost>& img_in, core::Buffer1DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,7,1>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,9,1> >(const core::Buffer1DView<float,core::TargetHost>& img_in, core::Buffer1DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,9,1>& kern);

// 1D CPU double
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,3,1> >(const core::Buffer1DView<double,core::TargetHost>& img_in, core::Buffer1DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,3,1>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,5,1> >(const core::Buffer1DView<double,core::TargetHost>& img_in, core::Buffer1DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,5,1>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,7,1> >(const core::Buffer1DView<double,core::TargetHost>& img_in, core::Buffer1DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,7,1>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,9,1> >(const core::Buffer1DView<double,core::TargetHost>& img_in, core::Buffer1DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,9,1>& kern);

// 2D CPU float
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,3,3> >(const core::Buffer2DView<float,core::TargetHost>& img_in, core::Buffer2DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,3,3>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,5,5> >(const core::Buffer2DView<float,core::TargetHost>& img_in, core::Buffer2DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,5,5>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,7,7> >(const core::Buffer2DView<float,core::TargetHost>& img_in, core::Buffer2DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,7,7>& kern);
template void core::math::convolve<float,core::TargetHost, Eigen::Matrix<float,9,9> >(const core::Buffer2DView<float,core::TargetHost>& img_in, core::Buffer2DView<float,core::TargetHost>& img_out, const Eigen::Matrix<float,9,9>& kern);

// 2D CPU double
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,3,3> >(const core::Buffer2DView<double,core::TargetHost>& img_in, core::Buffer2DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,3,3>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,5,5> >(const core::Buffer2DView<double,core::TargetHost>& img_in, core::Buffer2DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,5,5>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,7,7> >(const core::Buffer2DView<double,core::TargetHost>& img_in, core::Buffer2DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,7,7>& kern);
template void core::math::convolve<double,core::TargetHost, Eigen::Matrix<double,9,9> >(const core::Buffer2DView<double,core::TargetHost>& img_in, core::Buffer2DView<double,core::TargetHost>& img_out, const Eigen::Matrix<double,9,9>& kern);
