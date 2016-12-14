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

#ifndef CORE_IMAGE_BUFFEROPS_HPP
#define CORE_IMAGE_BUFFEROPS_HPP

#include <type_traits>

#include <buffers/Buffer1D.hpp>
#include <buffers/Buffer2D.hpp>
#include <buffers/Image2D.hpp>
#include <buffers/ImagePyramid.hpp>

/**
 * @note No dimension checking for now, also Thrust is non-pitched.
 */

namespace core
{

namespace image
{
    
/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, template<typename> class Target>
void rescaleBufferInplace(core::Buffer1DView<T, Target>& buf_in, T alpha, T beta = 0.0f, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, template<typename> class Target>
void rescaleBufferInplace(core::Buffer2DView<T, Target>& buf_in, T alpha, T beta = 0.0f, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T, template<typename> class Target>
void rescaleBufferInplaceMinMax(core::Buffer2DView<T, Target>& buf_in, T vmin, T vmax, T clamp_min = 0.0f, T clamp_max = 1.0f);

/**
 * Rescale element-wise and clamp.
 * out = in * alpha + beta
 */
template<typename T1, typename T2, template<typename> class Target>
void rescaleBuffer(const core::Buffer2DView<T1, Target>& buf_in, core::Buffer2DView<T2, Target>& buf_out, float alpha, float beta = 0.0f, float clamp_min = 0.0f, float clamp_max = 1.0f);

/**
 * Normalize buffer to 0..1 range (float only) for now.
 */
template<typename T, template<typename> class Target>
void normalizeBufferInplace(core::Buffer2DView<T, Target>& buf_in);

/**
 * Clamp Buffer
 */
template<typename T, template<typename> class Target>
void clampBuffer(core::Buffer1DView<T, Target>& buf_io, T a, T b);

/**
 * Clamp Buffer
 */
template<typename T, template<typename> class Target>
void clampBuffer(core::Buffer2DView<T, Target>& buf_io, T a, T b);


/**
 * Find minimal value of a 1D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMin(const core::Buffer1DView<T, Target>& buf_in);

/**
 * Find maximal value of a 1D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMax(const core::Buffer1DView<T, Target>& buf_in);

/**
 * Find mean value of a 1D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMean(const core::Buffer1DView<T, Target>& buf_in);

/**
 * Find minimal value of a 2D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMin(const core::Buffer2DView<T, Target>& buf_in);

/**
 * Find maximal value of a 2D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMax(const core::Buffer2DView<T, Target>& buf_in);

/**
 * Find mean value of a 2D buffer.
 */
template<typename T, template<typename> class Target>
T calcBufferMean(const core::Buffer2DView<T, Target>& buf_in);

/**
 * Downsample by half.
 */
template<typename T, template<typename> class Target>
void downsampleHalf(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out);

/**
 * Downsample by half (ignore invalid).
 */
template<typename T, template<typename> class Target>
void downsampleHalfNoInvalid(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out);

/**
 * Leave even rows and columns.
 */
template<typename T, template<typename> class Target>
void leaveQuarter(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out);

/**
 * Fills remaining pyramid levels with downsampleHalf.
 */
template<typename T, std::size_t Levels, template<typename> class Target>
static inline void fillPyramidBilinear(core::ImagePyramidView<T,Levels,Target>& pyr)
{
    for(std::size_t l = 1 ; l < Levels ; ++l) 
    {
        downsampleHalf(pyr[l-1],pyr[l]);
    }
}

/**
 * Fills remaining pyramid levels with leaveQuarter.
 */
template<typename T, std::size_t Levels, template<typename> class Target>
static inline void fillPyramidCrude(core::ImagePyramidView<T,Levels,Target>& pyr)
{
    for(std::size_t l = 1 ; l < Levels ; ++l) 
    {
        leaveQuarter(pyr[l-1],pyr[l]);
    }
}

/**
 * Join buffers.
 */
template<typename TCOMP, template<typename> class Target>
void join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, core::Buffer2DView<TCOMP, Target>& buf_out);
template<typename TCOMP, template<typename> class Target>
void join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, core::Buffer2DView<TCOMP, Target>& buf_out);
template<typename TCOMP, template<typename> class Target>
void join(const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in1, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in2, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in3, const core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_in4, core::Buffer2DView<TCOMP, Target>& buf_out);

/**
 * Split buffers.
 */
template<typename TCOMP, template<typename> class Target>
void split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2);
template<typename TCOMP, template<typename> class Target>
void split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3);
template<typename TCOMP, template<typename> class Target>
void split(const core::Buffer2DView<TCOMP, Target>& buf_in, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out1, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out2, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out3, core::Buffer2DView<typename core::type_traits<TCOMP>::ChannelType, Target>& buf_out4);

/**
 * fillBuffer
 */
template<typename T, template<typename> class Target>
void fillBuffer(core::Buffer1DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v);

/**
 * fillBuffer
 */
template<typename T, template<typename> class Target>
void fillBuffer(core::Buffer2DView<T, Target>& buf_in, const typename core::type_traits<T>::ChannelType& v);

/**
 * Invert Buffer
 */
template<typename T, template<typename> class Target>
void invertBuffer(core::Buffer1DView<T, Target>& buf_io);

/**
 * Invert Buffer
 */
template<typename T, template<typename> class Target>
void invertBuffer(core::Buffer2DView<T, Target>& buf_io);

/**
 * Threshold Buffer
 */
template<typename T, template<typename> class Target>
void thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above);

/**
 * Threshold Buffer
 */
template<typename T, template<typename> class Target>
void thresholdBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out, T thr, T val_below, T val_above, T minval, T maxval, bool saturation = false);

/**
 * Flip X.
 */
template<typename T, template<typename> class Target>
void flipXBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out);

/**
 * Flip Y.
 */
template<typename T, template<typename> class Target>
void flipYBuffer(const core::Buffer2DView<T, Target>& buf_in, core::Buffer2DView<T, Target>& buf_out);

/**
 * Substract.
 */
template<typename T, template<typename> class Target>
void bufferSubstract(const core::Buffer2DView<T, Target>& buf_in1,
                     const core::Buffer2DView<T, Target>& buf_in2,
                     core::Buffer2DView<T, Target>& buf_out);

/**
 * Substract L1.
 */
template<typename T, template<typename> class Target>
void bufferSubstractL1(const core::Buffer2DView<T, Target>& buf_in1,
                       const core::Buffer2DView<T, Target>& buf_in2,
                       core::Buffer2DView<T, Target>& buf_out);

/**
 * Substract L2.
 */
template<typename T, template<typename> class Target>
void bufferSubstractL2(const core::Buffer2DView<T, Target>& buf_in1,
                       const core::Buffer2DView<T, Target>& buf_in2,
                       core::Buffer2DView<T, Target>& buf_out);

template<typename T, template<typename> class Target>
T bufferSum(const core::Buffer1DView<T, Target>& buf_in, const T& initial);

template<typename T, template<typename> class Target>
T bufferSum(const core::Buffer2DView<T, Target>& buf_in, const T& initial);
}

}


#endif // CORE_IMAGE_BUFFEROPS_HPP
