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
 * Fourier Transform.
 * ****************************************************************************
 */

#ifndef CORE_IMAGE_FOURIER_HPP
#define CORE_IMAGE_FOURIER_HPP

#include <type_traits>

#include <buffers/Buffer1D.hpp>
#include <buffers/Buffer2D.hpp>
#include <buffers/Image2D.hpp>

namespace core
{
    
namespace image
{

/**
 * Versatile 2D Fast Fourier Transform.
 *
 * Performed transforms:
 * 
 * R2C: // forward
 *  float -> float2
 *  float -> Eigen::Vector2f
 * 
 * C2R: // inverse
 *  float2 -> float
 *  Eigen::Vector2f -> float
 *
 * C2C: // forward & inverse
 *  Eigen::Vector2f & float2, only Buffer2D
 * 
 */

template<typename T_INPUT, typename T_OUTPUT>
class PersistentTransform
{
public:
    PersistentTransform(const core::Buffer1DView<T_INPUT, core::TargetDeviceCUDA>& buf_in, 
                        core::Buffer1DView<T_OUTPUT, core::TargetDeviceCUDA>& buf_out, bool forward = true);
    PersistentTransform(const core::Buffer2DView<T_INPUT, core::TargetDeviceCUDA>& buf_in, 
                        core::Buffer2DView<T_OUTPUT, core::TargetDeviceCUDA>& buf_out, bool forward = true);
    virtual ~PersistentTransform();
    
    void transform();
private:
    int plan;
    void* input_ptr;
    void* output_ptr;
};

template<typename T_INPUT, typename T_OUTPUT>
void transform(const core::Buffer1DView<T_INPUT, core::TargetDeviceCUDA>& buf_in, 
               core::Buffer1DView<T_OUTPUT, core::TargetDeviceCUDA>& buf_out, bool forward = true);

template<typename T_INPUT, typename T_OUTPUT>
void transform(const core::Buffer2DView<T_INPUT, core::TargetDeviceCUDA>& buf_in, 
               core::Buffer2DView<T_OUTPUT, core::TargetDeviceCUDA>& buf_out, bool forward = true);

template<typename T_COMPLEX>
void splitComplex(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_real, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_imag);

template<typename T_COMPLEX>
void joinComplex(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_real, const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_imag, core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_out);

template<typename T_COMPLEX>
void magnitude(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);

template<typename T_COMPLEX>
void phase(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);

template<typename T_COMPLEX>
void convertToComplex(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_out);

template<typename T_COMPLEX>
void calculateCrossPowerSpectrum(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft1, 
                                 const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft2, 
                                 core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft_out);

}

}


#endif // CORE_IMAGE_FOURIER_HPP
