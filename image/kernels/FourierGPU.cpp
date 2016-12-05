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

#include <cufft.h>

#include <Platform.hpp>
#include <LaunchUtils.hpp>
#include <CUDAException.hpp>

#include <image/Fourier.hpp>

template<typename T>
struct FFTTypeTraits { };

template<>
struct FFTTypeTraits<Eigen::Vector2f>
{
    typedef float BaseType;
    static constexpr bool IsComplex = true;
    static constexpr unsigned int Fields = 2;
    static constexpr bool IsEigen = true;
};

template<>
struct FFTTypeTraits<cufftComplex>
{
    typedef float BaseType;
    static constexpr bool IsComplex = true;
    static constexpr unsigned int Fields = 2;
    static constexpr bool IsEigen = false;
};

template<>
struct FFTTypeTraits<float>
{
    typedef float BaseType;
    static constexpr bool IsComplex = false;
    static constexpr unsigned int Fields = 1;
    static constexpr bool IsEigen = false;
};

template<typename T_INPUT, typename T_OUTPUT>
struct FFTTransformType { };

// R2C

template<>
struct FFTTransformType<float, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_R2C;
    static constexpr char const* Name = "R2C";
    
    typedef cufftReal* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

template<>
struct FFTTransformType<float, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_R2C;
    static constexpr char const* Name = "R2C";
    
    typedef cufftReal* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

// C2R

template<>
struct FFTTransformType<cufftComplex, float>
{
    static constexpr bool ForwardFFT = false;
    static constexpr cufftType CUDAType = CUFFT_C2R;
    static constexpr char const* Name = "C2R";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftReal* OutputPtrType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, float>
{
    static constexpr bool ForwardFFT = false;
    static constexpr cufftType CUDAType = CUFFT_C2R;
    static constexpr char const* Name = "C2R";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftReal* OutputPtrType;
};

// C2C

template<>
struct FFTTransformType<cufftComplex, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

template<>
struct FFTTransformType<cufftComplex, Eigen::Vector2f>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

template<>
struct FFTTransformType<Eigen::Vector2f, cufftComplex>
{
    static constexpr bool ForwardFFT = true;
    static constexpr cufftType CUDAType = CUFFT_C2C;
    static constexpr char const* Name = "C2C";
    
    typedef cufftComplex* InputPtrType;
    typedef cufftComplex* OutputPtrType;
};

// -------------------------------------------

template<cufftType v>
struct ExecutionHelper
{
    
};

template<>
struct ExecutionHelper<CUFFT_R2C>
{
    static inline cufftResult exec(cufftHandle plan, cufftReal *idata, cufftComplex *odata, int dir = CUFFT_FORWARD)
    {
        return cufftExecR2C(plan, idata, odata);
    }
};

template<>
struct ExecutionHelper<CUFFT_C2R>
{
    static inline cufftResult exec(cufftHandle plan, cufftComplex *idata, cufftReal *odata, int dir = CUFFT_INVERSE)
    {
        return cufftExecC2R(plan, idata, odata);
    }
};

template<>
struct ExecutionHelper<CUFFT_C2C>
{
    static inline cufftResult exec(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int dir = CUFFT_FORWARD)
    {
        return cufftExecC2C(plan, idata, odata, dir);
    }
};

template<typename T_INPUT, typename T_OUTPUT>
struct ProperExecution
{
    static cufftResult exec(cufftHandle plan, const T_INPUT* buf_in, const T_OUTPUT* buf_out, bool fwd)
    {
        typedef typename FFTTransformType<T_INPUT,T_OUTPUT>::InputPtrType InputPtrType;
        typedef typename FFTTransformType<T_INPUT,T_OUTPUT>::OutputPtrType OutputPtrType;
        
        return ExecutionHelper<FFTTransformType<T_INPUT,T_OUTPUT>::CUDAType>::exec(plan, 
                                                                                   (InputPtrType)(buf_in), 
                                                                                   (OutputPtrType)(buf_out), 
                                                                                   fwd == true ? CUFFT_FORWARD : CUFFT_INVERSE);
    }
};

// -------------------------------------------

template<typename T_INPUT, typename T_OUTPUT>
void core::image::transform(const core::Buffer1DView<T_INPUT, core::TargetDeviceCUDA >& buf_in, 
                        core::Buffer1DView<T_OUTPUT, core::TargetDeviceCUDA >& buf_out, bool forward)
{
    cufftHandle plan;
    
    cufftResult res = cufftPlan1d(&plan, std::max(buf_in.size(), buf_out.size()), FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, 1);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }
    
    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_PADDING);
    
    res = ProperExecution<T_INPUT, T_OUTPUT>::exec(plan, buf_in.ptr(), buf_out.ptr(), forward);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
    
    cufftDestroy(plan);
}

template<typename T_INPUT, typename T_OUTPUT>
void core::image::transform(const core::Buffer2DView<T_INPUT, core::TargetDeviceCUDA>& buf_in, 
                        core::Buffer2DView<T_OUTPUT, core::TargetDeviceCUDA>& buf_out, bool forward)
{
    cufftHandle plan;
    
    int rank = 2; // 2D fft
    int n[] = {(int)(std::max(buf_in.height(), buf_out.height())), (int)(std::max(buf_in.width(), buf_out.width()))};    // Size of the Fourier transform
    int istride = 1, ostride = 1; // Stride lengths
    int idist = 1, odist = 1;     // Distance between batches
    int inembed[] = {(int)buf_in.height(), (int)(buf_in.pitch() / sizeof(T_INPUT))}; // Input size with pitch
    int onembed[] = {(int)buf_out.height(), (int)(buf_out.pitch() / sizeof(T_OUTPUT))}; // Output size with pitch
    int batch = 1;
    cufftResult res = cufftPlanMany(&plan, rank, n, 
                                    inembed, istride, idist,
                                    onembed, ostride, odist, 
                                    FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType, batch);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Plan Error"); }
    
    if(FFTTransformType<T_INPUT, T_OUTPUT>::CUDAType == CUFFT_C2R) { forward = false; }
    
    cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_FFTW_PADDING);
    
    res = ProperExecution<T_INPUT, T_OUTPUT>::exec(plan, buf_in.ptr(), buf_out.ptr(), forward);
    if(res != CUFFT_SUCCESS) { throw std::runtime_error("Exec Error"); }
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess) { throw core::CUDAException(err, "Error launching the kernel"); }
    
    cufftDestroy(plan);
}

template<typename T_COMPLEX, typename T_REAL>
struct complexOps { };

template<typename T_REAL>
struct complexOps<cufftComplex,T_REAL>
{
    EIGEN_DEVICE_FUNC static inline const T_REAL& getReal(const cufftComplex& cpx)
    {
        return cpx.x;
    }
    
    EIGEN_DEVICE_FUNC static inline const T_REAL& getImag(const cufftComplex& cpx)
    {
        return cpx.y;
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex makeComplex(const T_REAL& re, const T_REAL& im)
    {
        return make_cuComplex(re, im);
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex conjugate(const cufftComplex& cpx)
    {
        return make_cuComplex(cpx.x, -cpx.y);
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex multiply(const cufftComplex& cpx1, const cufftComplex& cpx2)
    {
        return make_cuComplex(cpx1.x * cpx2.x - cpx1.y * cpx2.y, cpx1.y * cpx2.x + cpx1.x * cpx2.y);
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex multiply(const cufftComplex& cpx1, const T_REAL& scalar)
    {
        return multiply(cpx1, make_cuComplex(scalar, 0.0f));
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex divide(const cufftComplex& cpx1, const cufftComplex& cpx2)
    {
        return make_cuComplex( (cpx1.x * cpx2.x + cpx1.y * cpx2.y) / (cpx2.x * cpx2.x + cpx2.y * cpx2.y) , (cpx1.y * cpx2.x - cpx1.x * cpx2.y) / (cpx2.x * cpx2.x + cpx2.y * cpx2.y) );
    }
    
    EIGEN_DEVICE_FUNC static inline cufftComplex divide(const cufftComplex& cpx1, const T_REAL& scalar)
    {
        return divide(cpx1, make_cuComplex(scalar, 0.0f));
    }
    
    EIGEN_DEVICE_FUNC static inline T_REAL norm(const cufftComplex& cpx)
    {
        return sqrt(cpx.x * cpx.x + cpx.y * cpx.y);
    }
};

template<typename T_REAL>
struct complexOps<Eigen::Vector2f,T_REAL>
{
    EIGEN_DEVICE_FUNC static inline const T_REAL& getReal(const Eigen::Vector2f& cpx)
    {
        return cpx(0);
    }
    
    EIGEN_DEVICE_FUNC static inline const T_REAL& getImag(const Eigen::Vector2f& cpx)
    {
        return cpx(1);
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f makeComplex(const T_REAL& re, const T_REAL& im)
    {
        return Eigen::Vector2f(re, im);
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f conjugate(const Eigen::Vector2f& cpx)
    {
        return Eigen::Vector2f(cpx(0), -cpx(1));
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f multiply(const Eigen::Vector2f& cpx1, const Eigen::Vector2f& cpx2)
    {
        return Eigen::Vector2f(cpx1(0) * cpx2(0) - cpx1(1) * cpx2(1), cpx1(1) * cpx2(0) + cpx1(0) * cpx2(1));
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f multiply(const Eigen::Vector2f& cpx1, const T_REAL& scalar)
    {
        return multiply(cpx1, Eigen::Vector2f(scalar, 0.0f));
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f divide(const Eigen::Vector2f& cpx1, const Eigen::Vector2f& cpx2)
    {
        return Eigen::Vector2f( (cpx1(0) * cpx2(0) + cpx1(1) * cpx2(1)) / (cpx2(0) * cpx2(0) + cpx2(1) * cpx2(1)) , (cpx1(1) * cpx2(0) - cpx1(0) * cpx2(1)) / (cpx2(0) * cpx2(0) + cpx2(1) * cpx2(1)) );
    }
    
    EIGEN_DEVICE_FUNC static inline Eigen::Vector2f divide(const Eigen::Vector2f& cpx1, const T_REAL& scalar)
    {
        return divide(cpx1, Eigen::Vector2f(scalar, 0.0f));
    }
    
    EIGEN_DEVICE_FUNC static inline T_REAL norm(const Eigen::Vector2f& cpx)
    {
        return sqrt(cpx(0) * cpx(0) + cpx(1) * cpx(1));
    }
};

template<typename T_COMPLEX, typename T_REAL>
__global__ void Kernel_splitComplex(const core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA > buf_in, 
                                    core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_real, 
                                    core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_imag)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_real(x,y) = complexOps<T_COMPLEX,T_REAL>::getReal(complex);
        buf_imag(x,y) = complexOps<T_COMPLEX,T_REAL>::getImag(complex);
    }
}

template<typename T_COMPLEX>
void core::image::splitComplex(const core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA >& buf_in, 
                           core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_real, 
                           core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_imag)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_in);
    
    // run kernel
    Kernel_splitComplex<T_COMPLEX, float><<<gridDim,blockDim>>>(buf_in, buf_real, buf_imag);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL>
__global__ void Kernel_joinComplex(const core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_real, 
                                   const core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_imag,
                                   core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_out.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_real(x,y), buf_imag(x,y));
    }
}

template<typename T_COMPLEX>
void core::image::joinComplex(const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_real, 
                          const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_imag, 
                          core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA >& buf_out)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_joinComplex<T_COMPLEX, float><<<gridDim,blockDim>>>(buf_real, buf_imag, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL>
__global__ void Kernel_magnitude(const core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA > buf_in,
                                 core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = sqrtf(complexOps<T_COMPLEX,T_REAL>::getReal(complex) * complexOps<T_COMPLEX,T_REAL>::getReal(complex) + 
                             complexOps<T_COMPLEX,T_REAL>::getImag(complex) * complexOps<T_COMPLEX,T_REAL>::getImag(complex));
    }
}

template<typename T_COMPLEX>
void core::image::magnitude(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_magnitude<T_COMPLEX, float><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL>
__global__ void Kernel_phase(const core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA > buf_in,
                             core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = atan2(complexOps<T_COMPLEX,T_REAL>::getImag(complex), complexOps<T_COMPLEX,T_REAL>::getReal(complex));
    }
}

template<typename T_COMPLEX>
void core::image::phase(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_phase<T_COMPLEX, float><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX, typename T_REAL>
__global__ void Kernel_convertToComplex(const core::Buffer2DView< T_REAL, core::TargetDeviceCUDA > buf_in,
                                        core::Buffer2DView< T_COMPLEX, core::TargetDeviceCUDA > buf_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_in.inBounds(x,y)) // is valid
    {
        buf_out(x,y) = complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_in(x,y), T_REAL(0.0f));
    }
}

template<typename T_COMPLEX>
void core::image::convertToComplex(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_out)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_out);
    
    // run kernel
    Kernel_convertToComplex<T_COMPLEX, float><<<gridDim,blockDim>>>(buf_in, buf_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

template<typename T_COMPLEX>
__global__ void Kernel_calculateCrossPowerSpectrum(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA> buf_fft1, 
                                                   const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA> buf_fft2, 
                                                   core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA> buf_fft_out)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;
    
    if(buf_fft_out.inBounds(x,y)) // is valid
    {
        const T_COMPLEX conj = complexOps<T_COMPLEX, float>::conjugate(buf_fft2(x,y));
        const T_COMPLEX num = complexOps<T_COMPLEX, float>::multiply(buf_fft1(x,y), conj);
        const float denom = complexOps<T_COMPLEX, float>::norm(complexOps<T_COMPLEX, float>::multiply(buf_fft1(x,y), buf_fft2(x,y)));
        buf_fft_out(x,y) = complexOps<T_COMPLEX, float>::divide(num, denom);
    }
}

template<typename T_COMPLEX>
void core::image::calculateCrossPowerSpectrum(const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft1, 
                                          const core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft2, 
                                          core::Buffer2DView<T_COMPLEX, core::TargetDeviceCUDA>& buf_fft_out)
{
    dim3 gridDim, blockDim;
    
    core::InitDimFromOutputImageOver(blockDim, gridDim, buf_fft_out);
    
    // run kernel
    Kernel_calculateCrossPowerSpectrum<T_COMPLEX><<<gridDim,blockDim>>>(buf_fft1, buf_fft2, buf_fft_out);
    
    // wait for it
    const cudaError err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        throw core::CUDAException(err, "Error launching the kernel");
    }
}

// R2C
template void core::image::transform<float, cufftComplex>(const core::Buffer1DView<float, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_out, bool forward);
template void core::image::transform<float, Eigen::Vector2f>(const core::Buffer1DView<float, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_out, bool forward);

template void core::image::transform<float, cufftComplex>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_out, bool forward);
template void core::image::transform<float, Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_out, bool forward);

// C2R
template void core::image::transform<cufftComplex, float>(const core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<float, core::TargetDeviceCUDA >& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, float>(const core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<float, core::TargetDeviceCUDA >& buf_out, bool forward);

template void core::image::transform<cufftComplex, float>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, float>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out, bool forward);

// C2C
template void core::image::transform<cufftComplex, cufftComplex>(const core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_out, bool forward);
template void core::image::transform<cufftComplex, Eigen::Vector2f>(const core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, cufftComplex>(const core::Buffer1DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_in, core::Buffer1DView<cufftComplex, core::TargetDeviceCUDA >& buf_out, bool forward);

template void core::image::transform<cufftComplex, cufftComplex>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_out, bool forward);
template void core::image::transform<cufftComplex, Eigen::Vector2f>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_out, bool forward);
template void core::image::transform<Eigen::Vector2f, cufftComplex>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_in, core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_out, bool forward);

// splitter
template void core::image::splitComplex<cufftComplex>(const core::Buffer2DView< cufftComplex, core::TargetDeviceCUDA >& buf_in, 
                                                  core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_real, 
                                                  core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_imag);
template void core::image::splitComplex<Eigen::Vector2f>(const core::Buffer2DView< Eigen::Vector2f, core::TargetDeviceCUDA >& buf_in, 
                                                     core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_real, 
                                                     core::Buffer2DView< float, core::TargetDeviceCUDA >& buf_imag);

// joiner
template void core::image::joinComplex<cufftComplex>(const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_real, 
                                                 const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_imag, 
                                                 core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA >& buf_out);
template void core::image::joinComplex<Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_real, 
                                                    const core::Buffer2DView<float, core::TargetDeviceCUDA >& buf_imag, 
                                                    core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA >& buf_out);

// magnitude
template void core::image::magnitude<cufftComplex>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_in, 
                                               core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);
template void core::image::magnitude<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_in, 
                                                  core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);

// phase
template void core::image::phase<cufftComplex>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_in, 
                                           core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);
template void core::image::phase<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_in, 
                                              core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_out);

// convert
template void core::image::convertToComplex<cufftComplex>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, 
                                                      core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_out);
template void core::image::convertToComplex<Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetDeviceCUDA>& buf_in, 
                                                         core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_out);

// cross power spectrum
template void core::image::calculateCrossPowerSpectrum<cufftComplex>(const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_fft1, 
                                                                 const core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_fft2, 
                                                                 core::Buffer2DView<cufftComplex, core::TargetDeviceCUDA>& buf_fft_out);
template void core::image::calculateCrossPowerSpectrum<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_fft1, 
                                                                    const core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_fft2, 
                                                                    core::Buffer2DView<Eigen::Vector2f, core::TargetDeviceCUDA>& buf_fft_out);
