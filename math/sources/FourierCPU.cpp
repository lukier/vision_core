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

#include <Platform.hpp>
#include <LaunchUtils.hpp>

#include <math/Fourier.hpp>

#include <fftw3.h>

template<typename T>
struct ToFFTWType { };

template<> struct ToFFTWType<float> { typedef float FFTWType; };
template<> struct ToFFTWType<double> { typedef double FFTWType; };
template<> struct ToFFTWType<std::complex<float>> { typedef fftwf_complex FFTWType; };
template<> struct ToFFTWType<std::complex<double>> { typedef fftw_complex FFTWType; };
template<> struct ToFFTWType<Eigen::Vector2f> { typedef fftwf_complex FFTWType; };
template<> struct ToFFTWType<Eigen::Vector2d> { typedef fftw_complex FFTWType; };

template<typename T_REAL>
struct plan_wrapper { };

template<>
struct plan_wrapper<double> : public core::math::PersistentFFT
{ 
    typedef fftw_plan FFTWPT; 
 
    plan_wrapper(const plan_wrapper&) = delete; // no copies
    plan_wrapper& operator=(const plan_wrapper& other) = delete; // no copies
    plan_wrapper(plan_wrapper&& other) noexcept : p(std::move(other.p)) { }
    plan_wrapper& operator=(plan_wrapper&& other) { p = std::move(other.p); return *this; }
    
    plan_wrapper() : p(nullptr) { }
    plan_wrapper(FFTWPT _p) : p(_p) { }
    ~plan_wrapper() { if(p != nullptr) { fftw_destroy_plan(p); } } 
    
    virtual void execute() { fftw_execute(p); }

    FFTWPT p;
};

template<>
struct plan_wrapper<float> : public core::math::PersistentFFT
{ 
    typedef fftwf_plan FFTWPT; 

    plan_wrapper(const plan_wrapper&) = delete; // no copies
    plan_wrapper& operator=(const plan_wrapper& other) = delete; // no copies
    plan_wrapper(plan_wrapper&& other) noexcept : p(std::move(other.p)) { }
    plan_wrapper& operator=(plan_wrapper&& other) { p = std::move(other.p); return *this; }
    
    plan_wrapper() : p(nullptr) { }
    plan_wrapper(FFTWPT _p) : p(_p) { }
    ~plan_wrapper() { if(p != nullptr) { fftwf_destroy_plan(p); } } 
    
    virtual void execute() { fftwf_execute(p); }

    FFTWPT p;
};

enum class FTT
{
    R2C,
    C2R,
    C2C
};

template<typename T1, typename T2>
struct TransformDirection { };

// R2C
template<typename T_REAL> struct TransformDirection<T_REAL, std::complex<T_REAL>> 
{ 
    static constexpr FTT TT = FTT::R2C; 
    typedef T_REAL FirstArgT; 
    typedef std::complex<T_REAL> SecondArgT; 
    typedef SecondArgT ComplexArgT;
};
template<typename T_REAL> struct TransformDirection<T_REAL, Eigen::Matrix<T_REAL,2,1>> 
{ 
    static constexpr FTT TT = FTT::R2C; 
    typedef T_REAL FirstArgT; 
    typedef Eigen::Matrix<T_REAL,2,1> SecondArgT; 
    typedef SecondArgT ComplexArgT;
};

// C2R
template<typename T_REAL> struct TransformDirection<std::complex<T_REAL>,T_REAL> 
{ 
    static constexpr FTT TT = FTT::C2R; 
    typedef std::complex<T_REAL> FirstArgT; 
    typedef T_REAL SecondArgT;     
    typedef FirstArgT ComplexArgT;
};
template<typename T_REAL> struct TransformDirection<Eigen::Matrix<T_REAL,2,1>,T_REAL> 
{ 
    static constexpr FTT TT = FTT::C2R; 
    typedef Eigen::Matrix<T_REAL,2,1> FirstArgT; 
    typedef T_REAL SecondArgT; 
    typedef FirstArgT ComplexArgT;
};

// C2C
template<typename T_REAL> struct TransformDirection<std::complex<T_REAL>,std::complex<T_REAL>> 
{ 
    static constexpr FTT TT = FTT::C2C; 
    typedef std::complex<T_REAL> FirstArgT; 
    typedef std::complex<T_REAL> SecondArgT; 
    typedef FirstArgT ComplexArgT;
};
template<typename T_REAL> struct TransformDirection<std::complex<T_REAL>,Eigen::Matrix<T_REAL,2,1>> 
{ 
    static constexpr FTT TT = FTT::C2C; 
    typedef std::complex<T_REAL> FirstArgT; 
    typedef Eigen::Matrix<T_REAL,2,1> SecondArgT; 
    typedef FirstArgT ComplexArgT;
};
template<typename T_REAL> struct TransformDirection<Eigen::Matrix<T_REAL,2,1>,Eigen::Matrix<T_REAL,2,1>> 
{ 
    static constexpr FTT TT = FTT::C2C; 
    typedef Eigen::Matrix<T_REAL,2,1> FirstArgT; 
    typedef Eigen::Matrix<T_REAL,2,1> SecondArgT; 
    typedef FirstArgT ComplexArgT;
};
template<typename T_REAL> struct TransformDirection<Eigen::Matrix<T_REAL,2,1>,std::complex<T_REAL>> 
{ 
    static constexpr FTT TT = FTT::C2C; 
    typedef Eigen::Matrix<T_REAL,2,1> FirstArgT; 
    typedef std::complex<T_REAL> SecondArgT; 
    typedef FirstArgT ComplexArgT;
};

template<FTT,typename T_COMPLEX, typename T_REAL = typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType>
struct PlanHelper { };

// R2C
template<typename T_COMPLEX>
struct PlanHelper<FTT::R2C, T_COMPLEX, float>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWRealT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftwf_plan_dft_r2c_1d(N, idata, odata, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWRealT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftwf_plan_dft_r2c_2d(W, H, idata, odata, FFTW_ESTIMATE));
    }
};

template<typename T_COMPLEX>
struct PlanHelper<FTT::R2C, T_COMPLEX, double>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWRealT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftw_plan_dft_r2c_1d(N, idata, odata, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWRealT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftw_plan_dft_r2c_2d(W, H, idata, odata, FFTW_ESTIMATE));
    }
};

// C2R
template<typename T_COMPLEX>
struct PlanHelper<FTT::C2R, T_COMPLEX, float>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWComplexT* idata, FFTWRealT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftwf_plan_dft_c2r_1d(N, idata, odata, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWComplexT* idata, FFTWRealT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftwf_plan_dft_c2r_2d(W, H, idata, odata, FFTW_ESTIMATE));
    }
};

template<typename T_COMPLEX>
struct PlanHelper<FTT::C2R, T_COMPLEX, double>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWComplexT* idata, FFTWRealT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftw_plan_dft_c2r_1d(N, idata, odata, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWComplexT* idata, FFTWRealT* odata, int dir = FFTW_FORWARD)
    {
        (void)dir;
        return plan_wrapper<T_REAL>(fftw_plan_dft_c2r_2d(W, H, idata, odata, FFTW_ESTIMATE));
    }
};

// C2C
template<typename T_COMPLEX>
struct PlanHelper<FTT::C2C, T_COMPLEX, float>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWComplexT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        return plan_wrapper<T_REAL>(fftwf_plan_dft_1d(N, idata, odata, dir, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWComplexT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        return plan_wrapper<T_REAL>(fftwf_plan_dft_2d(W, H, idata, odata, dir, FFTW_ESTIMATE));
    }
};

template<typename T_COMPLEX>
struct PlanHelper<FTT::C2C, T_COMPLEX, double>
{
    typedef typename core::math::internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    typedef typename ToFFTWType<T_REAL>::FFTWType FFTWRealT;
    typedef typename ToFFTWType<T_COMPLEX>::FFTWType FFTWComplexT;
    
    static inline plan_wrapper<T_REAL> makePlan1D(std::size_t N, FFTWComplexT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        return plan_wrapper<T_REAL>(fftw_plan_dft_1d(N, idata, odata, dir, FFTW_ESTIMATE));
    }
    
    static inline plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, FFTWComplexT* idata, FFTWComplexT* odata, int dir = FFTW_FORWARD)
    {
        return plan_wrapper<T_REAL>(fftw_plan_dft_2d(W, H, idata, odata, dir, FFTW_ESTIMATE));
    }
};

template<typename T_INPUT, typename T_OUTPUT>
struct ProperPlan
{
    static constexpr FTT TT = TransformDirection<T_INPUT,T_OUTPUT>::TT;
    typedef typename TransformDirection<T_INPUT,T_OUTPUT>::ComplexArgT ComplexArgT;
    typedef PlanHelper<TT, ComplexArgT> PlanHelperT;
    typedef typename core::math::internal::FFTTypeTraits<ComplexArgT>::BaseType T_REAL;
    
    static plan_wrapper<T_REAL> makePlan1D(std::size_t N, T_INPUT* buf_in, T_OUTPUT* buf_out, bool fwd)
    {
        typedef typename ToFFTWType<typename TransformDirection<T_INPUT,T_OUTPUT>::FirstArgT>::FFTWType FFTWFirstArgT;
        typedef typename ToFFTWType<typename TransformDirection<T_INPUT,T_OUTPUT>::SecondArgT>::FFTWType FFTWSecondArgT;
        
        return PlanHelperT::makePlan1D(N, reinterpret_cast<FFTWFirstArgT*>(buf_in), reinterpret_cast<FFTWSecondArgT*>(buf_out), fwd == true ? FFTW_FORWARD : FFTW_BACKWARD);
    }
    
    static plan_wrapper<T_REAL> makePlan2D(std::size_t W, std::size_t H, T_INPUT* buf_in, T_OUTPUT* buf_out, bool fwd)
    {
        typedef typename ToFFTWType<typename TransformDirection<T_INPUT,T_OUTPUT>::FirstArgT>::FFTWType FFTWFirstArgT;
        typedef typename ToFFTWType<typename TransformDirection<T_INPUT,T_OUTPUT>::SecondArgT>::FFTWType FFTWSecondArgT;
        
        return PlanHelperT::makePlan2D(W, H, reinterpret_cast<FFTWFirstArgT*>(buf_in), reinterpret_cast<FFTWSecondArgT*>(buf_out), fwd == true ? FFTW_FORWARD : FFTW_BACKWARD);
    }
};

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void core::math::fft(const core::Buffer1DView<T_INPUT, Target >& buf_in, 
                           core::Buffer1DView<T_OUTPUT, Target >& buf_out, bool forward)
{
    typedef ProperPlan<T_INPUT, T_OUTPUT> ProperPlanT;
    
    plan_wrapper<typename ProperPlanT::T_REAL> p = ProperPlanT::makePlan1D(buf_in.size(), const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
    
    p.execute();
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
void core::math::fft(const core::Buffer2DView<T_INPUT, Target>& buf_in, 
                           core::Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward)
{
    typedef ProperPlan<T_INPUT, T_OUTPUT> ProperPlanT;
    
    plan_wrapper<typename ProperPlanT::T_REAL> p = ProperPlanT::makePlan2D(buf_in.width(), buf_in.height(), const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward);
    
    p.execute();
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT(const core::Buffer1DView<T_INPUT, Target >& buf_in, 
                                                               core::Buffer1DView<T_OUTPUT, Target >& buf_out, bool forward)
{
    typedef ProperPlan<T_INPUT, T_OUTPUT> ProperPlanT;
    typedef plan_wrapper<typename ProperPlanT::T_REAL> PlanT;
    
    return std::unique_ptr<core::math::PersistentFFT>(new PlanT(ProperPlanT::makePlan1D(buf_in.size(), const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward)));
}

template<typename T_INPUT, typename T_OUTPUT, typename Target>
std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT(const core::Buffer2DView<T_INPUT, Target>& buf_in, 
                                                               core::Buffer2DView<T_OUTPUT, Target>& buf_out, bool forward)
{
    typedef ProperPlan<T_INPUT, T_OUTPUT> ProperPlanT;
    typedef plan_wrapper<typename ProperPlanT::T_REAL> PlanT;
    
    return std::unique_ptr<core::math::PersistentFFT>(new PlanT(ProperPlanT::makePlan2D(buf_in.width(), buf_in.height(), const_cast<T_INPUT*>(buf_in.ptr()), buf_out.ptr(), forward)));
}

template<typename T_COMPLEX, typename Target>
void core::math::splitComplex(const core::Buffer1DView< T_COMPLEX, Target >& buf_in, 
                              core::Buffer1DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real, 
                              core::Buffer1DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_in.size(), [&](std::size_t x)
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_real(x) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex);
        buf_imag(x) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex);
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::splitComplex(const core::Buffer2DView< T_COMPLEX, Target >& buf_in, 
                              core::Buffer2DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real, 
                              core::Buffer2DView< typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_in.width(), buf_in.height(), [&](std::size_t x, std::size_t y)
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_real(x,y) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex);
        buf_imag(x,y) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex);
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::joinComplex(const core::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real, 
                             const core::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag, 
                             core::Buffer1DView<T_COMPLEX, Target >& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.size(), [&](std::size_t x)
    {
        buf_out(x) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_real(x), buf_imag(x));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::joinComplex(const core::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_real, 
                             const core::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target >& buf_imag, 
                             core::Buffer2DView<T_COMPLEX, Target >& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        buf_out(x,y) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_real(x,y), buf_imag(x,y));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::magnitude(const core::Buffer1DView<T_COMPLEX, Target>& buf_in, 
                           core::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.size(), [&](std::size_t x)
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_out(x) = sqrtf(core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex) * core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex)
                         + core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex) * core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex)); 
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::magnitude(const core::Buffer2DView<T_COMPLEX, Target>& buf_in, 
                           core::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = sqrtf(core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex) * core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex)
                           + core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex) * core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex)); 
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::phase(const core::Buffer1DView<T_COMPLEX, Target>& buf_in, 
                       core::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.size(), [&](std::size_t x)
    {
        const T_COMPLEX& complex = buf_in(x);
        buf_out(x) = atan2(core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex), core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::phase(const core::Buffer2DView<T_COMPLEX, Target>& buf_in, 
                       core::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        const T_COMPLEX& complex = buf_in(x,y);
        buf_out(x,y) = atan2(core::math::internal::complexOps<T_COMPLEX,T_REAL>::getImag(complex), core::math::internal::complexOps<T_COMPLEX,T_REAL>::getReal(complex));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::convertToComplex(const core::Buffer1DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in, 
                                  core::Buffer1DView<T_COMPLEX, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.size(), [&](std::size_t x)
    {
        buf_out(x) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_in(x), T_REAL(0.0));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::convertToComplex(const core::Buffer2DView<typename internal::FFTTypeTraits<T_COMPLEX>::BaseType, Target>& buf_in, 
                                  core::Buffer2DView<T_COMPLEX, Target>& buf_out)
{
    typedef typename internal::FFTTypeTraits<T_COMPLEX>::BaseType T_REAL;
    
    core::launchParallelFor(buf_out.width(), buf_out.height(), [&](std::size_t x, std::size_t y)
    {
        buf_out(x,y) = core::math::internal::complexOps<T_COMPLEX,T_REAL>::makeComplex(buf_in(x,y), T_REAL(0.0));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::calculateCrossPowerSpectrum(const core::Buffer1DView<T_COMPLEX, Target>& buf_fft1, 
                                             const core::Buffer1DView<T_COMPLEX, Target>& buf_fft2, 
                                             core::Buffer1DView<T_COMPLEX, Target>& buf_fft_out)
{
    core::launchParallelFor(buf_fft_out.size(), [&](std::size_t x)
    {
        buf_fft_out(x) = core::math::internal::crossPowerSpectrum<T_COMPLEX>(buf_fft1(x),buf_fft2(x));
    });
}

template<typename T_COMPLEX, typename Target>
void core::math::calculateCrossPowerSpectrum(const core::Buffer2DView<T_COMPLEX, Target>& buf_fft1, 
                                             const core::Buffer2DView<T_COMPLEX, Target>& buf_fft2, 
                                             core::Buffer2DView<T_COMPLEX, Target>& buf_fft_out)
{
    core::launchParallelFor(buf_fft_out.width(), buf_fft_out.height(), [&](std::size_t x, std::size_t y)
    {
        buf_fft_out(x,y) = core::math::internal::crossPowerSpectrum<T_COMPLEX>(buf_fft1(x,y),buf_fft2(x,y));
    });
}

// R2C - 1D - float
template void core::math::fft<float, Eigen::Vector2f>(const core::Buffer1DView<float, core::TargetHost >& buf_in, 
                                                            core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<float, std::complex<float>>(const core::Buffer1DView<float, core::TargetHost >& buf_in, 
                                                                core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<float, Eigen::Vector2f>(const core::Buffer1DView<float, core::TargetHost >& buf_in, 
                                                                                                core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<float, std::complex<float>>(const core::Buffer1DView<float, core::TargetHost >& buf_in, 
                                                                                                    core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);

// R2C - 1D - double
template void core::math::fft<double, Eigen::Vector2d>(const core::Buffer1DView<double, core::TargetHost >& buf_in, 
                                                       core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<double, std::complex<double>>(const core::Buffer1DView<double, core::TargetHost >& buf_in, 
                                                           core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<double, Eigen::Vector2d>(const core::Buffer1DView<double, core::TargetHost >& buf_in, 
                                                                                                 core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<double, std::complex<double>>(const core::Buffer1DView<double, core::TargetHost >& buf_in, 
                                                                                                      core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);

// R2C - 2D - float
template void core::math::fft<float, Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                            core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<float, std::complex<float>>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                                core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<float, Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                                                                core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<float, std::complex<float>>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                                                                    core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);

// R2C - 2D - double
template void core::math::fft<double, Eigen::Vector2d>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                       core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<double, std::complex<double>>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                            core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<double, Eigen::Vector2d>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                                                                 core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<double, std::complex<double>>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                                                                      core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);

// C2R - 1D - float
template void core::math::fft<Eigen::Vector2f, float>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                            core::Buffer1DView<float, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<float>, float>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                                core::Buffer1DView<float, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, float>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                                                                core::Buffer1DView<float, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, float>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                                                                    core::Buffer1DView<float, core::TargetHost >& buf_out, bool forward);

// C2R - 1D - double
template void core::math::fft<Eigen::Vector2d, double>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                       core::Buffer1DView<double, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<double>, double>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                            core::Buffer1DView<double, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, double>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                                                                 core::Buffer1DView<double, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, double>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                                                                      core::Buffer1DView<double, core::TargetHost >& buf_out, bool forward);

// C2R - 2D - float
template void core::math::fft<Eigen::Vector2f, float>(const core::Buffer2DView<Eigen::Vector2f, 
                                                            core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<std::complex<float>, float>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                core::Buffer2DView<float, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, float>(const core::Buffer2DView<Eigen::Vector2f, 
                                                                                                core::TargetHost>& buf_in, core::Buffer2DView<float, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, float>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                                                    core::Buffer2DView<float, core::TargetHost>& buf_out, bool forward);

// C2R - 2D - double
template void core::math::fft<Eigen::Vector2d, double>(const core::Buffer2DView<Eigen::Vector2d, 
                                                       core::TargetHost>& buf_in, core::Buffer2DView<double, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<std::complex<double>, double>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                            core::Buffer2DView<double, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, double>(const core::Buffer2DView<Eigen::Vector2d, 
                                                                                                 core::TargetHost>& buf_in, core::Buffer2DView<double, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, double>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                                                                      core::Buffer2DView<double, core::TargetHost>& buf_out, bool forward);

// C2C - 1D - float
template void core::math::fft<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in,
                                                                      core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<Eigen::Vector2f, std::complex<float>>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                                          core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<float>, std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in,
                                                                              core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<float>, Eigen::Vector2f>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                                          core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in,
                                                                                                          core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, std::complex<float>>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                                                                              core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in,
                                                                                                                  core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, Eigen::Vector2f>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                                                                              core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out, bool forward);

// C2C - 1D - double
template void core::math::fft<Eigen::Vector2d, Eigen::Vector2d>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in,
                                                                core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<Eigen::Vector2d, std::complex<double>>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                                    core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<double>, std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in,
                                                                        core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);
template void core::math::fft<std::complex<double>, Eigen::Vector2d>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                                     core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, Eigen::Vector2d>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in,
                                                                                                          core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, std::complex<double>>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                                                                              core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in,
                                                                                                                  core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, Eigen::Vector2d>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                                                                               core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out, bool forward);

// C2C - 2D - float
template void core::math::fft<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                                      core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<Eigen::Vector2f, std::complex<float>>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                                          core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);

template void core::math::fft<std::complex<float>, std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                              core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<std::complex<float>, Eigen::Vector2f>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                          core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                                                                          core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2f, std::complex<float>>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                                                                              core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                                                                  core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<float>, Eigen::Vector2f>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                                                                              core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out, bool forward);

// C2C - 2D - double
template void core::math::fft<Eigen::Vector2d, Eigen::Vector2d>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                                core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<Eigen::Vector2d, std::complex<double>>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                                     core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);

template void core::math::fft<std::complex<double>, std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                                          core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);
template void core::math::fft<std::complex<double>, Eigen::Vector2d>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                                     core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, Eigen::Vector2d>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                                                                          core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<Eigen::Vector2d, std::complex<double>>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                                                                               core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);

template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                                                                                    core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out, bool forward);
template std::unique_ptr<core::math::PersistentFFT> core::math::makeFFT<std::complex<double>, Eigen::Vector2d>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                                                                               core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out, bool forward);

// splitter - 1D - float
template void core::math::splitComplex<Eigen::Vector2f>(const core::Buffer1DView< Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                        core::Buffer1DView< float, core::TargetHost >& buf_real, 
                                                        core::Buffer1DView< float, core::TargetHost >& buf_imag);
template void core::math::splitComplex<std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                            core::Buffer1DView< float, core::TargetHost >& buf_real, 
                                                            core::Buffer1DView< float, core::TargetHost >& buf_imag);

// splitter - 1D - double
template void core::math::splitComplex<Eigen::Vector2d>(const core::Buffer1DView< Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                        core::Buffer1DView< double, core::TargetHost >& buf_real, 
                                                        core::Buffer1DView< double, core::TargetHost >& buf_imag);
template void core::math::splitComplex<std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                            core::Buffer1DView< double, core::TargetHost >& buf_real, 
                                                            core::Buffer1DView< double, core::TargetHost >& buf_imag);

// splitter - 2D - float
template void core::math::splitComplex<Eigen::Vector2f>(const core::Buffer2DView< Eigen::Vector2f, core::TargetHost >& buf_in, 
                                                        core::Buffer2DView< float, core::TargetHost >& buf_real, 
                                                        core::Buffer2DView< float, core::TargetHost >& buf_imag);
template void core::math::splitComplex<std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost >& buf_in, 
                                                            core::Buffer2DView< float, core::TargetHost >& buf_real, 
                                                            core::Buffer2DView< float, core::TargetHost >& buf_imag);

// splitter - 2D - double
template void core::math::splitComplex<Eigen::Vector2d>(const core::Buffer2DView< Eigen::Vector2d, core::TargetHost >& buf_in, 
                                                        core::Buffer2DView< double, core::TargetHost >& buf_real, 
                                                        core::Buffer2DView< double, core::TargetHost >& buf_imag);
template void core::math::splitComplex<std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost >& buf_in, 
                                                             core::Buffer2DView< double, core::TargetHost >& buf_real, 
                                                             core::Buffer2DView< double, core::TargetHost >& buf_imag);

// joiner - 1D - float
template void core::math::joinComplex<Eigen::Vector2f>(const core::Buffer1DView<float, core::TargetHost >& buf_real, 
                                                       const core::Buffer1DView<float, core::TargetHost >& buf_imag, 
                                                       core::Buffer1DView<Eigen::Vector2f, core::TargetHost >& buf_out);
template void core::math::joinComplex<std::complex<float>>(const core::Buffer1DView<float, core::TargetHost >& buf_real, 
                                                           const core::Buffer1DView<float, core::TargetHost >& buf_imag, 
                                                           core::Buffer1DView<std::complex<float>, core::TargetHost >& buf_out);

// joiner - 1D - double
template void core::math::joinComplex<Eigen::Vector2d>(const core::Buffer1DView<double, core::TargetHost >& buf_real, 
                                                       const core::Buffer1DView<double, core::TargetHost >& buf_imag, 
                                                       core::Buffer1DView<Eigen::Vector2d, core::TargetHost >& buf_out);
template void core::math::joinComplex<std::complex<double>>(const core::Buffer1DView<double, core::TargetHost >& buf_real, 
                                                            const core::Buffer1DView<double, core::TargetHost >& buf_imag, 
                                                            core::Buffer1DView<std::complex<double>, core::TargetHost >& buf_out);

// joiner - 2D - float
template void core::math::joinComplex<Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetHost >& buf_real, 
                                                       const core::Buffer2DView<float, core::TargetHost >& buf_imag, 
                                                       core::Buffer2DView<Eigen::Vector2f, core::TargetHost >& buf_out);
template void core::math::joinComplex<std::complex<float>>(const core::Buffer2DView<float, core::TargetHost >& buf_real, 
                                                           const core::Buffer2DView<float, core::TargetHost >& buf_imag, 
                                                           core::Buffer2DView<std::complex<float>, core::TargetHost >& buf_out);

// joiner - 2D - double
template void core::math::joinComplex<Eigen::Vector2d>(const core::Buffer2DView<double, core::TargetHost >& buf_real, 
                                                       const core::Buffer2DView<double, core::TargetHost >& buf_imag, 
                                                       core::Buffer2DView<Eigen::Vector2d, core::TargetHost >& buf_out);
template void core::math::joinComplex<std::complex<double>>(const core::Buffer2DView<double, core::TargetHost >& buf_real, 
                                                            const core::Buffer2DView<double, core::TargetHost >& buf_imag, 
                                                            core::Buffer2DView<std::complex<double>, core::TargetHost >& buf_out);

// magnitude - 1D - float
template void core::math::magnitude<Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                     core::Buffer1DView<float, core::TargetHost>& buf_out);
template void core::math::magnitude<std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                         core::Buffer1DView<float, core::TargetHost>& buf_out);

// magnitude - 1D - double
template void core::math::magnitude<Eigen::Vector2d>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                     core::Buffer1DView<double, core::TargetHost>& buf_out);
template void core::math::magnitude<std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                          core::Buffer1DView<double, core::TargetHost>& buf_out);

// magnitude - 2D - float
template void core::math::magnitude<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                     core::Buffer2DView<float, core::TargetHost>& buf_out);
template void core::math::magnitude<std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                         core::Buffer2DView<float, core::TargetHost>& buf_out);

// magnitude - 2D - double
template void core::math::magnitude<Eigen::Vector2d>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                     core::Buffer2DView<double, core::TargetHost>& buf_out);
template void core::math::magnitude<std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                          core::Buffer2DView<double, core::TargetHost>& buf_out);

// phase - 1D - float
template void core::math::phase<Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                 core::Buffer1DView<float, core::TargetHost>& buf_out);
template void core::math::phase<std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                     core::Buffer1DView<float, core::TargetHost>& buf_out);

// phase - 1D - double
template void core::math::phase<Eigen::Vector2d>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                 core::Buffer1DView<double, core::TargetHost>& buf_out);
template void core::math::phase<std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                      core::Buffer1DView<double, core::TargetHost>& buf_out);

// phase - 2D - float
template void core::math::phase<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_in, 
                                                 core::Buffer2DView<float, core::TargetHost>& buf_out);
template void core::math::phase<std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_in, 
                                                     core::Buffer2DView<float, core::TargetHost>& buf_out);

// phase - 2D - double
template void core::math::phase<Eigen::Vector2d>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_in, 
                                                 core::Buffer2DView<double, core::TargetHost>& buf_out);
template void core::math::phase<std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_in, 
                                                      core::Buffer2DView<double, core::TargetHost>& buf_out);

// convert - 1D - float
template void core::math::convertToComplex<Eigen::Vector2f>(const core::Buffer1DView<float, core::TargetHost>& buf_in, 
                                                            core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_out);
template void core::math::convertToComplex<std::complex<float>>(const core::Buffer1DView<float, core::TargetHost>& buf_in, 
                                                                core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_out);

// convert - 2D - double
template void core::math::convertToComplex<Eigen::Vector2d>(const core::Buffer1DView<double, core::TargetHost>& buf_in, 
                                                            core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_out);
template void core::math::convertToComplex<std::complex<double>>(const core::Buffer1DView<double, core::TargetHost>& buf_in, 
                                                                 core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_out);

// convert - 2D - float
template void core::math::convertToComplex<Eigen::Vector2f>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                            core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_out);
template void core::math::convertToComplex<std::complex<float>>(const core::Buffer2DView<float, core::TargetHost>& buf_in, 
                                                                core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_out);

// convert - 2D - double
template void core::math::convertToComplex<Eigen::Vector2d>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                            core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_out);
template void core::math::convertToComplex<std::complex<double>>(const core::Buffer2DView<double, core::TargetHost>& buf_in, 
                                                                 core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_out);

// cross power spectrum - 1D - float
template void core::math::calculateCrossPowerSpectrum<Eigen::Vector2f>(const core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_fft1, 
                                                                       const core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_fft2, 
                                                                       core::Buffer1DView<Eigen::Vector2f, core::TargetHost>& buf_fft_out);
template void core::math::calculateCrossPowerSpectrum<std::complex<float>>(const core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_fft1, 
                                                                           const core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_fft2, 
                                                                           core::Buffer1DView<std::complex<float>, core::TargetHost>& buf_fft_out);
// cross power spectrum - 1D - double
template void core::math::calculateCrossPowerSpectrum<Eigen::Vector2d>(const core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_fft1, 
                                                                       const core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_fft2, 
                                                                       core::Buffer1DView<Eigen::Vector2d, core::TargetHost>& buf_fft_out);
template void core::math::calculateCrossPowerSpectrum<std::complex<double>>(const core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_fft1, 
                                                                            const core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_fft2, 
                                                                            core::Buffer1DView<std::complex<double>, core::TargetHost>& buf_fft_out);

// cross power spectrum - 2D - float
template void core::math::calculateCrossPowerSpectrum<Eigen::Vector2f>(const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_fft1, 
                                                                       const core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_fft2, 
                                                                       core::Buffer2DView<Eigen::Vector2f, core::TargetHost>& buf_fft_out);
template void core::math::calculateCrossPowerSpectrum<std::complex<float>>(const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_fft1, 
                                                                           const core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_fft2, 
                                                                           core::Buffer2DView<std::complex<float>, core::TargetHost>& buf_fft_out);

// cross power spectrum - 2D - double
template void core::math::calculateCrossPowerSpectrum<Eigen::Vector2d>(const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_fft1, 
                                                                       const core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_fft2, 
                                                                       core::Buffer2DView<Eigen::Vector2d, core::TargetHost>& buf_fft_out);
template void core::math::calculateCrossPowerSpectrum<std::complex<double>>(const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_fft1, 
                                                                            const core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_fft2, 
                                                                            core::Buffer2DView<std::complex<double>, core::TargetHost>& buf_fft_out);
