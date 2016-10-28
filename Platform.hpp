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
 * Generic macros, functions, traits etc.
 * ****************************************************************************
 */

#ifndef CORE_PLATFORM_HPP
#define CORE_PLATFORM_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <iosfwd>

#if defined(__CUDACC__) // NVCC
#define CORE_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define CORE_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define CORE_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for CORE_ALIGN macro for your host compiler!"
#endif

#ifdef __clang__
#define CORE_COMPILER_CLANG
#endif // __clang__

// ---------------------------------------------------------------------------
// Eigen is mandatory
// ---------------------------------------------------------------------------
#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

// ---------------------------------------------------------------------------
// CUDA Macros and CUDA types if CUDA not available
// ---------------------------------------------------------------------------
#include <CUDATypes.hpp>

// ---------------------------------------------------------------------------
// OpenCL
// ---------------------------------------------------------------------------
#ifdef CORE_HAVE_OPENCL
#include <CL/cl.hpp>
#endif // CORE_HAVE_OPENCL

// ---------------------------------------------------------------------------
// Ceres-Solver
// ---------------------------------------------------------------------------
#ifdef CORE_HAVE_CERES
#include <ceres/jet.h>
#endif // CORE_HAVE_CERES

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

namespace core
{
// ---------------------------------------------------------------------------
// Our internal type traits
// ---------------------------------------------------------------------------
    template<typename T> struct type_traits 
    { 
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned char>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned short>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned int>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<unsigned long>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long double>
    {
        typedef long double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char1>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar1>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char2>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar2>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char3>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar3>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<char4>
    {
        typedef char ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uchar4>
    {
        typedef unsigned char ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short1>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort1>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short2>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort2>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short3>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort3>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<short4>
    {
        typedef short ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ushort4>
    {
        typedef unsigned short ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int1>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint1>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int2>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint2>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int3>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint3>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<int4>
    {
        typedef int ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<uint4>
    {
        typedef unsigned int ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long1>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong1>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long2>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong2>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long3>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong3>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<long4>
    {
        typedef long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulong4>
    {
        typedef unsigned long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float1>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float2>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float3>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<float4>
    {
        typedef float ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong1>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong1>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong2>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong2>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong3>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong3>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<longlong4>
    {
        typedef long long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<ulonglong4>
    {
        typedef unsigned long long ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double1>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 1;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double2>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 2;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double3>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 3;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<> struct type_traits<double4>
    {
        typedef double ChannelType;
        static constexpr int ChannelCount = 4;
        static constexpr bool IsCUDAType = true;
        static constexpr bool IsEigenType = false;
    };
    
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    struct type_traits<Eigen::Array<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
    {
        typedef _Scalar ChannelType;
        static constexpr int ChannelCount = _Rows * _Cols;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = true;
    };
    
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    struct type_traits<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
    {
        typedef _Scalar ChannelType;
        static constexpr int ChannelCount = _Rows * _Cols;
        static constexpr bool IsCUDAType = false;
        static constexpr bool IsEigenType = true;
    };
    
// ---------------------------------------------------------------------------
// CUDA compatible numeric_limits
// ---------------------------------------------------------------------------

    template<typename _Tp>
    struct numeric_limits
    {
        /** The minimum finite value, or for floating types with denormalization, the minimum positive normalized value.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp min() noexcept;
        
        /** The maximum finite value.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp max() noexcept;

        /** A finite value x such that there is no other finite value y where y < x.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp lowest() noexcept;
        
        /** The @e machine @e epsilon:  the difference between 1 and the least value greater than 1 that is representable.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp epsilon() noexcept;
        
        /** The maximum rounding error measurement (see LIA-1).  */
        EIGEN_DEVICE_FUNC static constexpr _Tp round_error() noexcept;
        
        /** The representation of positive infinity, if @c has_infinity.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp infinity() noexcept;
        
        /** The representation of a quiet Not a Number,
         *if @c has_quiet_NaN. */
        EIGEN_DEVICE_FUNC static constexpr _Tp quiet_NaN() noexcept;
        
        /** The representation of a signaling Not a Number, if
         *@c has_signaling_NaN. */
        EIGEN_DEVICE_FUNC static constexpr _Tp signaling_NaN() noexcept;
        
        /** The minimum positive denormalized value.  For types where
         * @c has_denorm is false, this is the minimum positive normalized
         * value.  */
        EIGEN_DEVICE_FUNC static constexpr _Tp denorm_min() noexcept;
        
        static constexpr bool is_specialized = std::numeric_limits<_Tp>::is_specialized;
        static constexpr int digits = std::numeric_limits<_Tp>::digits;
        static constexpr int digits10 = std::numeric_limits<_Tp>::digits10;
        static constexpr int max_digits10 = std::numeric_limits<_Tp>::max_digits10;
        static constexpr bool is_signed = std::numeric_limits<_Tp>::is_signed;
        static constexpr bool is_integer = std::numeric_limits<_Tp>::is_integer;
        static constexpr bool is_exact = std::numeric_limits<_Tp>::is_exact;
        static constexpr int radix = std::numeric_limits<_Tp>::radix;
        static constexpr int min_exponent = std::numeric_limits<_Tp>::min_exponent;
        static constexpr int min_exponent10 = std::numeric_limits<_Tp>::min_exponent10;
        static constexpr int max_exponent = std::numeric_limits<_Tp>::max_exponent;
        static constexpr int max_exponent10 = std::numeric_limits<_Tp>::max_exponent10;
        static constexpr bool has_infinity = std::numeric_limits<_Tp>::has_infinity;
        static constexpr bool has_quiet_NaN = std::numeric_limits<_Tp>::has_quiet_NaN;
        static constexpr bool has_signaling_NaN = std::numeric_limits<_Tp>::has_signaling_NaN;
        static constexpr std::float_denorm_style has_denorm = std::numeric_limits<_Tp>::has_denorm;
        static constexpr bool has_denorm_loss = std::numeric_limits<_Tp>::has_denorm_loss;
        static constexpr bool is_iec559 = std::numeric_limits<_Tp>::is_iec559;
        static constexpr bool is_bounded = std::numeric_limits<_Tp>::is_bounded;
        static constexpr bool is_modulo = std::numeric_limits<_Tp>::is_modulo;
        static constexpr bool traps = std::numeric_limits<_Tp>::traps;
        static constexpr bool tinyness_before = std::numeric_limits<_Tp>::tinyness_before;
        static constexpr std::float_round_style round_style = std::numeric_limits<_Tp>::round_style;
    };
    
    template<typename _Tp> struct numeric_limits<const _Tp> : public numeric_limits<_Tp> { };
    template<typename _Tp> struct numeric_limits<volatile _Tp> : public numeric_limits<_Tp> { };
    template<typename _Tp> struct numeric_limits<const volatile _Tp> : public numeric_limits<_Tp> { };

    template<>
    struct numeric_limits<bool>
    {
        EIGEN_DEVICE_FUNC static constexpr bool min() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool max() noexcept { return true; }
        EIGEN_DEVICE_FUNC static constexpr bool lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr bool epsilon() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool round_error() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool infinity() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool quiet_NaN() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool signaling_NaN() noexcept { return false; }
        EIGEN_DEVICE_FUNC static constexpr bool denorm_min() noexcept { return false; }
    };
    
    /// numeric_limits<char> specialization.
    template<>
    struct numeric_limits<char>
    {
        EIGEN_DEVICE_FUNC static constexpr char min() noexcept { return CHAR_MIN; }
        EIGEN_DEVICE_FUNC static constexpr char max() noexcept { return CHAR_MAX; }
        EIGEN_DEVICE_FUNC static constexpr char lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr char epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr char round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr char infinity() noexcept { return char(); }
        EIGEN_DEVICE_FUNC static constexpr char quiet_NaN() noexcept { return char(); }
        EIGEN_DEVICE_FUNC static constexpr char signaling_NaN() noexcept { return char(); }
        EIGEN_DEVICE_FUNC static constexpr char denorm_min() noexcept { return static_cast<char>(0); }
    };
    
    /// numeric_limits<signed char> specialization.
    template<>
    struct numeric_limits<signed char>
    {
        EIGEN_DEVICE_FUNC static constexpr signed char min() noexcept { return -__SCHAR_MAX__ - 1; }
        EIGEN_DEVICE_FUNC static constexpr signed char max() noexcept { return __SCHAR_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr signed char lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr signed char epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr signed char round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr signed char infinity() noexcept { return static_cast<signed char>(0); }
        EIGEN_DEVICE_FUNC static constexpr signed char quiet_NaN() noexcept { return static_cast<signed char>(0); }
        EIGEN_DEVICE_FUNC static constexpr signed char signaling_NaN() noexcept { return static_cast<signed char>(0); }
        EIGEN_DEVICE_FUNC static constexpr signed char denorm_min() noexcept { return static_cast<signed char>(0); }
    };
    
    /// numeric_limits<unsigned char> specialization.
    template<>
    struct numeric_limits<unsigned char>
    {
        EIGEN_DEVICE_FUNC static constexpr unsigned char min() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned char max() noexcept { return __SCHAR_MAX__ * 2U + 1; }
        EIGEN_DEVICE_FUNC static constexpr unsigned char lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr unsigned char epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned char round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned char infinity() noexcept { return static_cast<unsigned char>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned char quiet_NaN() noexcept { return static_cast<unsigned char>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned char signaling_NaN() noexcept { return static_cast<unsigned char>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned char denorm_min() noexcept { return static_cast<unsigned char>(0); }
    };
    
    /// numeric_limits<wchar_t> specialization.
    template<>
    struct numeric_limits<wchar_t>
    {
        EIGEN_DEVICE_FUNC static constexpr wchar_t min() noexcept { return WCHAR_MIN; }
        EIGEN_DEVICE_FUNC static constexpr wchar_t max() noexcept { return WCHAR_MAX; }
        EIGEN_DEVICE_FUNC static constexpr wchar_t lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr wchar_t epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr wchar_t round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr wchar_t infinity() noexcept { return wchar_t(); }
        EIGEN_DEVICE_FUNC static constexpr wchar_t quiet_NaN() noexcept { return wchar_t(); }
        EIGEN_DEVICE_FUNC static constexpr wchar_t signaling_NaN() noexcept { return wchar_t(); }
        EIGEN_DEVICE_FUNC static constexpr wchar_t denorm_min() noexcept { return wchar_t(); }
    };
    
    /// numeric_limits<short> specialization.
    template<>
    struct numeric_limits<short>
    {
        EIGEN_DEVICE_FUNC static constexpr short min() noexcept { return -__SHRT_MAX__ - 1; }
        EIGEN_DEVICE_FUNC static constexpr short max() noexcept { return __SHRT_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr short lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr short epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr short round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr short infinity() noexcept { return short(); }
        EIGEN_DEVICE_FUNC static constexpr short quiet_NaN() noexcept { return short(); }
        EIGEN_DEVICE_FUNC static constexpr short signaling_NaN() noexcept { return short(); }
        EIGEN_DEVICE_FUNC static constexpr short denorm_min() noexcept { return short(); }
    };
    
    /// numeric_limits<unsigned short> specialization.
    template<>
    struct numeric_limits<unsigned short>
    {
        EIGEN_DEVICE_FUNC static constexpr unsigned short min() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned short max() noexcept { return __SHRT_MAX__ * 2U + 1; }
        EIGEN_DEVICE_FUNC static constexpr unsigned short lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr unsigned short epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned short round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned short infinity() noexcept { return static_cast<unsigned short>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned short quiet_NaN() noexcept { return static_cast<unsigned short>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned short signaling_NaN() noexcept { return static_cast<unsigned short>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned short denorm_min() noexcept { return static_cast<unsigned short>(0); }
    };
    
    /// numeric_limits<int> specialization.
    template<>
    struct numeric_limits<int>
    {
        EIGEN_DEVICE_FUNC static constexpr int min() noexcept { return -__INT_MAX__ - 1; }
        EIGEN_DEVICE_FUNC static constexpr int max() noexcept { return __INT_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr int lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr int epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr int round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr int infinity() noexcept { return static_cast<int>(0); }
        EIGEN_DEVICE_FUNC static constexpr int quiet_NaN() noexcept { return static_cast<int>(0); }
        EIGEN_DEVICE_FUNC static constexpr int signaling_NaN() noexcept { return static_cast<int>(0); }
        EIGEN_DEVICE_FUNC static constexpr int denorm_min() noexcept { return static_cast<int>(0); }
    };
    
    /// numeric_limits<unsigned int> specialization.
    template<>
    struct numeric_limits<unsigned int>
    {
        EIGEN_DEVICE_FUNC static constexpr unsigned int min() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned int max() noexcept { return __INT_MAX__ * 2U + 1; }
        EIGEN_DEVICE_FUNC static constexpr unsigned int lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr unsigned int epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned int round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned int infinity() noexcept { return static_cast<unsigned int>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned int quiet_NaN() noexcept { return static_cast<unsigned int>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned int signaling_NaN() noexcept { return static_cast<unsigned int>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned int denorm_min() noexcept { return static_cast<unsigned int>(0); }
    };
    
    /// numeric_limits<long> specialization.
    template<>
    struct numeric_limits<long>
    {
        EIGEN_DEVICE_FUNC static constexpr long min() noexcept { return -__LONG_MAX__ - 1; }
        EIGEN_DEVICE_FUNC static constexpr long max() noexcept { return __LONG_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr long lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr long epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr long round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr long infinity() noexcept { return static_cast<long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long quiet_NaN() noexcept { return static_cast<long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long signaling_NaN() noexcept { return static_cast<long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long denorm_min() noexcept { return static_cast<long>(0); }
    };
    
    /// numeric_limits<unsigned long> specialization.
    template<>
    struct numeric_limits<unsigned long>
    {
        EIGEN_DEVICE_FUNC static constexpr unsigned long min() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long max() noexcept { return __LONG_MAX__ * 2UL + 1; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long infinity() noexcept { return static_cast<unsigned long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long quiet_NaN() noexcept { return static_cast<unsigned long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long signaling_NaN() noexcept { return static_cast<unsigned long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long denorm_min() noexcept { return static_cast<unsigned long>(0); }
    };
    
    /// numeric_limits<long long> specialization.
    template<>
    struct numeric_limits<long long>
    {
        EIGEN_DEVICE_FUNC static constexpr long long min() noexcept { return -__LONG_LONG_MAX__ - 1; }
        EIGEN_DEVICE_FUNC static constexpr long long max() noexcept { return __LONG_LONG_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr long long lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr long long epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr long long round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr long long infinity() noexcept { return static_cast<long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long long quiet_NaN() noexcept { return static_cast<long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long long signaling_NaN() noexcept { return static_cast<long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr long long denorm_min() noexcept { return static_cast<long long>(0); }
    };
    
    /// numeric_limits<unsigned long long> specialization.
    template<>
    struct numeric_limits<unsigned long long>
    {
        EIGEN_DEVICE_FUNC static constexpr unsigned long long min() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long max() noexcept { return __LONG_LONG_MAX__ * 2ULL + 1; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long lowest() noexcept { return min(); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long epsilon() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long round_error() noexcept { return 0; }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long infinity() noexcept { return static_cast<unsigned long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long quiet_NaN() noexcept { return static_cast<unsigned long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long signaling_NaN() noexcept { return static_cast<unsigned long long>(0); }
        EIGEN_DEVICE_FUNC static constexpr unsigned long long denorm_min() noexcept { return static_cast<unsigned long long>(0); }
    }; 
    
    /// numeric_limits<float> specialization.
    template<>
    struct numeric_limits<float>
    {
        EIGEN_DEVICE_FUNC static constexpr float min() noexcept { return __FLT_MIN__; }
        EIGEN_DEVICE_FUNC static constexpr float max() noexcept { return __FLT_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr float lowest() noexcept { return -__FLT_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr float epsilon() noexcept { return __FLT_EPSILON__; }
        EIGEN_DEVICE_FUNC static constexpr float round_error() noexcept { return 0.5F; }        
        EIGEN_DEVICE_FUNC static constexpr float infinity() noexcept { return __builtin_huge_valf(); }
        EIGEN_DEVICE_FUNC static constexpr float quiet_NaN() noexcept { return __builtin_nanf(""); } // CUDART_NAN_F
        EIGEN_DEVICE_FUNC static constexpr float signaling_NaN() noexcept { return __builtin_nansf(""); }
        EIGEN_DEVICE_FUNC static constexpr float denorm_min() noexcept { return __FLT_DENORM_MIN__; }
    };
    
    /// numeric_limits<double> specialization.
    template<>
    struct numeric_limits<double>
    {
        EIGEN_DEVICE_FUNC static constexpr double min() noexcept { return __DBL_MIN__; }
        EIGEN_DEVICE_FUNC static constexpr double max() noexcept { return __DBL_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr double lowest() noexcept { return -__DBL_MAX__; }        
        EIGEN_DEVICE_FUNC static constexpr double epsilon() noexcept { return __DBL_EPSILON__; }
        EIGEN_DEVICE_FUNC static constexpr double round_error() noexcept { return 0.5; }
        EIGEN_DEVICE_FUNC static constexpr double infinity() noexcept { return __builtin_huge_val(); }
        EIGEN_DEVICE_FUNC static constexpr double quiet_NaN() noexcept { return __builtin_nan(""); }
        EIGEN_DEVICE_FUNC static constexpr double  signaling_NaN() noexcept { return __builtin_nans(""); }
        EIGEN_DEVICE_FUNC static constexpr double  denorm_min() noexcept { return __DBL_DENORM_MIN__; }
    };
    
    /// numeric_limits<long double> specialization.
    template<>
    struct numeric_limits<long double>
    {
        EIGEN_DEVICE_FUNC static constexpr long double  min() noexcept { return __LDBL_MIN__; }
        EIGEN_DEVICE_FUNC static constexpr long double  max() noexcept { return __LDBL_MAX__; }
        EIGEN_DEVICE_FUNC static constexpr long double lowest() noexcept { return -__LDBL_MAX__; }        
        EIGEN_DEVICE_FUNC static constexpr long double  epsilon() noexcept { return __LDBL_EPSILON__; }
        EIGEN_DEVICE_FUNC static constexpr long double  round_error() noexcept { return 0.5L; }
        EIGEN_DEVICE_FUNC static constexpr long double  infinity() noexcept { return __builtin_huge_vall(); }
        EIGEN_DEVICE_FUNC static constexpr long double  quiet_NaN() noexcept { return __builtin_nanl(""); }
        EIGEN_DEVICE_FUNC static constexpr long double  signaling_NaN() noexcept { return __builtin_nansl(""); }
        EIGEN_DEVICE_FUNC static constexpr long double  denorm_min() noexcept { return __LDBL_DENORM_MIN__; }
    };
#ifdef CORE_HAVE_CERES
    template<typename Scalar, int N>
    struct numeric_limits<ceres::Jet<Scalar, N>> 
    {
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> min() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::min()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> max() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::max()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> lowest() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::lowest()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> epsilon() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::epsilon()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> round_error() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::round_error()); }        
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> infinity() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::infinity()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> quiet_NaN() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::quiet_NaN()); } 
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> signaling_NaN() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::signaling_NaN()); }
        EIGEN_DEVICE_FUNC static constexpr ceres::Jet<Scalar, N> denorm_min() noexcept { return ceres::Jet<Scalar, N>(core::numeric_limits<float>::denorm_min()); }
    };
#endif // CORE_HAVE_CERES    
// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
    
    namespace internal
    {
        template<typename T>
        EIGEN_DEVICE_FUNC static inline bool isfinite_wrapper(const T& v)
        {
#ifndef CORE_CUDA_KERNEL_SPACE
            using std::isfinite;
#endif // CORE_CUDA_KERNEL_SPACE
            return isfinite(v);
        }
        
        template<typename T, int CC>
        struct cuda_type_funcs;
        
        template<typename TT> struct cuda_type_funcs<TT,1> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename core::type_traits<TT>::ChannelType val) { out.x = val; } 
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) { return isfinite_wrapper(val.x); }
            static inline void toStream(std::ostream& os, const TT& val) { os << "(" << val.x << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,2> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename core::type_traits<TT>::ChannelType val) { out.x = val; out.y = val; } 
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y); }
            static inline void toStream(std::ostream& os, const TT& val) { os << "(" << val.x << "," << val.y << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,3> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename core::type_traits<TT>::ChannelType val) { out.x = val; out.y = val; out.z = val; } 
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y) && isfinite_wrapper(val.z); }
            static inline void toStream(std::ostream& os, const TT& val) { os << "(" << val.x << "," << val.y << "," << val.z << ")"; }
        };
        
        template<typename TT> struct cuda_type_funcs<TT,4> 
        { 
            EIGEN_DEVICE_FUNC static inline void fill(TT& out, typename core::type_traits<TT>::ChannelType val) { out.x = val; out.y = val; out.z = val; out.w = val; } 
            EIGEN_DEVICE_FUNC static inline bool isvalid(const TT& val) { return isfinite_wrapper(val.x) && isfinite_wrapper(val.y) && isfinite_wrapper(val.z) && isfinite_wrapper(val.w); }
            static inline void toStream(std::ostream& os, const TT& val) { os << "(" << val.x << "," << val.y << "," << val.z << "," << val.w << ")"; }
        };
        
        template<typename T> struct eigen_isvalid;
        
        template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
        struct eigen_isvalid<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>>
        {
            typedef Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> MatrixT;
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const MatrixT& val)
            {
                for(int r = 0 ; r < _Rows ; ++r)
                {
                    for(int c = 0 ; c < _Cols ; ++c)
                    {
                        if(!isfinite_wrapper(val(r,c)))
                        {
                            return false;
                        }
                    }
                }
                
                return true;
            }
        };
        
        template<typename T, bool is_eig = core::type_traits<T>::IsEigenType , bool is_cud = core::type_traits<T>::IsCUDAType >
        struct type_dispatcher_helper;
        
        // for Eigen
        template<typename T>
        struct type_dispatcher_helper<T,true,false>
        {
            EIGEN_DEVICE_FUNC static inline T fill(typename core::type_traits<T>::ChannelType v)
            {
                return T::Constant(v);
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return eigen_isvalid<T>::isvalid(v);
            }
        };
        
        template<typename T>
        struct type_dispatcher_helper<T,false,true>
        {
            EIGEN_DEVICE_FUNC static inline T fill(typename core::type_traits<T>::ChannelType v)
            {
                T ret;
                cuda_type_funcs<T,core::type_traits<T>::ChannelCount>::fill(ret, v);
                return ret;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return cuda_type_funcs<T,core::type_traits<T>::ChannelCount>::isvalid(v);
            }
        };
        
        template<>
        struct type_dispatcher_helper<float,false,false>
        {
            EIGEN_DEVICE_FUNC static inline float fill(float v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const float& v)
            {
                return isfinite_wrapper(v);
            }
        };
        
        template<>
        struct type_dispatcher_helper<double,false,false>
        {
            EIGEN_DEVICE_FUNC static inline float fill(double v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const double& v)
            {
                return isfinite_wrapper(v);
            }
        };
        
        template<typename T>
        struct type_dispatcher_helper<T,false,false>
        {
            EIGEN_DEVICE_FUNC static inline T fill(T v)
            {
                return v;
            }
            
            EIGEN_DEVICE_FUNC static inline bool isvalid(const T& v)
            {
                return true;
            }
        };
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T zero()
    {
        return internal::type_dispatcher_helper<T>::fill(0.0);
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T setAll(typename core::type_traits<T>::ChannelType sv)
    {
        return internal::type_dispatcher_helper<T>::fill(sv);
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline T getInvalid()
    {
        typedef typename core::type_traits<T>::ChannelType ScalarT;
        return internal::type_dispatcher_helper<T>::fill(core::numeric_limits<ScalarT>::quiet_NaN());
    }
    
    template<typename T>
    EIGEN_DEVICE_FUNC static inline bool isvalid(T val)
    {
        return internal::type_dispatcher_helper<T>::isvalid(val);
    }
}

#define GENERATE_CUDATYPE_OPERATOR(XXXXX)  \
inline std::ostream& operator<<(std::ostream& os, const XXXXX& p) \
{ \
    core::internal::cuda_type_funcs<XXXXX,core::type_traits<XXXXX>::ChannelCount>::toStream(os,p); \
    return os; \
}

GENERATE_CUDATYPE_OPERATOR(char1)
GENERATE_CUDATYPE_OPERATOR(uchar1)
GENERATE_CUDATYPE_OPERATOR(char2)
GENERATE_CUDATYPE_OPERATOR(uchar2)
GENERATE_CUDATYPE_OPERATOR(char3)
GENERATE_CUDATYPE_OPERATOR(uchar3)
GENERATE_CUDATYPE_OPERATOR(char4)
GENERATE_CUDATYPE_OPERATOR(uchar4)
GENERATE_CUDATYPE_OPERATOR(short1)
GENERATE_CUDATYPE_OPERATOR(ushort1)
GENERATE_CUDATYPE_OPERATOR(short2)
GENERATE_CUDATYPE_OPERATOR(ushort2)
GENERATE_CUDATYPE_OPERATOR(short3)
GENERATE_CUDATYPE_OPERATOR(ushort3)
GENERATE_CUDATYPE_OPERATOR(short4)
GENERATE_CUDATYPE_OPERATOR(ushort4)
GENERATE_CUDATYPE_OPERATOR(int1)
GENERATE_CUDATYPE_OPERATOR(uint1)
GENERATE_CUDATYPE_OPERATOR(int2)
GENERATE_CUDATYPE_OPERATOR(uint2)
GENERATE_CUDATYPE_OPERATOR(int3)
GENERATE_CUDATYPE_OPERATOR(uint3)
GENERATE_CUDATYPE_OPERATOR(int4)
GENERATE_CUDATYPE_OPERATOR(uint4)
GENERATE_CUDATYPE_OPERATOR(long1)
GENERATE_CUDATYPE_OPERATOR(ulong1)
GENERATE_CUDATYPE_OPERATOR(long2)
GENERATE_CUDATYPE_OPERATOR(ulong2)
GENERATE_CUDATYPE_OPERATOR(long3)
GENERATE_CUDATYPE_OPERATOR(ulong3)
GENERATE_CUDATYPE_OPERATOR(long4)
GENERATE_CUDATYPE_OPERATOR(ulong4)
GENERATE_CUDATYPE_OPERATOR(float1)
GENERATE_CUDATYPE_OPERATOR(float2)
GENERATE_CUDATYPE_OPERATOR(float3)
GENERATE_CUDATYPE_OPERATOR(float4)
GENERATE_CUDATYPE_OPERATOR(longlong1)
GENERATE_CUDATYPE_OPERATOR(ulonglong1)
GENERATE_CUDATYPE_OPERATOR(longlong2)
GENERATE_CUDATYPE_OPERATOR(ulonglong2)
GENERATE_CUDATYPE_OPERATOR(longlong3)
GENERATE_CUDATYPE_OPERATOR(ulonglong3)
GENERATE_CUDATYPE_OPERATOR(longlong4)
GENERATE_CUDATYPE_OPERATOR(ulonglong4)
GENERATE_CUDATYPE_OPERATOR(double1)
GENERATE_CUDATYPE_OPERATOR(double2)
GENERATE_CUDATYPE_OPERATOR(double3)
GENERATE_CUDATYPE_OPERATOR(double4)

// ---------------------------------------------------------------------------
// NOTE Why include Cereal here?
// ---------------------------------------------------------------------------
#if defined(CORE_HAVE_CEREAL) && !defined(CORE_CUDA_COMPILER)
#define CORE_ENABLE_CEREAL
#include <cereal/cereal.hpp>
#include <EigenSerializers.hpp>
#endif // CORE_ENABLE_CEREAL

inline EIGEN_DEVICE_FUNC float lerp(unsigned char a, unsigned char b, float t)
{
    return (float)a + t*((float)b-(float)a);
}

inline EIGEN_DEVICE_FUNC float2 lerp(uchar2 a, uchar2 b, float t)
{
    return make_float2(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y)
    );
}

inline EIGEN_DEVICE_FUNC float3 lerp(uchar3 a, uchar3 b, float t)
{
    return make_float3(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y),
        a.z + t*(b.z-a.z)
    );
}

inline EIGEN_DEVICE_FUNC float4 lerp(uchar4 a, uchar4 b, float t)
{
    return make_float4(
        a.x + t*(b.x-a.x),
        a.y + t*(b.y-a.y),
        a.z + t*(b.z-a.z),
        a.w + t*(b.w-a.w)
    );
}

inline EIGEN_DEVICE_FUNC Eigen::Vector2f lerp(Eigen::Vector2f a, Eigen::Vector2f b, float t)
{
    return Eigen::Vector2f(a(0) + t*(b(0) - a(0)), a(1) + t*(b(1) - a(1)));
}

inline EIGEN_DEVICE_FUNC Eigen::Vector3f lerp(Eigen::Vector3f a, Eigen::Vector3f b, float t)
{
    return Eigen::Vector3f(a(0) + t*(b(0) - a(0)), a(1) + t*(b(1) - a(1)), a(2) + t*(b(2) - a(2)));
}

inline EIGEN_DEVICE_FUNC Eigen::Vector4f lerp(Eigen::Vector4f a, Eigen::Vector4f b, float t)
{
    return Eigen::Vector4f(a(0) + t*(b(0) - a(0)), a(1) + t*(b(1) - a(1)), a(2) + t*(b(2) - a(2)), a(4) + t*(b(4) - a(4)));
}

namespace core
{
// A traits class to make it easier to work with mixed auto / numeric diff.
template<typename T>
struct ADTraits 
{
    typedef T Scalar;
    static constexpr std::size_t DerDimension = 0;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return true; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const Scalar& t)  { return t; }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, Scalar* t)  { *t = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const Scalar& t, std::size_t n) { return 0.0f; }
};

template<typename ADT>
struct ADTraits<Eigen::AutoDiffScalar<ADT>> 
{
    typedef typename Eigen::AutoDiffScalar<ADT>::Scalar Scalar;
    static constexpr std::size_t DerDimension = Eigen::AutoDiffScalar<ADT>::DerType::RowsAtCompileTime;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return false; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const Eigen::AutoDiffScalar<ADT>& t) { return t.value(); }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, Eigen::AutoDiffScalar<ADT>* t)  { t->value() = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const Eigen::AutoDiffScalar<ADT>& t, std::size_t n) { return t.derivatives()(n); }
};
#ifdef CORE_HAVE_CERES
template<typename T, int N>
struct ADTraits<ceres::Jet<T, N> > 
{
    typedef T Scalar;
    static constexpr std::size_t DerDimension = N;
    EIGEN_DEVICE_FUNC inline static constexpr bool isScalar() { return false; }
    EIGEN_DEVICE_FUNC inline static Scalar getScalar(const ceres::Jet<T, N>& t) { return t.a; }
    EIGEN_DEVICE_FUNC static void setScalar(const Scalar& scalar, ceres::Jet<T, N>* t)  { t->a = scalar; }
    EIGEN_DEVICE_FUNC inline static Scalar getDerivative(const ceres::Jet<T,N>& t, std::size_t n) { return t.v(n); }
};
#endif // CORE_HAVE_CERES

// Chain rule
template<typename FunctionType, int kNumArgs, typename ArgumentType>
struct Chain 
{
    EIGEN_DEVICE_FUNC inline static ArgumentType Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], const ArgumentType x[kNumArgs]) 
    {
        // In the default case of scalars, there's nothing to do since there are no
        // derivatives to propagate.
        (void) dfdx;  // Ignored.
        (void) x;  // Ignored.
        return f;
    }
};

template<typename FunctionType, int kNumArgs, typename ADT>
struct Chain<FunctionType, kNumArgs, Eigen::AutoDiffScalar<ADT> > 
{
    EIGEN_DEVICE_FUNC inline static Eigen::AutoDiffScalar<ADT> Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], const Eigen::AutoDiffScalar<ADT> x[kNumArgs]) 
    {
        // x is itself a function of another variable ("z"); what this function
        // needs to return is "f", but with the derivative with respect to z
        // attached to the jet. So combine the derivative part of x's jets to form
        // a Jacobian matrix between x and z (i.e. dx/dz).
        Eigen::Matrix<typename Eigen::AutoDiffScalar<ADT>::Scalar, kNumArgs, ADT::RowsAtCompileTime> dxdz;
        for (int i = 0; i < kNumArgs; ++i) 
        {
            dxdz.row(i) = x[i].derivatives().transpose();
        }
        
        // Map the input gradient dfdx into an Eigen row vector.
        Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);
        
        // Now apply the chain rule to obtain df/dz. Combine the derivative with
        // the scalar part to obtain f with full derivative information.
        Eigen::AutoDiffScalar<ADT> jet_f;
        jet_f.value() = f;
        jet_f.derivatives() = vector_dfdx.template cast<typename Eigen::AutoDiffScalar<ADT>::Scalar>() * dxdz;  // Also known as dfdz.
        return jet_f;
    }
};
#ifdef CORE_HAVE_CERES
template<typename FunctionType, int kNumArgs, typename T, int N>
struct Chain<FunctionType, kNumArgs, ceres::Jet<T, N> > 
{
    EIGEN_DEVICE_FUNC inline static ceres::Jet<T, N> Rule(const FunctionType &f, const FunctionType dfdx[kNumArgs], const ceres::Jet<T, N> x[kNumArgs]) 
    {
        // x is itself a function of another variable ("z"); what this function
        // needs to return is "f", but with the derivative with respect to z
        // attached to the jet. So combine the derivative part of x's jets to form
        // a Jacobian matrix between x and z (i.e. dx/dz).
        Eigen::Matrix<T, kNumArgs, N> dxdz;
        for (int i = 0; i < kNumArgs; ++i) 
        {
            dxdz.row(i) = x[i].v.transpose();
        }
        
        // Map the input gradient dfdx into an Eigen row vector.
        Eigen::Map<const Eigen::Matrix<FunctionType, 1, kNumArgs> >
        vector_dfdx(dfdx, 1, kNumArgs);
        
        // Now apply the chain rule to obtain df/dz. Combine the derivative with
        // the scalar part to obtain f with full derivative information.
        ceres::Jet<T, N> jet_f;
        jet_f.a = f;
        jet_f.v = vector_dfdx.template cast<T>() * dxdz;  // Also known as dfdz.
        return jet_f;
    }
};
#endif // CORE_HAVE_CERES
}

#endif // CORE_PLATFORM_HPP
