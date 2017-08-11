/**
 * ****************************************************************************
 * Copyright (c) 2017, Robert Lukierski.
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
 * Bits and bobs missing from Eigen.
 * ****************************************************************************
 */

#ifndef VISIONCORE_EIGEN_MISSING_BITS_HPP
#define VISIONCORE_EIGEN_MISSING_BITS_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/AutoDiff>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/rxso3.hpp>
#include <sophus/sim3.hpp>

namespace Eigen 
{
  
namespace numext
{
  
template<typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T atan2(const T& y, const T &x) {
    EIGEN_USING_STD_MATH(atan2);
    return atan2(y,x);
}

#ifdef __CUDACC__
template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
float atan2(const float& y, const float &x) { return ::atan2f(y,x); }

template<> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
double atan2(const double& y, const double &x) { return ::atan2(y,x); }
#endif
  
}

template<typename DerType> 
inline const Eigen::AutoDiffScalar<EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename Eigen::internal::remove_all<DerType>::type, typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar, product)> 
atan(const Eigen::AutoDiffScalar<DerType>& x) 
{ 
  using namespace Eigen; 
  EIGEN_UNUSED typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; 
  using numext::atan;
  return Eigen::MakeAutoDiffScalar(atan(x.value()),x.derivatives() * ( Scalar(1) / (Scalar(1) + x.value() * x.value()) ));
}

}

#endif // VISIONCORE_EIGEN_MISSING_BITS_HPP
