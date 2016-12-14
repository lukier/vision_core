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
 * Divergence functions.
 * ****************************************************************************
 */

#ifndef CORE_MATH_DIVERGENCE_HPP
#define CORE_MATH_DIVERGENCE_HPP

#include <Platform.hpp>

#include <buffers/Buffer2D.hpp>

namespace core 
{
    
namespace math
{

EIGEN_DEVICE_FUNC inline float projectUnitBall(float val, float maxrad = 1.0f)
{
    return val / max(1.0f, fabs(val) / maxrad );
}

EIGEN_DEVICE_FUNC inline Eigen::Vector2f projectUnitBall(const Eigen::Vector2f& val, float maxrad = 1.0f)
{
    return val / max(1.0f, sqrt(val(0) * val(0) + val(1) * val(1)) / maxrad );
}

EIGEN_DEVICE_FUNC inline Eigen::Vector3f projectUnitBall(const Eigen::Vector3f& val, float maxrad = 1.0f)
{
    return val / max(1.0f, sqrt(val(0) * val(0) + val(1) * val(1) + val(2) * val(2))  / maxrad );
}

EIGEN_DEVICE_FUNC inline Eigen::Vector4f projectUnitBall(const Eigen::Vector4f& val, float maxrad = 1.0f)
{
    return val / max(1.0f, sqrt(val(0) * val(0) + val(1) * val(1) + val(2) * val(2) + val(3) * val(3))  / maxrad );
}

template<template<typename> class Target>
EIGEN_DEVICE_FUNC inline Eigen::Vector2f gradUFwd(const Buffer2DView<float, Target>& imgu, float u, size_t x, size_t y)
{
    Eigen::Vector2f du(0.0f, 0.0f);
    if(x < imgu.width() - 1) du(0) = imgu(x+1,y) - u;
    if(y < imgu.height() - 1) du(1) = imgu(x,y+1) - u;
    return du;
}

template<template<typename> class Target>
EIGEN_DEVICE_FUNC inline float divA(const Buffer2DView<Eigen::Vector2f, Target>& A, int x, int y)
{
    const Eigen::Vector2f& p = A(x,y);
    float divA = p(0) + p(1);
    if(x > 0) divA -= A(x - 1, y)(0);
    if(y > 0) divA -= A(x, y - 1)(1);
    return divA;
}

template<template<typename> class Target>
EIGEN_DEVICE_FUNC inline Eigen::Vector4f TGVEpsilon(const Buffer2DView<Eigen::Vector2f, Target>& imgA, size_t x, size_t y)
{
    const Eigen::Vector2f& A = imgA(x,y);

    float dy_v0 = 0;
    float dx_v0 = 0;
    float dx_v1 = 0;
    float dy_v1 = 0;

    if(x < imgA.width() - 1) 
    {
        const Eigen::Vector2f& Apx = imgA(x + 1, y);
        dx_v0 = Apx(0) - A(0);
        dx_v1 = Apx(1) - A(1);
    }

    if(y < imgA.height() - 1) 
    {
        const Eigen::Vector2f& Apy = imgA(x, y + 1);
        dy_v0 = Apy(0) - A(0);
        dy_v1 = Apy(1) - A(1);
    }

    return Eigen::Vector4f(dx_v0, dy_v1, (dy_v0+dx_v1)/2.0f, (dy_v0+dx_v1)/2.0f );
}

template<template<typename> class Target>
EIGEN_DEVICE_FUNC inline Eigen::Vector2f TGVDivA(const Buffer2DView<Eigen::Vector4f, Target>& A, int x, int y)
{
    const Eigen::Vector4f& p = A(x,y);
    Eigen::Vector2f divA(p(0) + p(2), p(2) + p(1));

    if(0 < x)
    {
        divA(0) -= A(x - 1, y)(0);
        divA(1) -= A(x - 1, y)(2);
    }

    if(0 < y)
    {
        divA(0) -= A(x, y - 1)(2);
        divA(1) -= A(x, y - 1)(1);
    }

    return divA;
}

}

}

#endif // CORE_MATH_DIVERGENCE_HPP
