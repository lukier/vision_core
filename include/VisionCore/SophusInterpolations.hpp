/**
 * 
 * Core Libraries.
 * Sophus Interpolations.
 * 
 * Copyright (c) Robert Lukierski 2016. All rights reserved.
 * Author: Robert Lukierski.
 * 
 */


#ifndef VISIONCORE_SOPHUS_INTERPOLATIONS_HPP
#define VISIONCORE_SOPHUS_INTERPOLATIONS_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/rxso3.hpp>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

namespace Sophus
{

// SO2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO2Group<T> interpolateLinear(const Sophus::SO2Group<T>& t0, const Sophus::SO2Group<T>& t1, T ratio)
{
    return Sophus::SO2Group<T>(t0 * Sophus::SO2Group<T>::exp(ratio * ( t0.inverse() * t1 ).log() ));
}
    
// SE2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE2Group<T> interpolateLinear(const Sophus::SE2Group<T>& t0, const Sophus::SE2Group<T>& t1, T ratio)
{
    return Sophus::SE2Group<T>(t0.so2() * Sophus::SO2Group<T>::exp(ratio * ( t0.so2().inverse() * t1.so2() ).log() ), t0.translation() + ratio * (t1.translation() - t0.translation()));
}    

// SO3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO3Group<T> interpolateLinear(const Sophus::SO3Group<T>& t0, const Sophus::SO3Group<T>& t1, T ratio)
{
    return Sophus::SO3Group<T>(t0 * Sophus::SO3Group<T>::exp(ratio * ( t0.inverse() * t1 ).log() ));
}

// SE3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE3Group<T> interpolateLinear(const Sophus::SE3Group<T>& t0, const Sophus::SE3Group<T>& t1, T ratio)
{
    return Sophus::SE3Group<T>(t0.so3() * Sophus::SO3Group<T>::exp(ratio * ( t0.so3().inverse() * t1.so3() ).log() ), t0.translation() + ratio * (t1.translation() - t0.translation()));
}
    
template<typename T>
struct B4SplineGenerator 
{ 

    EIGEN_DEVICE_FUNC static inline void get(const T& u, T& b1, T& b2, T& b3)
    {
        const T u2 = u*u;
        const T u3 = u*u*u;
        
        b1 = (u3 - T(3.0) * u2 + T(3.0) * u + T(5.0)) / T(6.0);
        b2 = (T(-2.0) * u3 + T(3.0) * u2 + T(3.0) * u + T(1.0)) / T(6.0);
        b3 = u3 / T(6.0);
    }

#if 0
    EIGEN_DEVICE_FUNC static inline void get(const T& u, T& b1, T& b2, T& b3)
    {
        Eigen::Matrix<T,4,1> v(T(1.0),u,u*u,u*u*u);
        Eigen::Matrix<T,4,4> m;
        m << T(6.0) , T(0.0) , T(0.0) , T(0.0),
             T(5.0) , T(3.0) , T(-3.0), T(1.0),
             T(1.0) , T(3.0) , T(3.0), T(-2.0),
             T(0.0) , T(0.0) , T(0.0) , T(1.0);
        
        const Eigen::Matrix<T,4,1> uvec = (m * v) / T(6.0);
        b1 = uvec(1);
        b2 = uvec(2);
        b3 = uvec(3);
    }
#endif
};

// B4-Spline SO2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO2Group<T> interpolateB4Spline(const Sophus::SO2Group<T>& tm1, const Sophus::SO2Group<T>& t0, const Sophus::SO2Group<T>& t1, const Sophus::SO2Group<T>& t2, const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    return tm1 * Sophus::SO2Group<T>::exp(b1 * ( tm1.inverse() * t0 ).log() + 
                                          b2 * ( t0.inverse()  * t1 ).log() +
                                          b3 * ( t1.inverse()  * t2 ).log() );
}

// B4-Spline SE2
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE2Group<T> interpolateB4Spline(const Sophus::SE2Group<T>& tm1, const Sophus::SE2Group<T>& t0, const Sophus::SE2Group<T>& t1, const Sophus::SE2Group<T>& t2, const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    const Sophus::SO2Group<T> rot_final = tm1.so2() * Sophus::SO2Group<T>::exp(b1 * ( tm1.so2().inverse() * t0.so2() ).log() + 
                                                                               b2 * ( t0.so2().inverse()  * t1.so2() ).log() +
                                                                               b3 * ( t1.so2().inverse()  * t2.so2() ).log() );
    
    const typename Sophus::SE2Group<T>::Point tr_final = tm1.translation() + (b1 * (t0.translation() - tm1.translation())) + 
                                                                             (b2 * (t1.translation() - t0.translation())) + 
                                                                             (b3 * (t2.translation() - t1.translation()));
    
    return Sophus::SE2Group<T>(rot_final, tr_final);
}

// B4-Spline SO3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SO3Group<T> interpolateB4Spline(const Sophus::SO3Group<T>& tm1, const Sophus::SO3Group<T>& t0, const Sophus::SO3Group<T>& t1, const Sophus::SO3Group<T>& t2, const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    return tm1 * Sophus::SO3Group<T>::exp(b1 * ( tm1.inverse() * t0 ).log() + 
                                          b2 * ( t0.inverse()  * t1 ).log() +
                                          b3 * ( t1.inverse()  * t2 ).log() );
}

// B4-Spline SE3
template<typename T>
EIGEN_DEVICE_FUNC static inline Sophus::SE3Group<T> interpolateB4Spline(const Sophus::SE3Group<T>& tm1, const Sophus::SE3Group<T>& t0, const Sophus::SE3Group<T>& t1, const Sophus::SE3Group<T>& t2, const T& u)
{
    T b1, b2, b3;
    
    B4SplineGenerator<T>::get(u,b1,b2,b3);
    
    const Sophus::SO3Group<T> rot_final = tm1.so3() * Sophus::SO3Group<T>::exp(b1 * ( tm1.so3().inverse() * t0.so3() ).log() + 
                                                                               b2 * ( t0.so3().inverse()  * t1.so3() ).log() +
                                                                               b3 * ( t1.so3().inverse()  * t2.so3() ).log() );
    
    const typename Sophus::SE3Group<T>::Point tr_final = tm1.translation() + (b1 * (t0.translation() - tm1.translation())) + 
                                                                             (b2 * (t1.translation() - t0.translation())) + 
                                                                             (b3 * (t2.translation() - t1.translation()));
    
    return Sophus::SE3Group<T>(rot_final, tr_final);
}
    
}

#endif // SOPHUS_INTERPOLATIONS_HPP
