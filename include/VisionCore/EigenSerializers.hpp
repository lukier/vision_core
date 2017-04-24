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
 * Serialization of Eigen into Cereal.
 * ****************************************************************************
 */

#ifndef VISIONCORE_EIGEN_SERIALIZERS_HPP
#define VISIONCORE_EIGEN_SERIALIZERS_HPP

#include <VisionCore/Platform.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/so2.hpp>
#include <sophus/so3.hpp>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <sophus/rxso3.hpp>
#include <sophus/sim3.hpp>

#ifdef VISIONCORE_HAVE_CEREAL
namespace Eigen
{
 
/**
 * Matrix
 */ 
template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive & archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & m, std::uint32_t const version)
{
    for(int r = 0 ; r < m.rows() ; ++r)
    {
        for(int c = 0 ; c < m.cols() ; ++c)
        {
            archive(m(r,c));
        }
    }
}

template<typename Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive & archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> const & m, std::uint32_t const version)
{
    for(int r = 0 ; r < m.rows() ; ++r)
    {
        for(int c = 0 ; c < m.cols() ; ++c)
        {
            archive(m(r,c));
        }
    }
}

/**
 * Quaternion
 */ 
template<typename Archive, typename _Scalar, int _Options>
void load(Archive & archive, Eigen::Quaternion<_Scalar, _Options> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("X", m.x()));
    archive(cereal::make_nvp("Y", m.y()));
    archive(cereal::make_nvp("Z", m.z()));
    archive(cereal::make_nvp("W", m.w()));
}

template<typename Archive, typename _Scalar, int _Options>
void save(Archive & archive, Eigen::Quaternion<_Scalar, _Options> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("X", m.x()));
    archive(cereal::make_nvp("Y", m.y()));
    archive(cereal::make_nvp("Z", m.z()));
    archive(cereal::make_nvp("W", m.w()));
}

/**
 * AngleAxis
 */ 
template<typename Archive, typename _Scalar>
void load(Archive & archive, Eigen::AngleAxis<_Scalar> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
    archive(cereal::make_nvp("Axis", m.axis()));
}

template<typename Archive, typename _Scalar>
void save(Archive & archive, Eigen::AngleAxis<_Scalar> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
    archive(cereal::make_nvp("Axis", m.axis()));
}

/**
 * Rotation2D
 */ 
template<typename Archive, typename _Scalar>
void load(Archive & archive, Eigen::Rotation2D<_Scalar> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
}

template<typename Archive, typename _Scalar>
void save(Archive & archive, Eigen::Rotation2D<_Scalar> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Angle", m.angle()));
}

/**
 * AutoDiffScalar
 */ 
template<typename Archive, typename ADT>
void load(Archive & archive, Eigen::AutoDiffScalar<ADT> & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Value", m.value()));
    archive(cereal::make_nvp("Deriviatives", m.derivatives()));
}

template<typename Archive, typename ADT>
void save(Archive & archive, Eigen::AutoDiffScalar<ADT> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Value", m.value()));
    archive(cereal::make_nvp("Deriviatives", m.derivatives()));
}
    
}

namespace Sophus
{

/**
 * SO2
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SO2GroupBase<Derived>& m, std::uint32_t const version)
{
    typename SO2GroupBase<Derived>::Point cplx;
    archive(cplx);
    m.setComplex(cplx);
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, SO2GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(m.unit_complex());
}

/**
 * SO3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SO3GroupBase<Derived>& m, std::uint32_t const version)
{
    Eigen::Quaternion<typename SO3GroupBase<Derived>::Scalar> quaternion;
    archive(cereal::make_nvp("Quaternion", quaternion));
    m.setQuaternion(quaternion);
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, SO3GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.unit_quaternion()));
}

/**
 * SE2
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SE2GroupBase<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so2()));
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, SE2GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so2()));
}

/**
 * SE3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, SE3GroupBase<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so3()));
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, SE3GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.so3()));
}

/**
 * RxSO3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, RxSO3GroupBase<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.quaternion()));
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, RxSO3GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Quaternion", m.quaternion()));
}

/**
 * Sim3
 */    
template<typename Archive, typename Derived>
void load(Archive & archive, Sim3GroupBase<Derived>& m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.rxso3()));
}
  
template<typename Archive, typename Derived>
void save(Archive & archive, Sim3GroupBase<Derived> const & m, std::uint32_t const version)
{
    archive(cereal::make_nvp("Translation", m.translation()));
    archive(cereal::make_nvp("Rotation", m.rxso3()));
}

}

#endif // VISIONCORE_HAVE_CEREAL

// let's put ostreams here
namespace Sophus
{

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SO2GroupBase<Derived>& p)
{
    os << "(" << p.log() << ")"; 
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SO3GroupBase<Derived>& p)
{
    os << "(" << p.unit_quaternion().x() << "," << p.unit_quaternion().y() << "," << p.unit_quaternion().z() << "|" << p.unit_quaternion().w() << ")"; 
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SE2GroupBase<Derived>& p)
{
    os << "[t = " << p.translation()(0) << "," << p.translation()(1) << " | r = " << p.so2() << ")";
    return os;
}

template<typename Derived>
inline std::ostream& operator<<(std::ostream& os, const SE3GroupBase<Derived>& p)
{
    os << "[t = " << p.translation()(0) << "," << p.translation()(1) << "," << p.translation()(2) << " | r = " << p.so3() << ")";
    return os;
}

}

namespace Eigen
{

template <typename _Scalar, int _AmbientDim, int _Options>
inline std::ostream& operator<<(std::ostream& os, const Hyperplane<_Scalar,_AmbientDim,_Options>& p)
{
    os << "Hyperplane(" << p.normal() << " , " << p.offset() << ")";
    return os;
}

}

#endif // VISIONCORE_EIGEN_SERIALIZERS_HPP
