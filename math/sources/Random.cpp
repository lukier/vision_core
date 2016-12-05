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
 * Simple generation of random numbers.
 * ****************************************************************************
 */

#include <curand.h>

#include <math/Random.hpp>

template<typename Target>
struct TargetDispatcher { };

template<>
struct TargetDispatcher<core::TargetDeviceCUDA>
{
    static inline curandGenerator_t create(curandRngType rngt)
    {
        curandGenerator_t ret;
        curandCreateGenerator(&ret, rngt);
        return ret;
    }
};

template<>
struct TargetDispatcher<core::TargetHost>
{
    static inline curandGenerator_t create(curandRngType rngt)
    {
        curandGenerator_t ret;
        curandCreateGeneratorHost(&ret, rngt);
        return ret;
    }
};

template<typename Target>
core::math::RandomGenerator<Target>::RandomGenerator(uint64_t seed)
{
    handle = TargetDispatcher<Target>::create(CURAND_RNG_PSEUDO_DEFAULT);
    
    curandStatus_t res = curandSetPseudoRandomGeneratorSeed(handle, seed);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename Target>
core::math::RandomGenerator<Target>::~RandomGenerator()
{
    curandDestroyGenerator(handle);
}

template<typename T, typename Target>
void core::math::generateRandom(RandomGenerator<Target>& gen, core::Buffer1DView<T,Target>& bufout, const core::types::Gaussian<typename core::type_traits<T>::ChannelType>& gauss)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.size(), gauss.mean(), gauss.stddev());
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void core::math::generateRandom(RandomGenerator<Target>& gen, core::Buffer2DView<T,Target>& bufout, const core::types::Gaussian<typename core::type_traits<T>::ChannelType>& gauss)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.height() * bufout.pitch(), gauss.mean(), gauss.stddev());
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void core::math::generateRandom(RandomGenerator<Target>& gen, core::Buffer1DView<T,Target>& bufout, const typename core::type_traits<T>::ChannelType& mean, const typename core::type_traits<T>::ChannelType& stddev)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.size(), mean, stddev);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

template<typename T, typename Target>
void core::math::generateRandom(RandomGenerator<Target>& gen, core::Buffer2DView<T,Target>& bufout, const typename core::type_traits<T>::ChannelType& mean, const typename core::type_traits<T>::ChannelType& stddev)
{
    curandStatus_t res = curandGenerateNormal(gen.handle, bufout.ptr(), bufout.height() * bufout.pitch(), mean, stddev);
    if(res != CURAND_STATUS_SUCCESS)
    {
        std::stringstream ss;
        ss << "curand error: " << res;
        throw std::runtime_error(ss.str());
    }
}

#define GENERATE_CODE(TYPE) \
template void core::math::generateRandom<TYPE,core::TargetDeviceCUDA>(RandomGenerator<core::TargetDeviceCUDA>& gen, core::Buffer1DView<TYPE,core::TargetDeviceCUDA>& bufout, const core::types::Gaussian<typename core::type_traits<TYPE>::ChannelType>& gauss); \
template void core::math::generateRandom<TYPE,core::TargetHost>(RandomGenerator<core::TargetHost>& gen, core::Buffer1DView<TYPE,core::TargetHost>& bufout, const core::types::Gaussian<typename core::type_traits<TYPE>::ChannelType>& gauss); \
template void core::math::generateRandom<TYPE,core::TargetDeviceCUDA>(RandomGenerator<core::TargetDeviceCUDA>& gen, core::Buffer2DView<TYPE,core::TargetDeviceCUDA>& bufout, const core::types::Gaussian<typename core::type_traits<TYPE>::ChannelType>& gauss); \
template void core::math::generateRandom<TYPE,core::TargetHost>(RandomGenerator<core::TargetHost>& gen, core::Buffer2DView<TYPE,core::TargetHost>& bufout, const core::types::Gaussian<typename core::type_traits<TYPE>::ChannelType>& gauss); \
template void core::math::generateRandom<TYPE,core::TargetDeviceCUDA>(RandomGenerator<core::TargetDeviceCUDA>& gen, core::Buffer1DView<TYPE,core::TargetDeviceCUDA>& bufout, const typename core::type_traits<TYPE>::ChannelType& mean, const typename core::type_traits<TYPE>::ChannelType& stddev); \
template void core::math::generateRandom<TYPE,core::TargetHost>(RandomGenerator<core::TargetHost>& gen, core::Buffer1DView<TYPE,core::TargetHost>& bufout, const typename core::type_traits<TYPE>::ChannelType& mean, const typename core::type_traits<TYPE>::ChannelType& stddev); \
template void core::math::generateRandom<TYPE,core::TargetDeviceCUDA>(RandomGenerator<core::TargetDeviceCUDA>& gen, core::Buffer2DView<TYPE,core::TargetDeviceCUDA>& bufout, const typename core::type_traits<TYPE>::ChannelType& mean, const typename core::type_traits<TYPE>::ChannelType& stddev); \
template void core::math::generateRandom<TYPE,core::TargetHost>(RandomGenerator<core::TargetHost>& gen, core::Buffer2DView<TYPE,core::TargetHost>& bufout, const typename core::type_traits<TYPE>::ChannelType& mean, const typename core::type_traits<TYPE>::ChannelType& stddev);

GENERATE_CODE(float)

template struct core::math::RandomGenerator<core::TargetDeviceCUDA>;
template struct core::math::RandomGenerator<core::TargetHost>;
