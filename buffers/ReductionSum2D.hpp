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
 * Helper for CUDA 2D reductions.
 * ****************************************************************************
 */

#ifndef CORE_REDUCTION_SUM2D_HPP
#define CORE_REDUCTION_SUM2D_HPP

#ifdef CORE_HAVE_CUDA

#include <Platform.hpp>
#include <LaunchUtils.hpp>

#include <buffers/Buffer1D.hpp>

namespace core
{

template<typename T>
class HostReductionSum2DView : public Buffer1DView<T,core::TargetDeviceCUDA>
{
public:
    typedef Buffer1DView<T,core::TargetDeviceCUDA> BaseType;
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView() 
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ~HostReductionSum2DView()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView(const HostReductionSum2DView<T>& img ) : BaseType(img)
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView(HostReductionSum2DView<T>&& img) : BaseType(std::move(img))
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView(T* optr, std::size_t s) : BaseType(optr,s)
    {  
        
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView<T>& operator=(const HostReductionSum2DView<T>& other)
    {
        BaseType::operator=(other);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DView<T>& operator=(HostReductionSum2DView<T>&& img)
    {
        BaseType::operator=(img);
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline const BaseType& getSumImage() const { return (const BaseType&)*this; }
    EIGEN_DEVICE_FUNC inline BaseType& getSumImage() { return (BaseType&)*this; }

    inline void getFinalSum(T& sum)
    {
        sum = thrust::reduce(BaseType::begin(), BaseType::end(), core::zero<T>() , thrust::plus<T>() );
    }
    
    static inline std::size_t getSharedMemorySize(const dim3& blockDim) { return sizeof(T) * blockDim.x * blockDim.y; }
};

template<typename T>
class HostReductionSum2DManaged : public HostReductionSum2DView<T>
{
public:
    typedef HostReductionSum2DView<T> ViewT;
    
    HostReductionSum2DManaged() = delete;
    
    HostReductionSum2DManaged(std::size_t s) : ViewT()
    {
        ViewT::memptr = 0;
        ViewT::xsize = s;
        typename TargetDeviceCUDA<T>::PointerType ptr = 0;
        TargetDeviceCUDA<T>::AllocateMem(&ptr, ViewT::xsize);
        ViewT::memptr = ptr;
    }
    
    HostReductionSum2DManaged(dim3 gridDim) : HostReductionSum2DManaged(gridDim.x * gridDim.y * gridDim.z)
    {
        
    }
    
    ~HostReductionSum2DManaged()
    {
        if(ViewT::memptr != 0)
        {
            TargetDeviceCUDA<T>::DeallocatePitchedMem(ViewT::memptr);
        }
    }
    
    HostReductionSum2DManaged(const HostReductionSum2DManaged<T>& img) = delete;
    
    inline HostReductionSum2DManaged(HostReductionSum2DManaged<T>&& img) : ViewT(std::move(img))
    {
        
    }
    
    HostReductionSum2DManaged<T>& operator=(const HostReductionSum2DManaged<T>& img) = delete;
    
    inline HostReductionSum2DManaged<T>& operator=(HostReductionSum2DManaged<T>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

template<typename T, std::size_t Levels>
class HostReductionSum2DPyramidView
{
public:
    typedef HostReductionSum2DView<T> BufferType;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline ~HostReductionSum2DPyramidView()
    {
        
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView(const HostReductionSum2DPyramidView<T,LevelCount>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            rbufs[l] = pyramid.rbufs[l];
        }
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView(HostReductionSum2DPyramidView<T,LevelCount>&& pyramid) 
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            rbufs[l] = std::move(pyramid.rbufs[l]);
        }
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView<T,LevelCount>& operator=(const HostReductionSum2DPyramidView<T,LevelCount>& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            rbufs[l] = pyramid.rbufs[l];
        }
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView<T,LevelCount>& operator=(HostReductionSum2DPyramidView<T,LevelCount>&& pyramid)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            rbufs[l] = std::move(pyramid.rbufs[l]);
        }
        
        return *this;
    }
    
    EIGEN_DEVICE_FUNC inline BufferType& operator[](std::size_t i)
    {
        assert(i < LevelCount);
        return rbufs[i];
    }
    
    EIGEN_DEVICE_FUNC inline const BufferType& operator[](std::size_t i) const
    {
        assert(i < LevelCount);
        return rbufs[i];
    }
    
    EIGEN_DEVICE_FUNC inline BufferType& operator()(std::size_t i)
    {
        assert(i < LevelCount);
        return rbufs[i];
    }
    
    EIGEN_DEVICE_FUNC inline const BufferType& operator()(std::size_t i) const
    {
        assert(i < LevelCount);
        return rbufs[i];
    }
    
    template<std::size_t SubLevels>
    EIGEN_DEVICE_FUNC inline HostReductionSum2DPyramidView<T,SubLevels> subPyramid(std::size_t startLevel)
    {
        assert(startLevel + SubLevels < LevelCount);
        
        HostReductionSum2DPyramidView<T,SubLevels> pyr;
        
        for(std::size_t l = 0 ; l < SubLevels; ++l) 
        {
            pyr.rbufs[l] = rbufs[startLevel+l];
        }
        
        return pyr;
    }
    
    inline void memset(unsigned char v = 0)
    {
        for(std::size_t l = 0 ; l < LevelCount ; ++l) 
        {
            rbufs[l].memset(v);
        }
    }
protected:
    BufferType rbufs[LevelCount];
};

template<typename T, std::size_t Levels>
class HostReductionSum2DPyramidManaged : public HostReductionSum2DPyramidView<T, Levels>
{
public:
    typedef HostReductionSum2DPyramidView<T, Levels> ViewT;
    static const std::size_t LevelCount = Levels;
    typedef T ValueType;
    
    HostReductionSum2DPyramidManaged() = delete;
    
    HostReductionSum2DPyramidManaged(dim3 blockDim, std::size_t w, std::size_t h)
    {       
        // Build power of two structure
        for(std::size_t l = 0; l < LevelCount ; ++l ) 
        {
            dim3 gridDim = core::calculateGridDim(blockDim, w >> l, h >> l);
            typename TargetDeviceCUDA<T>::PointerType ptr = 0;
            const std::size_t lin_size = gridDim.x * gridDim.y * gridDim.z;
            TargetDeviceCUDA<T>::AllocateMem(&ptr, lin_size);
            ViewT::rbufs[l] = core::HostReductionSum2DView<T>((T*)ptr, lin_size);
        }
    }
    
    HostReductionSum2DPyramidManaged(const HostReductionSum2DPyramidManaged<T,LevelCount>& img) = delete;
    
    inline HostReductionSum2DPyramidManaged(HostReductionSum2DPyramidManaged<T,LevelCount>&& img) : ViewT(std::move(img))
    {
        
    }
    
    HostReductionSum2DPyramidManaged<T,LevelCount>& operator=(const HostReductionSum2DPyramidManaged<T,LevelCount>& img) = delete;
    
    inline HostReductionSum2DPyramidManaged<T,LevelCount>& operator=(HostReductionSum2DPyramidManaged<T,LevelCount>&& img)
    {
        ViewT::operator=(std::move(img));
        return *this;
    }
    
    inline ~HostReductionSum2DPyramidManaged()
    {
        for(std::size_t l = 0; l < LevelCount ; ++l)
        {
            TargetDeviceCUDA<T>::DeallocatePitchedMem((typename TargetDeviceCUDA<T>::PointerType)ViewT::rbufs[l].ptr());
        }
    }
    
    inline const ViewT& view() const { return (const ViewT&)*this; }
    inline ViewT& view() { return (ViewT&)*this; }
};

#ifdef __CUDACC__

template<typename T, unsigned MAX_BLOCK_X, unsigned MAX_BLOCK_Y>
class DeviceReductionSum2D
{
public:
    EIGEN_PURE_DEVICE_FUNC inline T& getThisBlock()
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        return sReduce[tid];
    }

    EIGEN_PURE_DEVICE_FUNC inline void reduceBlock(HostReductionSum2DView<T>& dSum)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        
        __syncthreads();
        
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  
        {
            if( tid < S ) 
            {
                sReduce[tid] += sReduce[tid+S];
            }
            __syncthreads();
        }
        
        if( tid == 0) 
        {
            dSum.getSumImage()(bid) = sReduce[0];
        }
    }
    
private:
    T sReduce[MAX_BLOCK_X * MAX_BLOCK_Y];
};

template<typename T>
class DeviceReductionSum2DDynamic
{
public:
    EIGEN_PURE_DEVICE_FUNC inline T& getThisBlock()
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        return sReduce[tid];
    }

    EIGEN_PURE_DEVICE_FUNC inline void reduceBlock(HostReductionSum2DView<T>& dSum)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        
        __syncthreads();
        
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  
        {
            if( tid < S ) 
            {
                sReduce[tid] += sReduce[tid+S];
            }
            __syncthreads();
        }
        
        if( tid == 0) 
        {
            dSum.getSumImage()(bid) = sReduce[0];
        }
    }
    
private:
    core::SharedMemory<T> sReduce;
};

template<typename TA, typename TB, unsigned MAX_BLOCK_X, unsigned MAX_BLOCK_Y>
class DeviceReductionSumTwo2D
{
public:
    EIGEN_PURE_DEVICE_FUNC inline TA& getThisBlockA() { return sReduceA[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TB& getThisBlockB() { return sReduceB[threadIdx.y*blockDim.x + threadIdx.x]; }
    
    EIGEN_PURE_DEVICE_FUNC inline void reduceBlock(HostReductionSum2DView<TA>& dSumA,
                                                   HostReductionSum2DView<TB>& dSumB)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        
        __syncthreads();
        
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  
        {
            if( tid < S ) 
            {
                sReduceA[tid] += sReduceA[tid+S];
                sReduceB[tid] += sReduceB[tid+S];
            }
            __syncthreads();
        }
        
        if( tid == 0) 
        {
            dSumA.getSumImage()(bid) = sReduceA[0];
            dSumB.getSumImage()(bid) = sReduceB[0];
        }
    }
    
private:
    TA sReduceA[MAX_BLOCK_X * MAX_BLOCK_Y];
    TB sReduceB[MAX_BLOCK_X * MAX_BLOCK_Y];
};

template<typename TA, typename TB, typename TC, unsigned MAX_BLOCK_X, unsigned MAX_BLOCK_Y>
class DeviceReductionSumThree2D
{
public:
    EIGEN_PURE_DEVICE_FUNC inline TA& getThisBlockA() { return sReduceA[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TB& getThisBlockB() { return sReduceB[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TC& getThisBlockC() { return sReduceC[threadIdx.y*blockDim.x + threadIdx.x]; }
    
    EIGEN_PURE_DEVICE_FUNC inline void reduceBlock(HostReductionSum2DView<TA>& dSumA,
                                                   HostReductionSum2DView<TB>& dSumB,
                                                   HostReductionSum2DView<TC>& dSumC)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        
        __syncthreads();
        
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  
        {
            if( tid < S ) 
            {
                sReduceA[tid] += sReduceA[tid+S];
                sReduceB[tid] += sReduceB[tid+S];
                sReduceC[tid] += sReduceC[tid+S];
            }
            __syncthreads();
        }
        
        if( tid == 0) 
        {
            dSumA.getSumImage()(bid) = sReduceA[0];
            dSumB.getSumImage()(bid) = sReduceB[0];
            dSumC.getSumImage()(bid) = sReduceC[0];
        }
    }
    
private:
    TA sReduceA[MAX_BLOCK_X * MAX_BLOCK_Y];
    TB sReduceB[MAX_BLOCK_X * MAX_BLOCK_Y];
    TC sReduceC[MAX_BLOCK_X * MAX_BLOCK_Y];
};

template<typename TA, typename TB, typename TC, typename TD, unsigned MAX_BLOCK_X, unsigned MAX_BLOCK_Y>
class DeviceReductionSumFour2D
{
public:
    EIGEN_PURE_DEVICE_FUNC inline TA& getThisBlockA() { return sReduceA[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TB& getThisBlockB() { return sReduceB[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TC& getThisBlockC() { return sReduceC[threadIdx.y*blockDim.x + threadIdx.x]; }
    EIGEN_PURE_DEVICE_FUNC inline TD& getThisBlockD() { return sReduceD[threadIdx.y*blockDim.x + threadIdx.x]; }
    
    EIGEN_PURE_DEVICE_FUNC inline void reduceBlock(HostReductionSum2DView<TA>& dSumA,
                                                   HostReductionSum2DView<TB>& dSumB,
                                                   HostReductionSum2DView<TC>& dSumC,
                                                   HostReductionSum2DView<TD>& dSumD)
    {
        const unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
        const unsigned int bid = blockIdx.y*gridDim.x + blockIdx.x;
        
        __syncthreads();
        
        for(unsigned S=blockDim.y*blockDim.x/2;S>0; S>>=1)  
        {
            if( tid < S ) 
            {
                sReduceA[tid] += sReduceA[tid+S];
                sReduceB[tid] += sReduceB[tid+S];
                sReduceC[tid] += sReduceC[tid+S];
                sReduceD[tid] += sReduceD[tid+S];
            }
            __syncthreads();
        }
        
        if( tid == 0) 
        {
            dSumA.getSumImage()(bid) = sReduceA[0];
            dSumB.getSumImage()(bid) = sReduceB[0];
            dSumC.getSumImage()(bid) = sReduceC[0];
            dSumD.getSumImage()(bid) = sReduceD[0];
        }
    }
    
private:
    TA sReduceA[MAX_BLOCK_X * MAX_BLOCK_Y];
    TB sReduceB[MAX_BLOCK_X * MAX_BLOCK_Y];
    TC sReduceC[MAX_BLOCK_X * MAX_BLOCK_Y];
    TD sReduceD[MAX_BLOCK_X * MAX_BLOCK_Y];
};
#endif // CORE_CUDA_KERNEL_SPACE

}

#endif // CORE_HAVE_CUDA

#endif // CORE_REDUCTION_SUM2D_HPP
