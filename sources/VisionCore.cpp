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
 * Compile all of them as a sanity check.
 * ****************************************************************************
 */

#include <CUDAException.hpp>
#include <EigenSerializers.hpp>
#include <LaunchUtils.hpp>
#include <MemoryPolicy.hpp>
#include <Platform.hpp>
#include <buffers/Buffer1D.hpp>
#include <buffers/Buffer2D.hpp>
#include <buffers/Buffer3D.hpp>
#include <buffers/BufferPyramid.hpp>
#include <buffers/CUDATexture.hpp>
#include <buffers/GPUVariable.hpp>
#include <buffers/Image2D.hpp>
#include <buffers/ImagePyramid.hpp>
#include <buffers/PyramidBase.hpp>
#include <buffers/ReductionSum2D.hpp>
#include <buffers/Volume.hpp>
#include <control/PID.hpp>
#include <control/VelocityProfile.hpp>
#include <image/ImagePatch.hpp>
#include <image/PixelConvert.hpp>
#include <math/Angles.hpp>
#include <math/DenavitHartenberg.hpp>
#include <math/Divergence.hpp>
#include <math/Fitting.hpp>
#include <math/HammingDistance.hpp>
#include <math/Kalman.hpp>
#include <math/LeastSquares.hpp>
#include <math/LiangBarsky.hpp>
//#include <math/LocalParamSE3.hpp>
#include <math/LossFunctions.hpp>
#include <math/PolarSpherical.hpp>
#include <math/Random.hpp>
#include <math/RANSAC.hpp>
#include <math/Statistics.hpp>
#include <types/AxisAlignedBoundingBox.hpp>
#include <types/CostVolumeElement.hpp>
#include <types/Endian.hpp>
#include <types/Gaussian.hpp>
#include <types/Hypersphere.hpp>
#include <types/Polynomial.hpp>
#include <types/Rectangle.hpp>
#include <types/SDF.hpp>
