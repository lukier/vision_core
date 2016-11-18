Vision/Core
=============

This is a small collection C++11 classes useful for various computer vision / robotics / GPGPU problems, largely inspired by the Kangaroo library.
Nothing tested properly, most under development, unstable.

## Features
### Buffers
* Buffer1D - linear vector of elements, either Host/CUDA or OpenCL allocators.
* Buffer2D - 2D buffer of elements, either Host/CUDA or OpenCL allocators.
* Buffer3D - 3D buffer of elements, either Host/CUDA or OpenCL allocators.
* BufferPyramid - 2D buffer pyramid, either Host/CUDA or OpenCL allocators.
* CUDATexture - wrapper to see 2D buffer or OpenGL to get CUDA texture access.
* GPUVariable - single element buffer pretty much.
* Image2D - derived from Buffer2D, but adds interpolations and derivatives.
* ImagePyramid - Image2D pyramid.
* ReductionSum2D - 2D reductions CUDA + thrust, up to 4 variables and reductions over pyramids.
* Volume - derived from Buffer3D, but adds interpolations and derivatives.

### Control
* PID - simple PID controller.
* VelocityProfile - Trapezoidal/Constant velocity profile generators.

### Image
* ImagePatch - convenient access to a patch in a Buffer2D.
* PixelConvert - pixel type conversions.

### Math
* Angles - angular quantities utilities + circular mean.
* DenavitHartenberg - robotic joint generator.
* Divergence - divergence operators.
* Fitting - fitting plane to points, circle or transformation.
* HammingDistance - portable Hamming distance (CPU/CUDA).
* Kalman - TBD
* LeastSquares - simple least squares solver.
* LiangBarsky - Liang-Barsky Intersection Test.
* LocalParamSE3 - Sophus::SE3 parametrization for Ceres
* LossFunctions - L1, L2, Huber, Cauchy, Turkey, GermanMcClure and Welsch.
* PolarSpherical - conversions from/to cartesian/polar/spherical coordinate systems.
* Random - simple wrapper over C random functions.
* RANSAC - TBD
* Statistics - running stats on single variables, Eigen or multivariate stats on Eigen types. Pearson product-moment.

### Types
* AxisAlignedBoundingBox - extending Eigen's AlignedBox.
* CostVolumeElement - for plane sweep cost volume.
* Endian - endianess converter.
* Gaussian - Eigen-like type for Gaussian distribution.
* Hypersphere - Eigen-like type for (Hyper)sphere.
* Polynomial - Eigen-like type for polynomials.
* Rectangle  - Eigen-like type for rectangle (2D).
* SDF - element for Sign-Distance function field.

### WrapGL
* WrapGLBuffer - C++ wrappers over OpenGL buffers.
* WrapGLContext - C++ wrappers over OpenGL context creation (only GLbinding).
* WrapGLFramebuffer - C++ wrappers over OpenGL framebuffers.
* WrapGLQuery - C++ wrappers over OpenGL queries.
* WrapGLSampler - C++ wrappers over OpenGL samplers.
* WrapGLShader - C++ wrappers over OpenGL shaders.
* WrapGLTexture - C++ wrappers over OpenGL textures.
* WrapGLTransformFeedback - C++ wrappers over OpenGL transform feedback.
* WrapGLVertexArrayObject - C++ wrappers over OpenGL VAO.

### Misc
* CUDAException - C++ exception from CUDA return codes.
* CUDATypes - CUDA types for non-CUDA platforms and various CUDA helpers.
* EigenSerializers - Cereal serializers for Eigen/Sophus. Some ostream operators.
* LaunchUtils - CUDA launch calculators + Intel TBB launchers.
* MemoryPolicy{CUDA,Host,OpenCL} - underlying allocators for Buffers.
* Plarform - main header file with macros, traits and helper functions.
* SophusInterpolations - linear and B-spline interpolations over Sophus Lie groups.

## TODO
Everything

## License

_Copyright © `2016`, `Robert Lukierski`_  
_All rights reserved._

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
