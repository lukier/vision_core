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
 */

// system
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <exception>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <valarray>

// testing framework & libraries
#include <gtest/gtest.h>

// google logger
#include <glog/logging.h>

#include <math/Fitting.hpp>
#include <EigenSerializers.hpp>

class Test_Fitting : public ::testing::Test
{
public:   
    Test_Fitting()
    {
        
    }
    
    virtual ~Test_Fitting()
    {
        
    }
};


TEST_F(Test_Fitting, Dummy) 
{
    typedef double Scalar;
    typedef core::PlaneFitting<Scalar> PFT;
    
    PFT pf;
    
    pf(Eigen::Matrix<Scalar,3,1>(-1.0f,-2.0f,0.0f));
    pf(Eigen::Matrix<Scalar,3,1>(3.0f,1.0f,4.0f));
    pf(Eigen::Matrix<Scalar,3,1>(0.0f,-1.0f,2.0f));
    
    typename PFT::PlaneT p1;
    Scalar cur1;
    
    bool ok1 = pf.getPlane(p1, cur1);
    
    LOG(INFO) << "Eigen: " << ok1 << " = " << p1 << " / " << cur1;
}
