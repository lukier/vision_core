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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != cv::EVENT_LBUTTONDOWN )
        return;
    
    LOG(INFO) << "Point " << x << " , " << y;
}

TEST_F(Test_Fitting, Dummy) 
{
    cv::namedWindow("Test");
    cv::setMouseCallback( "Test", onMouse, 0 );
    
    cv::Mat img = cv::imread("/home/lukier/test_image_new_cam.png", cv::IMREAD_GRAYSCALE);
    
    const Eigen::Vector2d pt1(407.0,455.0);
    const Eigen::Vector2d pt2(795.0,460.0);
    const Eigen::Vector2d pt3(612.0,832.0);
    
    const core::CircleT<double> c = core::circleFrom3Points(pt1, pt2, pt3);
    
    LOG(INFO) << "Circle: " << c;
    
    cv::circle(img, cv::Point2f(c.coeff()(0), c.coeff()(1)), c.radius()*1.025, cv::Scalar(255,0,0));
    cv::circle(img, cv::Point2f(c.coeff()(0), c.coeff()(1)), 544.0*0.975, cv::Scalar(255,0,0));
    
    while(1)
    {
        cv::imshow("Test", img);
        
        int key = cv::waitKey(-1);
        if(key == 'q') { break; }
    }
}
