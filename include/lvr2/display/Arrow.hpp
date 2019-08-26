/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
 * PointCloud.h
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#ifndef ARROW_H_
#define ARROW_H_

#include "lvr2/display/Renderable.hpp"

#include "lvr2/geometry/BaseVector.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;

namespace lvr2
{

class Arrow : public Renderable{
public:
    Arrow(string filename);
    Arrow(int color);
    virtual ~Arrow();
    virtual inline void render();

    void setPosition(double x, double y, double z, double roll, double pitch, double yaw);
    
    inline void setColor(int color) {
    	this->color = color;	
    };
    
private:
	double roll, pitch, yaw;
	int color;
    Matrix4<BaseVector<float>> rotation;
};

	


inline void EulerToMatrix(const double *rPos, const double *rPosTheta, float *alignxf)
{
  double sx = sin(rPosTheta[0]);
  double cx = cos(rPosTheta[0]);
  double sy = sin(rPosTheta[1]);
  double cy = cos(rPosTheta[1]);
  double sz = sin(rPosTheta[2]);
  double cz = cos(rPosTheta[2]);

  alignxf[0]  = cy*cz;
  alignxf[1]  = sx*sy*cz + cx*sz;
  alignxf[2]  = -cx*sy*cz + sx*sz;
  alignxf[3]  = 0.0;
  alignxf[4]  = -cy*sz;
  alignxf[5]  = -sx*sy*sz + cx*cz;
  alignxf[6]  = cx*sy*sz + sx*cz;
  alignxf[7]  = 0.0;
  alignxf[8]  = sy;
  alignxf[9]  = -sx*cy;
  alignxf[10] = cx*cy;
  
  alignxf[11] = 0.0;

  alignxf[12] = rPos[0];
  alignxf[13] = rPos[1];
  alignxf[14] = rPos[2];
  alignxf[15] = 1;
};

} // namespace lvr2

// @TODO Why do we need this?
void render();

#endif /* Arrow_H_ */
