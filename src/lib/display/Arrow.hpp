/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * PointCloud.h
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#ifndef ARROW_H_
#define ARROW_H_

#include "geometry/Vertex.hpp"
#include "Renderable.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;

namespace lssr
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
	Matrix4 rotation;
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

} // namespace lssr

void render();

#endif /* Arrow_H_ */
