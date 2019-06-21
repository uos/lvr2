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
 * CoordinateTransform.cpp
 *
 *  Created on: 17.04.2012
 *      Author: Thomas Wiemann
 */

#include "lvr2/io/CoordinateTransform.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <boost/shared_ptr.hpp>
#include <iostream>
using std::cout;
using std::endl;

namespace lvr2
{

void convert(COORD_SYSTEM from, COORD_SYSTEM to, float* point)
{
	if(from == OPENGL_METERS)
	{
		if(to == SLAM6D)
		{
			float x = point[0];
			float y = point[1];
			float z = point[2];

			point[0] = 100 * x;
			point[1] = 100 * y;
			point[2] = -100 * z;
		}
		else
		{
			cout << timestamp << "Target coordinate system not supported." << endl;
		}
	}
	else
	{
		cout << timestamp << "Source coordinate system not supported." << endl;
	}
}

void convert(COORD_SYSTEM from, COORD_SYSTEM to, PointBufferPtr& buffer)
{
	size_t n = buffer->numPoints();
	floatArr p = buffer->getPointArray();
	for(size_t i = 0; i < n; i++)
	{
		int pos = 3 * i;
		float* point = &p[pos];
		convert(OPENGL_METERS, SLAM6D, point);
	}
}

} // namespace lvr2
