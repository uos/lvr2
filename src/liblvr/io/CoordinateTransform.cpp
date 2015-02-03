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
 * CoordinateTransform.cpp
 *
 *  Created on: 17.04.2012
 *      Author: Thomas Wiemann
 */

#include "io/CoordinateTransform.hpp"

#include "io/Timestamp.hpp"

#include <boost/shared_ptr.hpp>
#include <iostream>
using std::cout;
using std::endl;

namespace lvr
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
	size_t n;
	floatArr p = buffer->getPointArray(n);
	for(int i = 0; i < n; i++)
	{
		int pos = 3 * i;
		float* point = &p[pos];
		convert(OPENGL_METERS, SLAM6D, point);
	}
}

} // namespace lvr
