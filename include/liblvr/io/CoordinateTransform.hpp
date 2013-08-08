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
 * CoordinateTransform.hpp
 *
 *  Created on: 17.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef COORDINATETRANSFORM_HPP_
#define COORDINATETRANSFORM_HPP_

#include "PointBuffer.hpp"

namespace lvr
{

	enum COORD_SYSTEM {SLAM6D, OPENGL_METERS, OPENGL_MM};

	void convert(COORD_SYSTEM from, COORD_SYSTEM to, float* point);
	void convert(COORD_SYSTEM from, COORD_SYSTEM to, PointBufferPtr &buffer);
}

#endif /* COORDINATETRANSFORM_HPP_ */
