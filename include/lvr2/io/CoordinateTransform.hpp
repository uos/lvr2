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
 * CoordinateTransform.hpp
 *
 *  Created on: 17.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef COORDINATETRANSFORM_HPP_
#define COORDINATETRANSFORM_HPP_

#include "lvr2/io/PointBuffer.hpp"

namespace lvr2
{

	enum COORD_SYSTEM {SLAM6D, OPENGL_METERS, OPENGL_MM};

	void convert(COORD_SYSTEM from, COORD_SYSTEM to, float* point);
	void convert(COORD_SYSTEM from, COORD_SYSTEM to, PointBufferPtr &buffer);

	/**
	 * @brief 	Stores information to transform a 3D point into
	 * 			a different coordinate system. It is assumed, that
	 * 			the coordinate called x is refered to as coordinate 0,
	 * 			y is 1 and z is 2. 
	 * 
	 */
	template<typename T> 
	struct CoordinateTransform
	{
		CoordinateTransform(const unsigned char& _x = 0,
						    const unsigned char& _y = 1,
						    const unsigned char& _z = 2,
						    const T& _sx = 1.0,
						    const T& _sy = 1.0,
						    const T& _sz = 1.0) : 
						    x(_x), y(_y), z(_z), sx(_sx), sy(_sy), sz(_sz) {}

		/// Returns true, if the saved information actually 
		/// is a transform. False only, if the 
		/// default values are used.
		bool transforms()
		{
			return ((x != 0) || (y != 1) || (z != 2) || (sx != 1.0) || (sy != 1.0) || (sz != 1.0));
		} 
						   
		/// Position of the x coordinate in the target system
		unsigned char x;

		/// Position of the y coordinate in the target system
		unsigned char y;

		/// Position of the z coordinate in the target system
		unsigned char z;

		/// Scale factor of the x coordinate in the source system
		/// to match the target systems' scale
		T sx;

		/// Scale factor of the y coordinate in the source system
		/// to match the target systems' scale
		T sy;

		/// Scale factor of the y coordinate in the source system
		/// to match the target systems' scale
		T sz;
	};
	
} // namespace lvr2

#endif /* COORDINATETRANSFORM_HPP_ */
