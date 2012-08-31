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
 * Trans.hpp
 *
 *  @date 06.08.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef TRANS_HPP_
#define TRANS_HPP_

#include <cstring>
#include <math.h>
#include <cstdio>
#include <texture/Texture.hpp>
#include <texture/ImageProcessor.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace lssr {


/**
 * @brief	This class holds a transformation matrix and some additional information
 */
class Trans
{
public:
	/**
	 * \brief Constructor.
	 *
	 * \param	p1	Three points in the first texture.
	 * \param	p2	Three corresponding points in the second texture.
	 * \param	w1	The width of the first texture.
	 * \param	h1	The height of the first texture.
	 * \param	w2	The width of the second texture.
	 * \param	h2	The height of the second texture.
	 *
	 */
	Trans(cv::Point2f* p1, cv::Point2f* p2, int w1, int h1, int w2, int h2);

	/**
	 * \brief == operator.
	 *
	 */	
	bool operator==(Trans other);
	
	///The number of votes this transformation has
	int m_votes;

	///The transformation matrix	
	cv::Mat m_trans;

	///0 = not mirrored, 1 = mirrored at horizontal axis, 2 = mirrored at vertical axis
	unsigned char m_mirrored;
	
};
}

#endif /* TRANS_HPP_ */
