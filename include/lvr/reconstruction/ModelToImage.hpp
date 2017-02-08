/* Copyright (C) 2016 Uni Osnabrück
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
 * ModelToImage.h
 *
 *  Created on: Jan 25, 2017
 *      Author: Thomas Wiemann (twiemann@uos.de)
 */

#ifndef SRC_LIBLVR_RECONSTRUCTION_MODELTOIMAGE_HPP_
#define SRC_LIBLVR_RECONSTRUCTION_MODELTOIMAGE_HPP_

#include <lvr/io/Model.hpp>

#include <opencv/cv.h>

namespace lvr {


class ModelToImage {
public:

    enum ProjectionType {CYLINDRICAL, CONICAL, EQUALAREACYLINDRICAL,
                         RECTILINEAR, PANNINI, STEREOGRAPHIC,
                         ZAXIS, AZIMUTHAL};

	ModelToImage(ModelPtr model);
	ModelToImage(
			PointBufferPtr buffer,
            ProjectionType projection,
			int width, int height,
            int minZ, int maxZ,
			int minHorizontenAngle, int maxHorizontalAngle,
			int mainVerticalAngle, int maxVerticalAngle,
            bool imageOptimization);

	virtual ~ModelToImage();

	void getCVMatrix(cv::Mat& image);

private:



	/// Pointer to the initial point cloud
	PointBufferPtr 	m_points;

	/// Image width
	int				m_width;

	/// Image height
	int				m_height;

	/// Min horizontal opening angle
	int				m_minHAngle;

	/// Max horizontal opening angle
	int				m_maxHAngle;

	/// Min horizontal opening angle
	int				m_minVAngle;

	/// Max horizontal opening angle
	int				m_maxVAngle;

	/// Image optimization flag
	bool			m_optimize;

    int             m_xSize;

    int             m_ySize;

    int             m_maxWidth;

    int             m_maxHeight;

    int             m_minHeight;

    float           m_xFactor;

    float           m_yFactor;
};

} /* namespace lvr */

#endif /* SRC_LIBLVR_RECONSTRUCTION_MODELTOIMAGE_HPP_ */
