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
 * Texture.hpp
 *
 *  @date 08.09.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include "Region.hpp"
#include "reconstruction/PointsetSurface.hpp"
#include "io/PPMIO.hpp"
#include <boost/lexical_cast.hpp>

namespace lssr {

template<typename VertexT, typename NormalT>
class HalfEdgeVertex;

/**
 * @brief	This class represents a texture.
 */
template<typename VertexT, typename NormalT>
class Texture {
public:

	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;

	/**
	 * @brief 	Constructor.
	 *
	 * @param 	pm	a PointCloudManager containing a colored PointCloud
	 *
	 * @param	region	a region to generate a texture for.
	 *
	 * @param	contours	All contours of the region.
	 *
	 */
	Texture( typename PointsetSurface<VertexT>::Ptr pm,   Region<VertexT, NormalT>* region,  vector<vector<HVertex*> > contours);

	/**
	 * @brief	computes texture coordinates corresponding to the give Vertex
	 *
	 * @param	v	the vertex to generate texture coordinates for
	 *
	 * @param	x	returns texture coordinates in x direction
	 *
	 * @param	y	returns texture coordinates in y direction
	 */
	void textureCoords(VertexT v, float &x, float &y);

	/**
	 *	@brief	Writes the texture to a file
	 */
	void save();

	/**
	 * Destructor.
	 */
	virtual ~Texture();

	///The pixel size determines the resolution
	static float m_texelSize;

private:
	struct ColorT{
		unsigned char r, g, b;
	};

	///The region belonging to the texture
	Region<VertexT, NormalT>* m_region;

	///The texture data
	ColorT** m_data;

	///The coordinate system of the texture plane
	NormalT best_v1, best_v2;

	///A point in the texture plane
	VertexT p;

	///The bounding box of the texture plane
	float best_a_min, best_b_min, best_a_max, best_b_max;

	///The dimensions of the texture
	int m_sizeX, m_sizeY;

};

}

#include "Texture.tcc"

#endif /* TEXTURE_HPP_ */
