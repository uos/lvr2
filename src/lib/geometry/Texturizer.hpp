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
 * Texturizer.hpp
 *
 *  @date 03.05.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef TEXTURIZER_HPP_
#define TEXTURIZER_HPP_

#include "reconstruction/PointsetSurface.hpp"

namespace lssr {

/**
 * @brief	This class creates textures
 */
template<typename VertexT, typename NormalT>
class Texturizer {
public:


	/**
	 * @brief 	Constructor.
	 *
	 * @param 	pm	 A PointCloudManager containing a colored PointCloud
	 *
	 * @param	filename A texture package
	 */
	Texturizer( typename PointsetSurface<VertexT>::Ptr pm, String filename = "");

	/**
	 * @brief 	Creates or searches a texture for the region given by its' contour
	 *
	 * @param	contour	The contour of the region to create or find a texture for
	 *
	 * @return	A TextureToken containing the generated texture
	 *
	**/
	TextureToken<VertexT, NormalT>* texturizePlane(vector<VertexT> contour);

private:

	/**
	 * @brief 	Creates a texture for the region given by its' contour using the colored point cloud
	 *
	 * @param	contour	The contour of the region to create a texture for
	 *
	 * @return	A TextureToken containing the generated texture
	 *
	**/
	TextureToken<VertexT, NormalT>* createInitialTexture(vector<VertexT> contour);
	
	///The point set surface used to generate textures from the point cloud
	typename PointsetSurface<VertexT>::Ptr m_pm
};

}

#include "Texturizer.tcc"

#endif /* TEXTURIZER_HPP_ */
