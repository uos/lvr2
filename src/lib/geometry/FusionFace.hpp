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
 * FusionFace.hpp
 *
 *  @date 14.07.2013
 *  @author Henning Deeken (hdeeken@uos.de)
 *  @author Ann-Katrin Häuser (ahaeuser@uos.de)
 */

#ifndef FUSIONFACE_H_
#define FUSIONFACE_H_

#include <vector>
#include <set>

using namespace std;

#include "FusionVertex.hpp"
#include "Normal.hpp"

#include <CGAL/AABB_triangle_primitive.h>

namespace lssr
{

template<typename VertexT, typename NormalT> class FusionVertex;

/**
 * @brief A face in a fusion mesh.
 *
 */
template<typename VertexT, typename NormalT> class FusionFace
{
public:

    /**
     * @brief   Constructs an empty face
     */
	FusionFace();

	/**
	 * @brief Destructor
	 */
	~FusionFace();

	/**
	 * @brief   Copy constructor
	 *
	 * @param o The mesh to copy
	 */
	//FusionFace(const FusionFace<VertexT, NormalT> &o);

	FusionVertex <VertexT, NormalT>* 	vertices[3];

	/// A three pointers to the face vertices
	int 							m_index[3];

	/// The face normal
	NormalT							m_normal;
	
	// all normal related functionality has been omitted for reasons of simplicity. if necessary see HalfEdeFace.hpp.

	/// Indicator if face is part of the mesh boundary
	bool 							m_is_border_face;
	
	int r;
	int g;
	int b;

};

}

#include "FusionFace.cpp"

#endif /* FUSIONFACE_H_ */
