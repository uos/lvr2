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
 * FastReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef FastReconstruction_H_
#define FastReconstruction_H_

#include "geometry/Vertex.hpp"
#include "geometry/BoundingBox.hpp"
#include "reconstruction/PointsetMeshGenerator.hpp"
#include "reconstruction/LocalApproximation.hpp"
#include "reconstruction/FastBox.hpp"
#include "reconstruction/TetraederBox.hpp"
#include "reconstruction/BilinearFastBox.hpp"
#include "reconstruction/QueryPoint.hpp"
#include "reconstruction/PointsetSurface.hpp"

#include "HashGrid.hpp"

/*#if _MSC_VER
#include <hash_map>
using namespace stdext;
#else
#include <ext/hash_map>
using namespace __gnu_cxx;
#endif */

#include <unordered_map>
using std::unordered_map;

namespace lvr
{

template<typename VertexT, typename NormalT>
class FastReconstructionBase
{
public:
    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<VertexT, NormalT> &mesh) = 0;
};

/**
 * @brief A surface reconstruction object that implements the standard
 *        marching cubes algorithm using a hashed grid structure for
 *        parallel computation.
 */
template<typename VertexT, typename NormalT, typename BoxT>
class FastReconstruction
{
public:

    /**
     * @brief Constructor.
     *
     * @param grid	A HashGrid instance on which the reconstruction is performed.
     */
    FastReconstruction(HashGrid<VertexT, BoxT>* grid);


    /**
     * @brief Destructor.
     */
    virtual ~FastReconstruction() {};

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<VertexT, NormalT> &mesh);

private:

    HashGrid<VertexT, BoxT>*		m_grid;
};


} // namespace lvr


#include "FastReconstruction.tcc"

#endif /* FastReconstruction_H_ */
