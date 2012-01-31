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

/**
 * PlanarFastBox.hpp
 *
 *  @date 27.01.2012
 *  @author Thomas Wiemann
 */

#ifndef PLANARFASTBOX_HPP_
#define PLANARFASTBOX_HPP_

#include "FastBox.hpp"
#include "TetraederBox.hpp"

namespace lssr
{

/**
 * @brief A volume representation used by the standard Marching Cubes
 *        implementation.
 */
template<typename VertexT, typename NormalT>
class PlanarFastBox : public TetraederBox<VertexT, NormalT>
{
public:

    /**
     * @brief Constructs a new box at the given center point defined
     *        by the used \ref{m_voxelsize}.
     */
    PlanarFastBox(VertexT &center);

    /**
     * @brief Destructor.
     */
    virtual ~PlanarFastBox() {};

    virtual void getSurface(
            BaseMesh<VertexT, NormalT> &mesh,
            vector<QueryPoint<VertexT> > &query_points,
            uint &globalIndex);


};



} /* namespace lssr */

#include "PlanarFastBox.tcc"
#endif /* PLANARFASTBOX_HPP_ */
