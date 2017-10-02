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
 * MeshGenerator.hpp
 *
 *  @date 25.01.2012
 *  @author Thomas Wiemann
 */

#ifndef _LVR2_RECONSTRUCTION_MESHGENERATOR_H_
#define _LVR2_RECONSTRUCTION_MESHGENERATOR_H_

#include <lvr2/geometry/BaseMesh.hpp>

namespace lvr2
{

/**
 * @brief Interface class for mesh based reconstruction algorithms
 */
template<typename BaseVecT>
class MeshGenerator
{
public:

    /**
     * @brief Generates a triangle mesh representation of the current
     *        point set.
     *
     * @param mesh      A surface representation of the current point
     *                  set.
     */
    virtual void getMesh(BaseMesh<BaseVecT>& mesh) = 0;

};


} // namepsace lvr2

#endif /* _LVR2_RECONSTRUCTION_MESHGENERATOR_H_ */