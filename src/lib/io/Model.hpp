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
 * Model.h
 *
 *  @date 27.10.2011
 *  @author Thomas Wiemann
 */

#ifndef MODEL_H_
#define MODEL_H_

#include "io/MeshBuffer.hpp"
#include "io/PointBuffer.hpp"

#include <algorithm>

typedef unsigned int uint;

namespace lssr
{

class Model
{
public:
    Model() : m_pointCloud(0), m_mesh(0) {};
    virtual ~Model() {};

    PointBuffer*        m_pointCloud;
    MeshBuffer*         m_mesh;
};

} // namespace lssr

#endif /* MODEL_H_ */
