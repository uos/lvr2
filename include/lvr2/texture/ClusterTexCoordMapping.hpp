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
* Texturizer.tcc
*
*  @date 23.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#ifndef LVR2_TEXTURE_CLUSTER_TEXCOORD_MAPPING_H_
#define LVR2_TEXTURE_CLUSTER_TEXCOORD_MAPPING_H_

#include <lvr2/geometry/Handles.hpp>

#include <vector>
#include <utility>

using std::vector;
using std::pair;

namespace lvr2
{

struct TexCoords
{
    float u, v;

    TexCoords(float u, float v) : u(u), v(v) {}

};

class ClusterTexCoordMapping
{
private:
    // vector<pair<ClusterHandle, TexCoords>> mapping;
    array<optional<pair<ClusterHandle, TexCoords>>, 100> m_mapping;
    size_t m_len;

public:
    ClusterTexCoordMapping() : m_len(0) {}

    void push(ClusterHandle handle, TexCoords tex);
    TexCoords getTexCoords(ClusterHandle clusterH) const;
};

} // namespace lvr2

#include "ClusterTexCoordMapping.tcc"

#endif /* LVR2_TEXTURE_CLUSTER_TEXCOORD_MAPPING_H_ */
