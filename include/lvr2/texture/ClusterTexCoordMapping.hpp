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

/**
 * @struct TexCoords
 * @brief Texture coordinates
 */
struct TexCoords
{
    /// u-coordinate
    float u;
    /// v-coordinate
    float v;

    /**
     * @brief Constructor
     */
    TexCoords(float u, float v) : u(u), v(v) {}

};

/**
 * @class ClusterTexCoordMapping
 * @brief Mapping of clusters to texture coordinates for a single vertex
 *
 * Each vertex can be contained in several clusters, for each cluster there will be one entry in the array. Each entry
 * is a pair that consists of a cluster handle and texture coordinates.
 * This implementation assumes that a vertex is not contained in more than 100 clusters.
 */
class ClusterTexCoordMapping
{
private:
    /// The mapping of cluster handles to texture coordinates
    array<optional<pair<ClusterHandle, TexCoords>>, 100> m_mapping;
    /// The number of stored pairs
    size_t m_len;

public:
    /**
     * @brief Constructor
     */
    ClusterTexCoordMapping() : m_len(0) {}

    /**
     * @brief Adds an entry to the mapping
     *
     * @param handle The cluster handle
     * @param tex The texture coordinates
     */
    inline void push(ClusterHandle handle, TexCoords tex)
    {
        if (m_len == m_mapping.size())
        {
            cout << "Error: Overflow in ClusterTexCoordMapping" << endl;
        }
        else
        {
            m_mapping[m_len] = std::make_pair(handle, tex);
            m_len++;
        }
    }

    /**
     * @brief Returns the texture coordinates to a given cluster handle
     *
     * @param clusterH The cluster handle
     *
     * @return The texture coordinates
     */
    inline TexCoords getTexCoords(ClusterHandle clusterH) const
    {
        for (size_t i = 0; i < m_len; i++)
        {
            if (m_mapping[i]->first == clusterH)
            {
                return m_mapping[i]->second;
            }
        }
    }



};

} // namespace lvr2

#endif /* LVR2_TEXTURE_CLUSTER_TEXCOORD_MAPPING_H_ */
