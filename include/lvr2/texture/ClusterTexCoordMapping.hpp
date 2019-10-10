/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include "lvr2/geometry/Handles.hpp"

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
    TexCoords(float u = 0, float v = 0) : u(u), v(v) {}

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
    array<boost::optional<pair<ClusterHandle, TexCoords>>, 100> m_mapping;
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
        return TexCoords();
    }



};

} // namespace lvr2

#endif /* LVR2_TEXTURE_CLUSTER_TEXCOORD_MAPPING_H_ */
