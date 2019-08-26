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
 * ClusterPainter.hpp
 *
 *  @date 18.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_CLUSTERPAINTER_H_
#define LVR2_ALGORITHM_CLUSTERPAINTER_H_

#include <cstdint>
#include <array>

using std::array;

#include "lvr2/geometry/Handles.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"

namespace lvr2
{

/**
 * @brief Algorithm which generates the same color for all vertices, which are in the same cluster.
 */
class ClusterPainter
{
public:
    ClusterPainter(const ClusterBiMap<FaceHandle>& clusterBiMap) : m_clusterBiMap(clusterBiMap) {};

    /**
     * @brief Assign a pseudo-color to each cluster.
     *
     * The color is deterministically determined by the cluster id.
     */
    template<typename BaseVecT>
    DenseClusterMap<Rgb8Color> simpsons(const BaseMesh<BaseVecT>& mesh) const;

private:
    ClusterBiMap<FaceHandle> m_clusterBiMap;
    inline Rgb8Color getSimpsonColorForIdx(size_t idx) const
    {
      return {
          static_cast<uint8_t>(fabs(cos(idx)) * 255),
          static_cast<uint8_t>(fabs(sin(idx * 30)) * 255),
          static_cast<uint8_t>(fabs(sin(idx * 2)) * 255)
      };
    }
};

} // namespace lvr2

#include "lvr2/algorithm/ClusterPainter.tcc"

#endif /* LVR2_ALGORITHM_CLUSTERPAINTER_H_ */
