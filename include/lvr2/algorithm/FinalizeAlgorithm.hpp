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
 * FinalizeAlgorithm.hpp
 *
 *  @date 13.06.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_ALGORITHM_FINALIZEALGORITHM_H_
#define LVR2_ALGORITHM_FINALIZEALGORITHM_H_

#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/optional.hpp>

using boost::optional;

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr/io/MeshBuffer.hpp>
#include "ClusterPainter.hpp"
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>
#include <lvr2/util/ClusterBiMap.hpp>

namespace lvr2
{

/**
 * @brief
 */
template<typename BaseVecT>
class FinalizeAlgorithm
{
private:
    optional<const VertexMap<Rgb8Color>&> m_colorData;
    optional<const VertexMap<Normal<BaseVecT>>&> m_normalData;

public:
    FinalizeAlgorithm() {};

    boost::shared_ptr<lvr::MeshBuffer> apply(const BaseMesh<BaseVecT>& mesh);
    void setColorData(const VertexMap<Rgb8Color>& colorData);
    void setNormalData(const VertexMap<Normal<BaseVecT>>& normalData);
};

template<typename BaseVecT>
class ClusterFlatteningFinalizer
{
public:
    ClusterFlatteningFinalizer(const ClusterBiMap<FaceHandle>& cluster);

    void setVertexNormals(const VertexMap<Normal<BaseVecT>>& normals);
    void setClusterColors(const ClusterMap<Rgb8Color>& colors);

    boost::shared_ptr<lvr::MeshBuffer> apply(const BaseMesh<BaseVecT>& mesh);

private:
    const ClusterBiMap<FaceHandle>& m_cluster;
    optional<const ClusterMap<Rgb8Color>&> m_clusterColors;
    optional<const VertexMap<Normal<BaseVecT>>&> m_vertexNormals;
};

} // namespace lvr2

#include <lvr2/algorithm/FinalizeAlgorithm.tcc>

#endif /* LVR2_ALGORITHM_FINALIZEALGORITHM_H_ */
