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
* Materializer.hpp
*
*  @date 17.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#ifndef LVR2_ALGORITHM_MATERIALIZER_H_
#define LVR2_ALGORITHM_MATERIALIZER_H_

#include <boost/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/optional.hpp>

#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/texture/ClusterTexCoordMapping.hpp>
#include <lvr2/texture/Texture.hpp>
#include <lvr2/util/ClusterBiMap.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/geometry/BoundingRectangle.hpp>
#include <lvr2/algorithm/Texturizer.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>


#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>
#include <unordered_map>

namespace lvr2
{

template<typename BaseVecT>
struct MaterializerResult
{
    DenseClusterMap<Material> m_clusterMaterials;
    optional<StableVector<TextureHandle, Texture<BaseVecT>>> m_textures;
    optional<SparseVertexMap<ClusterTexCoordMapping>> m_vertexTexCoords;
    optional<std::unordered_map<BaseVecT, std::vector<float>>> m_keypoints;

    MaterializerResult(
        DenseClusterMap<Material> clusterMaterials
    )   :
        m_clusterMaterials(clusterMaterials)
    {
    }

    MaterializerResult(
        DenseClusterMap<Material> clusterMaterials,
        StableVector<TextureHandle, Texture<BaseVecT>> textures,
        SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords,
        std::unordered_map<BaseVecT, std::vector<float>> keypoints
    ) :
        m_clusterMaterials(clusterMaterials),
        m_textures(textures),
        m_vertexTexCoords(vertexTexCoords),
        m_keypoints(keypoints)
    {
    }

};


template<typename BaseVecT>
class Materializer
{

public:

    Materializer(
        const BaseMesh<BaseVecT>& mesh,
        const ClusterBiMap<FaceHandle>& cluster,
        const FaceMap<Normal<BaseVecT>>& normals,
        const PointsetSurface<BaseVecT>& surface
    );

    void setTexturizer(Texturizer<BaseVecT> texturizer);

    MaterializerResult<BaseVecT> generateMaterials();

    void saveTextures();

private:

    const BaseMesh<BaseVecT>& m_mesh;
    const ClusterBiMap<FaceHandle>& m_cluster;
    const FaceMap<Normal<BaseVecT>>& m_normals;
    const PointsetSurface<BaseVecT>& m_surface;

    optional<Texturizer<BaseVecT>&> m_texturizer;

};

} // namespace lvr2

#include <lvr2/algorithm/Materializer.tcc>

#endif /* LVR2_ALGORITHM_MATERIALIZER_H_ */
