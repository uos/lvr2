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
 * FinalizeAlgorithms.hpp
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

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseMesh.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/attrmaps/AttrMaps.hpp>
#include <lvr2/algorithm/ColorAlgorithms.hpp>
#include <lvr2/util/ClusterBiMap.hpp>
#include <lvr2/texture/Texture.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/algorithm/ClusterPainter.hpp>

namespace lvr2
{


// Forward declaration
template<typename BaseVecT>
class MaterializerResult;

/**
 * Algorithm that converts a BaseMesh into a MeshBuffer while maintaining the original graph structure. This means
 * that no duplicate vertices will be created and therefor no textures can be generated.
 */
template<typename BaseVecT>
class FinalizeAlgorithm
{
private:
    optional<const VertexMap<Rgb8Color>&> m_colorData;
    optional<const VertexMap<Normal<BaseVecT>>&> m_normalData;

public:
    FinalizeAlgorithm() {};

    /**
     * Converts the given BaseMesh into a MeshBuffer and adds normal and color data if set
     *
     * @param mesh the mesh to convert
     * @return the generated buffer
     */
    boost::shared_ptr<MeshBuffer<BaseVecT>> apply(const BaseMesh<BaseVecT>& mesh);

    /**
     * Sets vertex colors for the apply method. This has to be done before apply is called.
     *
     * @param colorData color values for all vertices in the mesh which will be passed to apply
     */
    void setColorData(const VertexMap<Rgb8Color>& colorData);

    /**
     * Sets vertex normals for the apply method. This has to be done before apply is called.
     *
     * @param normalData normals for all vertices in the mesh which will be passed to apply
     */
    void setNormalData(const VertexMap<Normal<BaseVecT>>& normalData);
};

/**
 * Algorithm that converts a BaseMesh into a MeshBuffer while destroying the original graph structure. This means
 * that duplicate vertices will be added to the mesh buffer so that textures can be generated.
 */
template<typename BaseVecT>
class ClusterFlatteningFinalizer
{
public:
    /**
     * Constructor for the finalizer
     *
     * @param cluster a map which maps all faces to clusters and vice versa
     */
    ClusterFlatteningFinalizer(const ClusterBiMap<FaceHandle>& cluster);

    /**
     * Sets vertex normals for the apply method. This has to be done before apply is called.
     *
     * @param normals normals for all vertices in the mesh which will be passed to apply
     */
    void setVertexNormals(const VertexMap<Normal<BaseVecT>>& normals);

    /**
     * Sets color data for all clusters for the apply method. This has to be done before apply is called.
     *
     * @param colors color data for all clusters in the mesh which will be passed to apply
     */
    void setClusterColors(const ClusterMap<Rgb8Color>& colors);

    /**
     * Sets vertex colors for the apply method. This has to be done before apply is called.
     *
     * @param vertexColors color values for all vertices in the mesh which will be passed to apply
     */
    void setVertexColors(const VertexMap<Rgb8Color>& vertexColors);

    /**
     * Sets the materializer result for the apply method. This has to be done before apply is called.
     *
     * @param materializerResult the result of the materializer that was run on the mesh which will be passed to apply
     */
    void setMaterializerResult(const MaterializerResult<BaseVecT>& materializerResult);

    /**
     * Converts the given BaseMesh into a MeshBuffer and adds further data (e.g. colors, normals) if set
     *
     * @param mesh the mesh to convert
     * @return the resulting mesh buffer
     */
    boost::shared_ptr<MeshBuffer<BaseVecT>> apply(const BaseMesh<BaseVecT>& mesh);

private:

    // Clusters of mesh (mandatory)
    const ClusterBiMap<FaceHandle>& m_cluster;

    // Normals (optional)
    optional<const VertexMap<Normal<BaseVecT>>&> m_vertexNormals;

    // Basic colors
    // Cluster colors will color each vertex in the color of its corresponding cluster
    // These have lower priority when cluster colors, as only one mode can be used
    optional<const ClusterMap<Rgb8Color>&> m_clusterColors;

    // Vertex colors will color each vertex individually
    // These have a higher priority than cluster colors
    optional<const VertexMap<Rgb8Color>&> m_vertexColors;

    // Materials and textures
    optional<const MaterializerResult<BaseVecT>&> m_materializerResult;
};

} // namespace lvr2

#include <lvr2/algorithm/FinalizeAlgorithms.tcc>

#endif /* LVR2_ALGORITHM_FINALIZEALGORITHM_H_ */
