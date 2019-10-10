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



#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/geometry/Normal.hpp"
#include "lvr2/attrmaps/AttrMaps.hpp"
#include "lvr2/algorithm/ColorAlgorithms.hpp"
#include "lvr2/util/ClusterBiMap.hpp"
#include "lvr2/texture/Texture.hpp"
#include "lvr2/texture/Material.hpp"
#include "lvr2/algorithm/ClusterPainter.hpp"

#include "lvr2/io/ObjIO.hpp"

namespace lvr2
{


// Forward declaration
template<typename BaseVecT>
class MaterializerResult;

/**
 * Algorithm that converts a BaseMesh into a MeshBuffer while maintaining the original graph structure. This means
 * that no duplicate vertices will be created and therefore no textures can be generated.
 */
template<typename BaseVecT>
class SimpleFinalizer
{
private:
    boost::optional<const VertexMap<Rgb8Color>&> m_colorData;
    boost::optional<const VertexMap<Normal<typename BaseVecT::CoordType>>&> m_normalData;

public:
    SimpleFinalizer() {};

    /**
     * Converts the given BaseMesh into a MeshBuffer and adds normal and color data if set
     *
     * @param mesh the mesh to convert
     * @return the generated buffer
     */
    MeshBufferPtr apply(const BaseMesh<BaseVecT>& mesh);

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
    void setNormalData(const VertexMap<Normal<typename BaseVecT::CoordType>>& normalData);
};

/**
 * Algorithm that converts a BaseMesh into a MeshBuffer while destroying the original graph structure. This means
 * that duplicate vertices will be added to the mesh buffer so that textures can be generated.
 */
template<typename BaseVecT>
class TextureFinalizer
{
public:
    /**
     * Constructor for the finalizer
     *
     * @param cluster a map which maps all faces to clusters and vice versa
     */
    TextureFinalizer(const ClusterBiMap<FaceHandle>& cluster);

    /**
     * Sets vertex normals for the apply method. This has to be done before apply is called.
     *
     * @param normals normals for all vertices in the mesh which will be passed to apply
     */
    void setVertexNormals(const VertexMap<Normal<typename BaseVecT::CoordType>>& normals);

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
    MeshBufferPtr apply(const BaseMesh<BaseVecT>& mesh);

private:

    // Clusters of mesh (mandatory)
    const ClusterBiMap<FaceHandle>& m_cluster;

    // Normals (optional)
    boost::optional<const VertexMap<Normal<typename BaseVecT::CoordType>>&> m_vertexNormals;

    // Basic colors
    // Cluster colors will color each vertex in the color of its corresponding cluster
    // These have lower priority when cluster colors, as only one mode can be used
    boost::optional<const ClusterMap<Rgb8Color>&> m_clusterColors;

    // Vertex colors will color each vertex individually
    // These have a higher priority than cluster colors
    boost::optional<const VertexMap<Rgb8Color>&> m_vertexColors;

    // Materials and textures
    boost::optional<const MaterializerResult<BaseVecT>&> m_materializerResult;
};

} // namespace lvr2

#include "lvr2/algorithm/FinalizeAlgorithms.tcc"

#endif /* LVR2_ALGORITHM_FINALIZEALGORITHM_H_ */
