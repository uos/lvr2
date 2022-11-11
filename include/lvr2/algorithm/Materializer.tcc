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
* Materializer.tcc
*
*  @date 17.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#include "lvr2/algorithm/ClusterAlgorithms.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/util/Logging.hpp"
#include <opencv2/features2d.hpp>


namespace lvr2
{

template<typename BaseVecT>
Materializer<BaseVecT>::Materializer(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& cluster,
    const FaceMap<Normal<typename BaseVecT::CoordType>>& normals,
    const PointsetSurface<BaseVecT>& surface
) :
    m_mesh(mesh),
    m_cluster(cluster),
    m_normals(normals),
    m_surface(surface)
{
}

template<typename BaseVecT>
void Materializer<BaseVecT>::setTexturizer(std::shared_ptr<Texturizer<BaseVecT>> texturizer)
{
    m_texturizers = TexturizerPtrVec();
    m_texturizers->push_back(texturizer);
}

template<typename BaseVecT>
void Materializer<BaseVecT>::addTexturizer(std::shared_ptr<Texturizer<BaseVecT>> texturizer)
{
    if (!m_texturizers)
    {
        setTexturizer(texturizer);
    }
    else
    {
        m_texturizers->push_back(texturizer);
    }
}

template<typename BaseVecT>
void Materializer<BaseVecT>::saveTextures()
{
    if (m_texturizers)
    {
        for (auto texturizer_ptr: *m_texturizers)
        {
            texturizer_ptr->saveTextures();
        }
    }
}

template<typename BaseVecT>
MaterializerResult<BaseVecT> Materializer<BaseVecT>::generateMaterials()
{
    lvr2::Monitor monitor(lvr2::LogLevel::info, "Generating materials", m_cluster.numCluster());

    // Prepare result
    DenseClusterMap<Material> clusterMaterials;
    SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords;

    std::unordered_map<BaseVecT, std::vector<float>> keypoints_map;

    // Counters used for texturizing
    int numClustersTooSmall = 0;
    int numClustersTooLarge = 0;
    int textureCount = 0;
    int clusterCount = 0;

    // For all clusters ...
    for (auto clusterH : m_cluster)
    {
        ++monitor;


        // Get number of faces in cluster
        const Cluster<FaceHandle>& cluster = m_cluster.getCluster(clusterH);
        int numFacesInCluster = cluster.handles.size();

        // Texturizers have to have the same settings for minClusterSize/maxClusterSize
        if (!m_texturizers
            || (m_texturizers && numFacesInCluster < m_texturizers.get()[0]->m_texMinClusterSize
                && m_texturizers.get()[0]->m_texMinClusterSize != 0)
            || (m_texturizers && numFacesInCluster > m_texturizers.get()[0]->m_texMaxClusterSize
                && m_texturizers.get()[0]->m_texMaxClusterSize != 0)
        )
        {
            // No textures, or using textures and texture is too small/large
            // Generate plain color material
            // (texMin/MaxClustersize = 0 means: no limit)

            if (m_texturizers)
            {
                // If using textures, count whether this cluster was too small or too large
                if (numFacesInCluster < m_texturizers.get()[0]->m_texMinClusterSize)
                {
                    numClustersTooSmall++;
                }
                else if (numFacesInCluster > m_texturizers.get()[0]->m_texMaxClusterSize)
                {
                    numClustersTooLarge++;
                }
            }


            // Calculate (a sorta-kinda not really) median value
            std::map<RGB8Color, int> colorMap;
            int maxColorCount = 0;
            RGB8Color mostUsedColor;

            // For each face ...
            for (auto faceH : cluster.handles)
            {
                // Calculate color of centroid
                RGB8Color color = calcColorForFaceCentroid(m_mesh, m_surface, faceH);
                if (colorMap.count(color))
                {
                    colorMap[color]++;
                }
                else
                {
                    colorMap[color] = 1;
                }
                if (colorMap[color] > maxColorCount)
                {
                    mostUsedColor = color;
                }
            }

            // Create material and save in map
            Material material;
            std::array<unsigned char, 3> arr = {
                static_cast<uint8_t>(mostUsedColor[0]),
                static_cast<uint8_t>(mostUsedColor[1]),
                static_cast<uint8_t>(mostUsedColor[2])
            };

            material.m_color =  std::move(arr);
            clusterMaterials.insert(clusterH, material);

        }
        else
        {
            // Textures

            // Contour
            std::vector<VertexHandle> contour = calculateClusterContourVertices(
                clusterH,
                m_mesh,
                m_cluster
            );

            // Bounding rectangle
            BoundingRectangle<typename BaseVecT::CoordType> boundingRect = calculateBoundingRectangle(
                contour,
                m_mesh,
                cluster,
                m_normals,
                m_texturizers.get()[0]->m_texelSize,
                clusterH
            );

            // The texture handle to the texture from the first texturizer
            // only used for coordinate mapping
            boost::optional<TextureHandle> first_opt;
            Material::LayerMap layers;
            // Use each texturizer once for this cluster
            for (auto texturizer: *m_texturizers)
            {
                // Create texture
                TextureHandle texH = texturizer->generateTexture(
                textureCount,
                m_surface,
                boundingRect,
                clusterH
                );

                if (!first_opt) first_opt = texH;
                // Add layer create handle from global textureCount 
                // (The combined list will be created at the end)
                layers.insert(
                    std::pair(
                    texturizer->getTexture(texH).m_layerName,
                    TextureHandle(textureCount))
                    );

                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
                texturizer->findKeyPointsInTexture(texH,
                    boundingRect, detector, keypoints, descriptors);
                std::vector<BaseVecT> features3d =
                    texturizer->keypoints23d(keypoints, boundingRect, texH);

                // Transform descriptor from matrix row to float vector
                for (unsigned int row = 0; row < features3d.size(); ++row)
                {
                    keypoints_map[features3d[row]] =
                        std::vector<float>(descriptors.ptr(row), descriptors.ptr(row) + descriptors.cols);
                }
                textureCount++;
            }
            
            // Create material with default color and insert into face map
            Material material;
            material.m_texture = layers.begin()->second;
            material.m_layers = layers;

            std::array<unsigned char, 3> arr = {255, 255, 255};

            material.m_color = std::move(arr);            
            clusterMaterials.insert(clusterH, material);

            // Calculate tex coords
            // Insert material into face map for each face
            // Find unique vertices in cluster
            std::unordered_set<VertexHandle> verticesOfCluster;
            for (auto faceH : cluster.handles)
            {
                for (auto vertexH : m_mesh.getVerticesOfFace(faceH))
                {
                    verticesOfCluster.insert(vertexH);
                    // (doesnt insert duplicate vertices)
                }
            }
            // For each unique vertex in this cluster
            for (auto vertexH : verticesOfCluster)
            {
                // Use the first texturizer for mapping coordinates
                auto texturizer = m_texturizers.get()[0];
                // Calculate tex coords
                TexCoords texCoords = texturizer->calculateTexCoords(
                    *first_opt,
                    boundingRect,
                    m_mesh.getVertexPosition(vertexH)
                );

                // Insert into result map
                if (vertexTexCoords.get(vertexH))
                {
                    vertexTexCoords.get(vertexH).get().push(clusterH, texCoords);
                }
                else
                {
                    ClusterTexCoordMapping mapping;
                    mapping.push(clusterH, texCoords);
                    vertexTexCoords.insert(vertexH, mapping);
                }
            }
        }   
    }

    // Write result
    // TODO: Merge texturizer results
    if (m_texturizers)
    {

        lvr2::logout::get() << lvr2::info << "Skipped " << (numClustersTooSmall+numClustersTooLarge)
            << " clusters while generating textures" << lvr2::endl;

        lvr2::logout::get() << lvr2::info 
            << "(" << numClustersTooSmall << " below threshold, "
            << numClustersTooLarge << " above limit, " 
            << m_cluster.numCluster() << " total)" << lvr2::endl;

        lvr2::logout::get() << lvr2::info << 
            "Generated " << textureCount << " textures" << lvr2::endl;

        // Holds all textures in the order determined by Texture::m_index
        StableVector<TextureHandle, Texture> combined_textures;
        combined_textures.increaseSize(TextureHandle(textureCount));

        for (auto texturizer: *m_texturizers)
        {
            for (const auto texH: texturizer->getTextures())
            {
                auto tex = texturizer->getTextures()[texH];
                combined_textures.set(TextureHandle(tex.m_index), std::move(tex));
            }
        }

        return MaterializerResult<BaseVecT>(
            clusterMaterials,
            std::move(combined_textures),
            vertexTexCoords,
            keypoints_map
        );
    }
    else
    {
        return MaterializerResult<BaseVecT>(clusterMaterials);
    }

}



} // namespace lvr2
