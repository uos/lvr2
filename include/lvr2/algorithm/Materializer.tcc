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
* Materializer.tcc
*
*  @date 17.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

namespace lvr2
{

template<typename BaseVecT>
Materializer<BaseVecT>::Materializer(
    const BaseMesh<BaseVecT>& mesh,
    const ClusterBiMap<FaceHandle>& cluster,
    const FaceMap<Normal<BaseVecT>>& normals,
    const PointsetSurface<BaseVecT>& surface
) :
    m_mesh(mesh),
    m_cluster(cluster),
    m_normals(normals),
    m_surface(surface)
{
}

template<typename BaseVecT>
void Materializer<BaseVecT>::setTexturizer(Texturizer<BaseVecT> texturizer)
{
    m_texturizer = texturizer;
}

template<typename BaseVecT>
MaterializerResult<BaseVecT> Materializer<BaseVecT>::generateMaterials()
{
    string msg = lvr::timestamp.getElapsedTime() + "Generating materials ";
    lvr::ProgressBar progress(m_cluster.numCluster(), msg);

    // Prepare result
    DenseFaceMap<Material> faceMaterials;
    SparseVertexMap<ClusterTexCoordMapping> vertexTexCoords;

    // Counters used for texturizing
    int numClustersTooSmall = 0;
    int numClustersTooLarge = 0;
    int textureCount = 0;
    int clusterCount = 0;

    // For all clusters ...
    for (auto clusterH : m_cluster)
    {
        ++progress;

        // Get number of faces in cluster
        const Cluster<FaceHandle>& cluster = m_cluster.getCluster(clusterH);
        int numFacesInCluster = cluster.handles.size();


        if (!m_texturizer
            || (numFacesInCluster < m_texturizer.get().m_texMinClusterSize
                && m_texturizer.get().m_texMinClusterSize != 0)
            || (numFacesInCluster > m_texturizer.get().m_texMaxClusterSize
                && m_texturizer.get().m_texMaxClusterSize != 0)
        )
        {
            // No textures, or using textures and texture is too small/large
            // Generate plain color material
            // (texMin/MaxClustersize = 0 means: no limit)

            if (m_texturizer)
            {
                // If using textures, count wether this cluster was too small or too large
                if (numFacesInCluster < m_texturizer.get().m_texMinClusterSize)
                {
                    numClustersTooSmall++;
                }
                else if (numFacesInCluster > m_texturizer.get().m_texMaxClusterSize)
                {
                    numClustersTooLarge++;
                }
            }


            // For each face ...
            for (auto faceH : cluster.handles)
            {
                // Calculate color of centroid
                Rgb8Color color = calcColorForFaceCentroid(m_mesh, m_surface, faceH);
                // Create material and save in face map
                Material material;
                material.m_color = color;
                faceMaterials.insert(faceH, material);
            }

        }
        else if (m_texturizer) // FIXME: sollte unnötig sein, cond ist nur zum testen hier
        {
            // Textures

            // Contour
            std::vector<VertexHandle> contour = calculateClusterContourVertices(
                clusterH,
                m_mesh,
                m_cluster
            );

            // Counding rectangle
            BoundingRectangle<BaseVecT> boundintRect = calculateBoundingRectangle(
                contour,
                m_mesh,
                cluster,
                m_normals,
                m_texturizer.get().m_texelSize,
                clusterH
            );

            // Create texture
            TextureHandle texH = m_texturizer.get().generateTexture(
                textureCount,
                m_surface,
                boundintRect
            );

            // Create material and insert in face map
            Material material;
            material.m_texture = texH;

            // Calculate tex coords
            // Insert material into face map for each face
            // Find unique vertices in cluster
            std::unordered_set<VertexHandle> verticesOfCluster;
            for (auto faceH : cluster.handles)
            {
                faceMaterials.insert(faceH, material);

                for (auto vertexH : m_mesh.getVerticesOfFace(faceH))
                {
                    verticesOfCluster.insert(vertexH);
                    // (doesnt insert duplicate vertices)
                }
            }

            for (auto vertexH : verticesOfCluster)
            {
                // Calculate tex coords
                TexCoords texCoords = m_texturizer.get().calculateTexCoords(
                    texH,
                    boundintRect,
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
            textureCount++;
        }
    }

    cout << endl;

    // Write result
    if (m_texturizer)
    {

        cout << lvr::timestamp << "Skipped " << (numClustersTooSmall+numClustersTooLarge)
        << " clusters while generating textures" << endl;

        cout << lvr::timestamp << "(" << numClustersTooSmall << " below threshold, "
        << numClustersTooLarge << " above limit, " << m_cluster.numCluster() << " total)" << endl;

        cout << lvr::timestamp << "Generated " << textureCount << " textures" << endl;

        return MaterializerResult<BaseVecT>(
            faceMaterials,
            m_texturizer.get().getTextures(),
            vertexTexCoords
        );
    }
    else
    {
        return MaterializerResult<BaseVecT>(faceMaterials);
    }

}



} // namespace lvr2
