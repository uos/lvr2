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
*  @date 17.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/

#include "ClusterAlgorithm.hpp"


namespace lvr2
{

template<typename BaseVecT>
TexturizerResult<BaseVecT> generateTextures(
    float texelSize,
    int textureThreshold,
    int textureLimit,
    BaseMesh<BaseVecT>& mesh,
    ClusterBiMap<FaceHandle>& faceHandleClusterBiMap,
    PointsetSurfacePtr<BaseVecT> surface,
    const FaceMap<Normal<BaseVecT>>& normals
)
{

    string msg = lvr::timestamp.getElapsedTime() + "Generating textures ";
    lvr::ProgressBar progress(faceHandleClusterBiMap.numCluster(), msg);

    int numFacesThreshold = textureThreshold;
    int textureIndex = 0;

    int numClustersTooSmall = 0;
    int numClustersTooLarge = 0;

    TexturizerResult<BaseVecT> result;
    result.tcMap.reserve(mesh.numVertices());

    HalfEdgeMesh<BaseVecT> debugMesh;

    for (auto clusterH: faceHandleClusterBiMap)
    {
        ++progress;
        const Cluster<FaceHandle> cluster = faceHandleClusterBiMap.getCluster(clusterH);
        int numFacesInCluster = cluster.handles.size();

        // only create textures for clusters that are large enough
        if (numFacesInCluster >= textureThreshold && (numFacesInCluster < textureLimit || textureLimit < 1))
        // if (numFacesInCluster >= numFacesThreshold && numFacesInCluster < 200000)
        {
            // contour
            std::vector<VertexHandle> contour = calculateClusterContourVertices(clusterH, mesh, faceHandleClusterBiMap);

            // bounding rectangle
            BoundingRectangle<BaseVecT> br = calculateBoundingRectangle(contour, mesh, cluster, normals, texelSize, clusterH);

            // initial texture
            TextureToken<BaseVecT> texToken = generateTexture(br, surface, texelSize, textureIndex++);
            texToken.m_texture->save(texToken.m_textureIndex);

            // save textoken & texture
            result.texTokenClusterMap.insert(clusterH, texToken);
            result.textures.push_back(texToken.m_texture);

            // find unique vertices in cluster
            std::unordered_set<VertexHandle> verticesOfCluster;
            for (auto faceH : cluster.handles)
            {
                for (auto vertexH : mesh.getVerticesOfFace(faceH))
                {
                    verticesOfCluster.insert(vertexH);
                    // (doesnt insert duplicate vertices)
                }
            }

            for (auto vertexH : verticesOfCluster)
            {
                // Calculate texture coords
                // TODO: texturkoordinaten berechnen. aber ist das wirklich nötig?
                TexCoords texCoords = texToken.textureCoords(mesh.getVertexPosition(vertexH).asVector());

                if (result.tcMap.get(vertexH))
                {
                    result.tcMap[vertexH].push(clusterH, texCoords);
                }
                else
                {
                    ClusterTexCoordMapping tcMap;
                    tcMap.push(clusterH, texCoords);
                    result.tcMap.insert(vertexH, tcMap);
                }
            }

        }
        else if (numFacesInCluster < numFacesThreshold)
        {
            numClustersTooSmall++;
        }
        else if (textureLimit > 0 && numFacesInCluster > textureLimit)
        {
            numClustersTooLarge++;
        }
    }
    cout << endl;

    cout << lvr::timestamp << "Skipped " << (numClustersTooSmall+numClustersTooLarge)
    << " clusters while texturizing (" << numClustersTooSmall << " below threshold, "
    << numClustersTooLarge << " above limit, "
    << faceHandleClusterBiMap.numCluster() << " total)" << endl;

    cout << lvr::timestamp << "Generated " << result.textures.size() << " textures" << endl;

    return result;
}

template <typename BaseVecT>
TextureToken<BaseVecT> generateTexture(
    BoundingRectangle<BaseVecT>& boundingRect,
    PointsetSurfacePtr<BaseVecT> surface,
    float texelSize,
    int textureIndex
)
{
    // calculate the texture size
    unsigned short int sizeX = ceil((boundingRect.maxDistA - boundingRect.minDistA) / texelSize);
    unsigned short int sizeY = ceil((boundingRect.maxDistB - boundingRect.minDistB) / texelSize);

    // create texture
    Texture* texture = new Texture(sizeX, sizeY, 3, 1, 0);//, 0, 0, 0, 0, false, 0, 0);

    // create TextureToken
    TextureToken<BaseVecT> result = TextureToken<BaseVecT>(
        boundingRect.vec1,
        boundingRect.vec2,
        boundingRect.supportVector,
        boundingRect.minDistA,
        boundingRect.minDistB,
        texture,
        textureIndex,
        texelSize
    );

    int dataCounter = 0;

    for (int y = 0; y < sizeY; y++)
    {
        for (int x = 0; x < sizeX; x++)
        {
            std::vector<char> v;

            int k = 1; // k-nearest-neighbors

            vector<size_t> cv;

            Point<BaseVecT> currentPos =
                boundingRect.supportVector
                + boundingRect.vec1 * (x * texelSize + boundingRect.minDistA - texelSize / 2.0)
                + boundingRect.vec2 * (y * texelSize + boundingRect.minDistB - texelSize / 2.0);

            surface->searchTree().kSearch(currentPos, k, cv);

            uint8_t r = 0, g = 0, b = 0;

            for (size_t pointIdx : cv)
            {
                array<uint8_t,3> colors = *(surface->pointBuffer()->getRgbColor(pointIdx));
                r += colors[0];
                g += colors[1];
                b += colors[2];
            }

            r /= k;
            g /= k;
            b /= k;

            texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = r;
            texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = g;
            texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = b;
        }
    }

    return result;

}

} // namespace lvr2
