/**
 * Copyright (c) 2019, University Osnabrück
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

/**
 * ChunkManager.cpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 * @author Marcel Wiegand
 * @author Raphael Marx
 */

#include "lvr2/algorithm/ChunkManager.hpp"

#include "lvr2/io/ModelFactory.hpp"

#include <cmath>
namespace
{
    template<typename T>
    struct VectorCapsule {
    private:
        std::shared_ptr<std::vector<T>> arr_;
    public:
        explicit VectorCapsule(std::vector<T>&& arr): arr_(std::make_shared<std::vector<T>>(std::move(arr))) {}
        ~VectorCapsule() = default;
        void operator()(void*) {}
    };
}

namespace lvr2
{

ChunkManager::ChunkManager(MeshBufferPtr mesh,
                           float chunksize,
                           float maxChunkOverlap,
                           std::string savePath)
    : m_chunkSize(chunksize)
{
    initBoundingBox(mesh);

    // compute number of chunks for each dimension
    m_amount.x = static_cast<std::size_t>(std::ceil(m_boundingBox.getXSize() / m_chunkSize));
    m_amount.y = static_cast<std::size_t>(std::ceil(m_boundingBox.getYSize() / m_chunkSize));
    m_amount.z = static_cast<std::size_t>(std::ceil(m_boundingBox.getZSize() / m_chunkSize));

    buildChunks(mesh, maxChunkOverlap, savePath);
}

MeshBufferPtr ChunkManager::extractArea(const BoundingBox<BaseVector<float>>& area)
{
    std::unordered_map<std::size_t, MeshBufferPtr> chunks;

    // find all required chunks
    // TODO: check if we need + 1
    std::size_t numChunks = 0;
    const BaseVector<float> maxSteps = (area.getMax() - area.getMin()) / m_chunkSize;
    for (std::size_t i = 0; i < maxSteps.x; ++i)
    {
        for (std::size_t j = 0; j < maxSteps.y; ++j)
        {
            for (std::size_t k = 0; k < maxSteps.z; ++k)
            {
                std::size_t cellIndex = getCellIndex(area.getMin() + BaseVector<float>(i, j, k) * m_chunkSize);
                std::cout << "Point:" << area.getMin() + BaseVector<float>(i, j, k) * m_chunkSize << std::endl;
                std::cout << "Cell Index: " << cellIndex << std::endl;

                auto it = m_hashGrid.find(cellIndex);
                if (it == m_hashGrid.end())
                {
                    std::cout << "continue" << std::endl;
                    continue;
                }
                // TODO: remove saving tmp chunks later
                ModelFactory::saveModel(lvr2::ModelPtr(new lvr2::Model(it->second)),
                                        "area/" + std::to_string(cellIndex) + ".ply");
                chunks.insert(*it);
                std::cout << "numChunks " << numChunks++ << std::endl;
            }
        }
    }
    std::cout << "Extracted " << chunks.size() << " Chunks" << std::endl;

    // TODO: concat chunks
    std::vector<float> areaDuplicateVertices;
    std::vector<std::unordered_map<std::size_t, std::size_t> > areaVertexIndices;
    std::vector<float> areaUniqueVertices;
    std::size_t globalNumVertices = 0;
    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk = chunkIt->second;
        FloatChannel chunkVertices = *(chunk->getChannel<float>("vertices"));
        std::size_t numDuplicates = *chunk->getAtomic<unsigned int>("num_duplicates");
        std::size_t numVertices = chunk->numVertices();
        std::cout << "vertices per chunk: " << numVertices << std::endl;
        globalNumVertices += numVertices;
        std::cout << "global num of vertices: " << globalNumVertices << std::endl;
        std::unordered_map<std::size_t, std::size_t> chunkVertexIndices;

        if (numVertices == 0)
        {
            continue;
        }

        if (chunkIt == chunks.begin() || areaDuplicateVertices.empty())
        {
            std::cout << "add Duplicates" << std::endl;
            areaDuplicateVertices.insert(areaDuplicateVertices.end(),
                                  chunkVertices.dataPtr().get(),
                                  chunkVertices.dataPtr().get() + (numDuplicates * 3));
        }
        else
        {
            for (std::size_t i = 0; i < numDuplicates; ++i)
            {
                const size_t areaDuplicateVerticesSize = areaDuplicateVertices.size();

                // TODO: get and compare duplicate vertices
                bool found = false;
                for (std::size_t j = 0; j < areaDuplicateVerticesSize / 3; ++j)
                {
                    if ((areaDuplicateVertices[j * 3] == chunkVertices[i][0]) &&
                        (areaDuplicateVertices[j * 3 + 1] == chunkVertices[i][1]) &&
                        (areaDuplicateVertices[j * 3 + 2] == chunkVertices[i][2]))
                    {
                        found = true;
                        chunkVertexIndices.insert({i, j});
                        break;
                    }
                }

                if (!found) {
                    areaDuplicateVertices.push_back(chunkVertices[i][0]);
                    areaDuplicateVertices.push_back(chunkVertices[i][1]);
                    areaDuplicateVertices.push_back(chunkVertices[i][2]);

                    chunkVertexIndices.insert({i, areaDuplicateVertices.size() / 3 - 1});
                }
            }
        }

        areaUniqueVertices.insert(areaUniqueVertices.end(),
                                  chunkVertices.dataPtr().get() + (numDuplicates * 3),
                                  (chunkVertices.dataPtr().get() + (numVertices * 3 )));

        areaVertexIndices.push_back(chunkVertexIndices);
    }

    std::vector<unsigned int> areaFaceIndices;
    const std::size_t staticFaceIndexOffset = areaDuplicateVertices.size() / 3;
    std::size_t dynFaceIndexOffset = 0;
    auto areaVertexIndicesIt = areaVertexIndices.begin();
    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk = chunkIt->second;
        indexArray chunkFaceIndices = chunk->getFaceIndices();
        std::size_t numDuplicates = *chunk->getAtomic<unsigned int>("num_duplicates");
        std::size_t numVertices = chunk->numVertices();
        std::size_t numFaces = chunk->numFaces();
        std::size_t faceIndexOffset = staticFaceIndexOffset - numDuplicates + dynFaceIndexOffset;

        for (std::size_t i = 0; i < numFaces * 3; ++i)
        {
            std::size_t oldIndex = chunkFaceIndices[i];
            auto it = (*areaVertexIndicesIt).find(oldIndex);
            if(it != (*areaVertexIndicesIt).end())
            {
//                std::cout << "map" << std::endl;
                // TODO: use map entry
                areaFaceIndices.push_back(it->second);
            }
            else
            {
//                std::cout << "offset" << std::endl;
                // TODO: add offset
                areaFaceIndices.push_back(oldIndex + faceIndexOffset);
            }
        }
        dynFaceIndexOffset += (numVertices - numDuplicates);
        ++areaVertexIndicesIt;
    }

    std::cout << "combine vertices" << std::endl;
    std::cout << "Duplicates: " << areaDuplicateVertices.size() / 3 << std::endl;
    std::cout << "Unique: " << areaUniqueVertices.size() / 3 << std::endl;
    areaDuplicateVertices.insert(areaDuplicateVertices.end(), areaUniqueVertices.begin(), areaUniqueVertices.end());

    std::size_t areaVertexNum = areaDuplicateVertices.size() / 3;
    float* tmpVertices = areaDuplicateVertices.data();
    floatArr vertexArr = floatArr(tmpVertices, VectorCapsule<float>(std::move(areaDuplicateVertices)));

    std::size_t faceIndexNum = areaFaceIndices.size() / 3;
    auto* tmpFaceIndices = areaFaceIndices.data();
    indexArray faceIndexArr = indexArray(tmpFaceIndices, VectorCapsule<unsigned int>(std::move(areaFaceIndices)));

    MeshBufferPtr areaMeshPtr(new MeshBuffer);
    areaMeshPtr->setVertices(vertexArr, areaVertexNum);
    areaMeshPtr->setFaceIndices(faceIndexArr, faceIndexNum);

    std::cout << "Vertices: " << areaMeshPtr->numVertices() << ", Faces: " << areaMeshPtr->numFaces() << std::endl;

    return areaMeshPtr;
}

void ChunkManager::initBoundingBox(MeshBufferPtr mesh)
{
    FloatChannel vertices = mesh->getFloatChannel("vertices").get();
    for (unsigned int i = 0; i < vertices.numElements(); i++)
    {
        m_boundingBox.expand(static_cast<BaseVector<float>>(vertices[i]));
    }
}

void ChunkManager::cutLargeFaces(
    std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> halfEdgeMesh,
    float overlapRatio,
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitVertices,
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitFaces)
{
    // check all edges if they range too far into different chunks
    MeshHandleIteratorPtr<EdgeHandle> iterator = halfEdgeMesh->edgesBegin();
    while (iterator != halfEdgeMesh->edgesEnd())
    {
        // check both directions for each face
        std::array<VertexHandle, 2> vertices = halfEdgeMesh->getVerticesOfEdge(*iterator);
        for (unsigned int i = 0; i <= 1; i++)
        {
            VertexHandle referenceVertex = vertices[i];
            VertexHandle comparedVertex  = vertices[(i + 1) % 2];

            bool isLargeEdge = false;

            // check distance to nearest chunkBorder for all three directions
            for (unsigned int axis = 0; axis < 3; axis++)
            {
                // key for size comparison depending on the current axis
                float referenceVertexKey = halfEdgeMesh->getVertexPosition(referenceVertex)[axis];
                float comparedVertexKey  = halfEdgeMesh->getVertexPosition(comparedVertex)[axis];

                // if the edge goes over multiple chunks it is to large because of a chunk border
                // located in the middle of the edge
                if (fabs(referenceVertexKey - comparedVertexKey) > 2 * m_chunkSize)
                {
                    isLargeEdge = true;
                    break;
                }

                // get coordinate for plane in direction of the current axis
                float chunkBorder
                    = m_chunkSize * (static_cast<int>(referenceVertexKey / m_chunkSize))
                      + fmod(m_boundingBox.getMin()[axis], m_chunkSize);

                // select plane of chunk depending oh the relative position of the copared
                // vertex
                if (referenceVertexKey < comparedVertexKey)
                {
                    chunkBorder += m_chunkSize;
                }

                // chech whether or not to cut the face
                if (referenceVertexKey - chunkBorder < 0 && comparedVertexKey - chunkBorder >= 0
                    && chunkBorder - referenceVertexKey > overlapRatio * m_chunkSize
                    && comparedVertexKey - chunkBorder > overlapRatio * m_chunkSize)
                {
                    isLargeEdge = true;
                    break;
                }
                else if (referenceVertexKey - chunkBorder >= 0
                         && comparedVertexKey - chunkBorder < 0
                         && referenceVertexKey - chunkBorder > overlapRatio * m_chunkSize
                         && chunkBorder - comparedVertexKey > overlapRatio * m_chunkSize)
                {
                    isLargeEdge = true;
                    break;
                }
            }

            if (isLargeEdge)
            {
                std::array<OptionalFaceHandle, 2> faces = halfEdgeMesh->getFacesOfEdge(*iterator);

                // build newIndex -> oldIndex map to use
                unsigned int faceIndex = halfEdgeMesh->nextFaceIndex();
                if (faces[0])
                {
                    unsigned int face = faces[0].unwrap().idx();
                    while (splitFaces->find(face) != splitFaces->end())
                    {
                        face = splitFaces->at(face);
                    }
                    splitFaces->insert({faceIndex, face});
                    faceIndex++;
                }
                if (faces[1])
                {
                    unsigned int face = faces[1].unwrap().idx();
                    while (splitFaces->find(face) != splitFaces->end())
                    {
                        face = splitFaces->at(face);
                    }
                    splitFaces->insert({faceIndex, face});
                }

                unsigned int vertex = referenceVertex.idx();
                while (splitVertices->find(vertex) != splitVertices->end())
                {
                    vertex = splitVertices->at(vertex);
                }
                splitVertices->insert({halfEdgeMesh->nextVertexIndex(), vertex});

                // cut edge in half
                float cutRatio = 0.5;
                BaseVector<float> cutPoint
                    = halfEdgeMesh->getVertexPosition(referenceVertex) * cutRatio
                      + halfEdgeMesh->getVertexPosition(comparedVertex) * (1 - cutRatio);

                halfEdgeMesh->splitVertex(*iterator,
                                          referenceVertex,
                                          halfEdgeMesh->getVertexPosition(referenceVertex),
                                          cutPoint);
                break;
            }
        }

        ++iterator;
    }
}

void ChunkManager::buildChunks(MeshBufferPtr mesh, float maxChunkOverlap, std::string savePath)
{
    std::vector<ChunkBuilderPtr> chunkBuilders(m_amount.x * m_amount.y * m_amount.z);

    std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> halfEdgeMesh
        = std::shared_ptr<HalfEdgeMesh<BaseVector<float>>>(
            new HalfEdgeMesh<BaseVector<float>>(mesh));

    // map from new indices to old indices to allow attributes for cut faces
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitVertices(
        new std::unordered_map<unsigned int, unsigned int>);
    std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitFaces(
        new std::unordered_map<unsigned int, unsigned int>);

    // prepare mash to prevent faces from overlapping too much on chunk borders
    cutLargeFaces(halfEdgeMesh, maxChunkOverlap, splitVertices, splitFaces);

    // one vector of variable size for each vertex - this is used for duplicate detection
    std::shared_ptr<std::unordered_map<unsigned int, std::vector<std::weak_ptr<ChunkBuilder>>>>
    vertexUse(new std::unordered_map<unsigned int, std::vector<std::weak_ptr<ChunkBuilder>>>());

    for (std::size_t i = 0; i < m_amount.x; i++)
    {
        for (std::size_t j = 0; j < m_amount.y; j++)
        {
            for (std::size_t k = 0; k < m_amount.z; k++)
            {
                chunkBuilders[hashValue(i, j, k)]
                    = ChunkBuilderPtr(new ChunkBuilder(halfEdgeMesh, vertexUse));
            }
        }
    }

    // assign the faces to the chunks
    BaseVector<float> currentCenterPoint;
    MeshHandleIteratorPtr<FaceHandle> iterator = halfEdgeMesh->facesBegin();
    for (size_t i = 0; i < halfEdgeMesh->numFaces(); i++)
    {
        currentCenterPoint = getFaceCenter(halfEdgeMesh, *iterator);
        size_t cellIndex   = getCellIndex(currentCenterPoint);

        chunkBuilders[cellIndex]->addFace(*iterator);

        ++iterator;
    }

    // save the chunks as .ply
    for (std::size_t i = 0; i < m_amount.x; i++)
    {
        for (std::size_t j = 0; j < m_amount.y; j++)
        {
            for (std::size_t k = 0; k < m_amount.z; k++)
            {
                std::size_t hash = hashValue(i, j, k);

                if (chunkBuilders[hash]->numFaces() > 0)
                {
                    std::cout << "writing " << i << " " << j << " " << k << std::endl;

                    // get mesh of chunk from chunk builder
                    MeshBufferPtr chunkMeshPtr
                        = chunkBuilders[hash]->buildMesh(mesh, splitVertices, splitFaces);

                    // insert chunked mesh into hash grid
                    m_hashGrid.insert({hash, chunkMeshPtr});

                    // export chunked meshes for debugging
                    ModelFactory::saveModel(ModelPtr(new Model(chunkMeshPtr)),
                                            savePath + "/" + std::to_string(i) + "-"
                                                + std::to_string(j) + "-" + std::to_string(k)
                                                + ".ply");
                }
            }
        }
    }
}

BaseVector<float> ChunkManager::getFaceCenter(std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> mesh,
                                              const FaceHandle& handle) const
{
    return (mesh->getVertexPositionsOfFace(handle)[0] + mesh->getVertexPositionsOfFace(handle)[1]
            + mesh->getVertexPositionsOfFace(handle)[2])
           / 3;
}

std::size_t ChunkManager::getCellIndex(const BaseVector<float>& vec) const
{
    BaseVector<float> tmpVec = (vec - m_boundingBox.getMin()) / m_chunkSize;
    return static_cast<size_t>(tmpVec.x) * m_amount.y * m_amount.z
           + static_cast<size_t>(tmpVec.y) * m_amount.z + static_cast<size_t>(tmpVec.z);
}

} /* namespace lvr2 */
