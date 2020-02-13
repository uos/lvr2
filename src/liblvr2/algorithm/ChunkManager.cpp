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

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cmath>

namespace
{
template <typename T>
struct VectorCapsule
{
  private:
    std::shared_ptr<std::vector<T>> arr_;

  public:
    explicit VectorCapsule(std::vector<T>&& arr)
        : arr_(std::make_shared<std::vector<T>>(std::move(arr)))
    {
    }
    ~VectorCapsule() = default;
    void operator()(void*) {}
};
} // namespace

namespace lvr2
{

ChunkManager::ChunkManager(MeshBufferPtr mesh,
                           float chunksize,
                           float maxChunkOverlap,
                           std::string savePath,
                           std::string layer,
                           size_t cacheSize)
    : ChunkManager(std::vector<MeshBufferPtr>{mesh},
                   chunksize,
                   maxChunkOverlap,
                   savePath,
                   std::vector<std::string>{layer},
                   cacheSize)
{
}
ChunkManager::ChunkManager(std::vector<MeshBufferPtr> meshes,
                           float chunksize,
                           float maxChunkOverlap,
                           std::string savePath,
                           std::vector<std::string> layers,
                           size_t cacheSize)
    : ChunkHashGrid(savePath + "/chunk_mesh.h5", cacheSize)
{
    setChunkSize(chunksize);
    if (meshes.size() != layers.size())
    {
        std::cerr << lvr2::timestamp << "Number of meshes and layers do not match: \n"
                  << "Num meshes: " << meshes.size() << "\n"
                  << "Num layers: " << layers.size() << std::endl;

        return;
    }

    // TODO use biggest
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        initBoundingBox(meshes[i]);
    }

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        buildChunks(meshes[i], maxChunkOverlap, savePath, layers[i]);
    }
}

ChunkManager::ChunkManager(std::string hdf5Path, size_t cacheSize, float chunkSize)
    : ChunkHashGrid(hdf5Path, cacheSize, chunkSize)
{
}

std::vector<std::string> ChunkManager::getChannelsFromMesh(std::string layer)
{
    std::vector<std::string> attributeList;

    MeshBufferPtr chunkPtr;
    bool isChunkFound = false;
    for (int x = getChunkMinChunkIndex().x; x < getChunkMaxChunkIndex().x && !isChunkFound; x++)
    {
        for (int y = getChunkMinChunkIndex().y; y < getChunkMaxChunkIndex().y && !isChunkFound; y++)
        {
            for (int z = getChunkMinChunkIndex().z; z < getChunkMaxChunkIndex().z && !isChunkFound;
                 z++)
            {
                boost::optional<MeshBufferPtr> requestedChunk
                    = getChunk<MeshBufferPtr>(layer, x, y, z);
                if (requestedChunk)
                {
                    chunkPtr     = requestedChunk.get();
                    isChunkFound = true;
                }
            }
        }
    }

    if (isChunkFound)
    {
        for (auto channelIterator = chunkPtr->begin(); channelIterator != chunkPtr->end();
             ++channelIterator)
        {
            attributeList.push_back(channelIterator->first);
        }
    }
    return attributeList;
}

void ChunkManager::extractArea(const BoundingBox<BaseVector<float>>& area,
                               std::unordered_map<std::size_t, MeshBufferPtr>& chunks,
                               std::string layer)
{
    // adjust area to our maximum boundingBox
    BaseVector<float> adjustedAreaMin, adjustedAreaMax;
    adjustedAreaMax[0] = std::min(area.getMax()[0], getBoundingBox().getMax()[0]);
    adjustedAreaMax[1] = std::min(area.getMax()[1], getBoundingBox().getMax()[1]);
    adjustedAreaMax[2] = std::min(area.getMax()[2], getBoundingBox().getMax()[2]);
    adjustedAreaMin[0] = std::max(area.getMin()[0], getBoundingBox().getMin()[0]);
    adjustedAreaMin[1] = std::max(area.getMin()[1], getBoundingBox().getMin()[1]);
    adjustedAreaMin[2] = std::max(area.getMin()[2], getBoundingBox().getMin()[2]);
    BoundingBox<BaseVector<float>> adjustedArea
        = BoundingBox<BaseVector<float>>(adjustedAreaMin, adjustedAreaMax);

    // find all required chunks
    // TODO: check if we need + 1
    const BaseVector<float> maxSteps
        = (adjustedArea.getMax() - adjustedArea.getMin()) / getChunkSize();
    for (std::size_t i = 0; i < maxSteps.x; ++i)
    {
        for (std::size_t j = 0; j < maxSteps.y; ++j)
        {
            for (std::size_t k = 0; k < maxSteps.z; ++k)
            {
                BaseVector<int> cellCoord = getCellCoordinates(
                    adjustedArea.getMin() + BaseVector<float>(i, j, k) * getChunkSize());
                size_t cellIndex = hashValue(cellCoord.x, cellCoord.y, cellCoord.z);

                // if element is already loaded.
                // if(chunks.find(cellIndex) != chunks.end())
                //{
                //    continue;
                //}

                boost::optional<MeshBufferPtr> loadedChunk
                    = getChunk<MeshBufferPtr>(layer, cellCoord.x, cellCoord.y, cellCoord.z);

                if (loadedChunk)
                {
                    // TODO: remove saving tmp chunks later
                    // ModelFactory::saveModel(lvr2::ModelPtr(new lvr2::Model(loadedChunk)),
                    //                        "area/" + std::to_string(cellIndex) + ".ply");
                    chunks.insert({cellIndex, *loadedChunk});
                }
            }
        }
    }
    //    std::cout << "Num chunks " << chunks.size() << std::endl;
}

MeshBufferPtr ChunkManager::extractArea(const BoundingBox<BaseVector<float>>& area,
                                        std::string layer)
{
    std::unordered_map<std::size_t, MeshBufferPtr> chunks;

    // adjust area to our maximum boundingBox
    BaseVector<float> adjustedAreaMin, adjustedAreaMax;
    adjustedAreaMax[0] = std::min(area.getMax()[0], getBoundingBox().getMax()[0]);
    adjustedAreaMax[1] = std::min(area.getMax()[1], getBoundingBox().getMax()[1]);
    adjustedAreaMax[2] = std::min(area.getMax()[2], getBoundingBox().getMax()[2]);
    adjustedAreaMin[0] = std::max(area.getMin()[0], getBoundingBox().getMin()[0]);
    adjustedAreaMin[1] = std::max(area.getMin()[1], getBoundingBox().getMin()[1]);
    adjustedAreaMin[2] = std::max(area.getMin()[2], getBoundingBox().getMin()[2]);
    BoundingBox<BaseVector<float>> adjustedArea
        = BoundingBox<BaseVector<float>>(adjustedAreaMin, adjustedAreaMax);

    // find all required chunks
    // TODO: check if we need + 1
    const BaseVector<float> maxSteps
        = (adjustedArea.getMax() - adjustedArea.getMin()) / getChunkSize();
    for (std::size_t i = 0; i < maxSteps.x; ++i)
    {
        for (std::size_t j = 0; j < maxSteps.y; ++j)
        {
            for (std::size_t k = 0; k < maxSteps.z; ++k)
            {
                BaseVector<int> cellCoord = getCellCoordinates(
                    adjustedArea.getMin() + BaseVector<float>(i, j, k) * getChunkSize());
                size_t cellIndex = hashValue(cellCoord.x, cellCoord.y, cellCoord.z);

                boost::optional<MeshBufferPtr> loadedChunk
                    = getChunk<MeshBufferPtr>(layer, cellCoord.x, cellCoord.y, cellCoord.z);
                if (loadedChunk)
                {
                    // TODO: remove saving tmp chunks later
                    ModelFactory::saveModel(lvr2::ModelPtr(new lvr2::Model(*loadedChunk)),
                                            "area/" + std::to_string(cellIndex) + ".ply");
                    chunks.insert({cellIndex, *loadedChunk});
                }
            }
        }
    }
    std::cout << "Extracted " << chunks.size() << " Chunks" << std::endl;

    std::vector<float> areaDuplicateVertices;
    std::vector<std::unordered_map<std::size_t, std::size_t>> areaVertexIndices;
    std::vector<float> areaUniqueVertices;
    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk        = chunkIt->second;
        FloatChannel chunkVertices = *(chunk->getChannel<float>("vertices"));
        std::size_t numDuplicates  = *chunk->getAtomic<unsigned int>("num_duplicates");
        std::size_t numVertices    = chunk->numVertices();
        std::unordered_map<std::size_t, std::size_t> chunkVertexIndices;

        if (numVertices == 0)
        {
            continue;
        }

        if ((chunkIt == chunks.begin() || areaDuplicateVertices.empty()) && numDuplicates > 0)
        {
            areaDuplicateVertices.insert(areaDuplicateVertices.end(),
                                         chunkVertices.dataPtr().get(),
                                         chunkVertices.dataPtr().get() + (numDuplicates * 3));
        }

        for (std::size_t i = 0; i < numDuplicates; ++i)
        {
            const size_t areaDuplicateVerticesSize = areaDuplicateVertices.size();

            bool found = false;
            for (std::size_t j = 0; j < areaDuplicateVerticesSize / 3; ++j)
            {
                if ((areaDuplicateVertices[j * 3] == chunkVertices[i][0])
                    && (areaDuplicateVertices[j * 3 + 1] == chunkVertices[i][1])
                    && (areaDuplicateVertices[j * 3 + 2] == chunkVertices[i][2]))
                {
                    found = true;
                    chunkVertexIndices.insert({i, j});
                    break;
                }
            }

            if (!found)
            {
                areaDuplicateVertices.push_back(chunkVertices[i][0]);
                areaDuplicateVertices.push_back(chunkVertices[i][1]);
                areaDuplicateVertices.push_back(chunkVertices[i][2]);

                chunkVertexIndices.insert({i, areaDuplicateVertices.size() / 3 - 1});
            }
        }

        areaUniqueVertices.insert(areaUniqueVertices.end(),
                                  chunkVertices.dataPtr().get() + (numDuplicates * 3),
                                  (chunkVertices.dataPtr().get() + (numVertices * 3)));

        areaVertexIndices.push_back(chunkVertexIndices);
    }

    std::vector<unsigned int> areaFaceIndices;
    const std::size_t staticFaceIndexOffset = areaDuplicateVertices.size() / 3;
    std::size_t dynFaceIndexOffset          = 0;
    auto areaVertexIndicesIt                = areaVertexIndices.begin();
    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk         = chunkIt->second;
        indexArray chunkFaceIndices = chunk->getFaceIndices();
        std::size_t numDuplicates   = *chunk->getAtomic<unsigned int>("num_duplicates");
        std::size_t numVertices     = chunk->numVertices();
        std::size_t numFaces        = chunk->numFaces();
        std::size_t faceIndexOffset = staticFaceIndexOffset - numDuplicates + dynFaceIndexOffset;

        for (std::size_t i = 0; i < numFaces * 3; ++i)
        {
            std::size_t oldIndex = chunkFaceIndices[i];
            auto it              = (*areaVertexIndicesIt).find(oldIndex);
            if (it != (*areaVertexIndicesIt).end())
            {
                areaFaceIndices.push_back(it->second);
            }
            else
            {
                areaFaceIndices.push_back(oldIndex + faceIndexOffset);
            }
        }
        dynFaceIndexOffset += (numVertices - numDuplicates);
        ++areaVertexIndicesIt;
    }

    std::cout << "combine vertices" << std::endl;
    std::cout << "Duplicates: " << areaDuplicateVertices.size() / 3 << std::endl;
    std::cout << "Unique: " << areaUniqueVertices.size() / 3 << std::endl;
    areaDuplicateVertices.insert(
        areaDuplicateVertices.end(), areaUniqueVertices.begin(), areaUniqueVertices.end());

    std::size_t areaVertexNum = areaDuplicateVertices.size() / 3;
    float* tmpVertices        = areaDuplicateVertices.data();
    floatArr vertexArr
        = floatArr(tmpVertices, VectorCapsule<float>(std::move(areaDuplicateVertices)));

    std::size_t faceIndexNum = areaFaceIndices.size() / 3;
    auto* tmpFaceIndices     = areaFaceIndices.data();
    indexArray faceIndexArr
        = indexArray(tmpFaceIndices, VectorCapsule<unsigned int>(std::move(areaFaceIndices)));

    MeshBufferPtr areaMeshPtr(new MeshBuffer);
    areaMeshPtr->setVertices(vertexArr, areaVertexNum);
    areaMeshPtr->setFaceIndices(faceIndexArr, faceIndexNum);

    for (auto chunkIt = chunks.begin(); chunkIt != chunks.end(); ++chunkIt)
    {
        MeshBufferPtr chunk = chunkIt->second;
        for (auto elem : *chunk)
        {
            if (elem.first != "vertices" && elem.first != "face_indices"
                && elem.first != "num_duplicates")
            {
                if (areaMeshPtr->find(elem.first) == areaMeshPtr->end())
                {

                    if (elem.second.is_type<unsigned char>())
                    {
                        areaMeshPtr->template addChannel<unsigned char>(
                            extractChannelOfArea<unsigned char>(chunks,
                                                                elem.first,
                                                                staticFaceIndexOffset,
                                                                areaMeshPtr->numVertices(),
                                                                areaMeshPtr->numFaces(),
                                                                areaVertexIndices),
                            elem.first);
                    }
                    else if (elem.second.is_type<unsigned int>())
                    {
                        areaMeshPtr->template addChannel<unsigned int>(
                            extractChannelOfArea<unsigned int>(chunks,
                                                               elem.first,
                                                               staticFaceIndexOffset,
                                                               areaMeshPtr->numVertices(),
                                                               areaMeshPtr->numFaces(),
                                                               areaVertexIndices),
                            elem.first);
                    }
                    else if (elem.second.is_type<float>())
                    {
                        areaMeshPtr->template addChannel<float>(
                            extractChannelOfArea<float>(chunks,
                                                        elem.first,
                                                        staticFaceIndexOffset,
                                                        areaMeshPtr->numVertices(),
                                                        areaMeshPtr->numFaces(),
                                                        areaVertexIndices),
                            elem.first);
                    }
                }
            }
        }
    }

    std::cout << "Vertices: " << areaMeshPtr->numVertices()
              << ", Faces: " << areaMeshPtr->numFaces() << std::endl;

    ModelFactory::saveModel(ModelPtr(new Model(areaMeshPtr)), "test1.ply");

    return areaMeshPtr;
}

MeshBufferPtr ChunkManager::extractArea(const BoundingBox<BaseVector<float>>& area,
                                        const std::map<std::string, FilterFunction> filter,
                                        std::string layer)
{
    MeshBufferPtr areaMesh = extractArea(area, layer);

    // filter elements
    // filter lists: false is used to indicate that an element will not be used
    std::vector<bool> vertexFilter(areaMesh->numVertices(), true);
    std::vector<bool> faceFilter(areaMesh->numFaces(), true);
    std::size_t numVertices = areaMesh->numVertices();
    std::size_t numFaces    = areaMesh->numFaces();

    for (auto channelFilter : filter)
    {
        if (areaMesh->find(channelFilter.first) != areaMesh->end())
        {
            MultiChannelMap::val_type channel = areaMesh->at(channelFilter.first);
#pragma omp parallel for
            for (std::size_t i = 0; i < channel.numElements(); i++)
            {
                if (channel.numElements() == areaMesh->numVertices())
                {
                    if (vertexFilter[i] == true)
                    {
                        vertexFilter[i] = channelFilter.second(channel, i);
                        if (vertexFilter[i] == false)
                        {
                            numVertices--;
                        }
                    }
                }
                else if (channel.numElements() == areaMesh->numFaces())
                {
                    if (faceFilter[i] == true)
                    {
                        faceFilter[i] = channelFilter.second(channel, i);
                        if (faceFilter[i] == false)
                        {
                            numFaces--;
                        }
                    }
                }
            }
        }
    }

    // filter all faces that reference filtered vertices
    IndexChannel facesChannel = *areaMesh->getIndexChannel("face_indices");
    for (std::size_t i = 0; i < areaMesh->numFaces(); i++)
    {
        if (faceFilter[i] == true)
        {
            for (std::size_t j = 0; j < facesChannel.width(); j++)
            {
                if (vertexFilter[facesChannel[i][j]] == false)
                {
                    faceFilter[i] = false;
                    numFaces--;
                    break;
                }
            }
        }
    }

    // create a mapping from old vertices to new vertices
    std::vector<std::size_t> vertexIndexMapping(areaMesh->numVertices(), 0);
    std::size_t tmpIndex = 0;
    for (std::size_t i = 0; i < areaMesh->numVertices(); i++)
    {
        if (vertexFilter[i] == true)
        {
            vertexIndexMapping[i] = tmpIndex;
            tmpIndex++;
        }
    }

    // remove filtered elements
    // #pragma omp parallel
    {
        for (auto& channel : *areaMesh)
        {
            // #pragma omp single nowait
            {
                if (channel.second.is_type<unsigned char>())
                {
                    channel.second = applyChannelFilter<unsigned char>(
                        vertexFilter, faceFilter, numVertices, numFaces, areaMesh, channel.second);
                }
                else if (channel.second.is_type<unsigned int>())
                {
                    channel.second = applyChannelFilter<unsigned int>(
                        vertexFilter, faceFilter, numVertices, numFaces, areaMesh, channel.second);
                }
                else if (channel.second.is_type<float>())
                {
                    channel.second = applyChannelFilter<float>(
                        vertexFilter, faceFilter, numVertices, numFaces, areaMesh, channel.second);
                }
            }
        }
    }

    // use mapping from old vertex indices to new vertex indices to update face indices
    facesChannel = *areaMesh->getIndexChannel("face_indices");
    for (std::size_t i = 0; i < areaMesh->numFaces(); i++)
    {
        for (std::size_t j = 0; j < facesChannel.width(); j++)
        {
            facesChannel[i][j] = vertexIndexMapping[facesChannel[i][j]];
        }
    }

    return areaMesh;
}

void ChunkManager::initBoundingBox(MeshBufferPtr mesh)
{
    BoundingBox<BaseVector<float>> boundingBox;
    FloatChannel vertices = mesh->getFloatChannel("vertices").get();
    for (unsigned int i = 0; i < vertices.numElements(); i++)
    {
        boundingBox.expand(static_cast<BaseVector<float>>(vertices[i]));
    }
    setBoundingBox(boundingBox);
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

                // if the edge goes over multiple chunks it is to large because of a chunk
                // border located in the middle of the edge
                if (fabs(referenceVertexKey - comparedVertexKey) > 2 * getChunkSize())
                {
                    isLargeEdge = true;
                    break;
                }

                // get coordinate for plane in direction of the current axis
                float chunkBorder
                    = getChunkSize() * (static_cast<int>(referenceVertexKey / getChunkSize()))
                      + fmod(getBoundingBox().getMin()[axis], getChunkSize());

                // select plane of chunk depending on the relative position of the compared
                // vertex
                if (referenceVertexKey < comparedVertexKey)
                {
                    chunkBorder += getChunkSize();
                }

                // check whether or not to cut the face
                if (referenceVertexKey - chunkBorder < 0 && comparedVertexKey - chunkBorder >= 0
                    && chunkBorder - referenceVertexKey > overlapRatio * getChunkSize()
                    && comparedVertexKey - chunkBorder > overlapRatio * getChunkSize())
                {
                    isLargeEdge = true;
                    break;
                }
                else if (referenceVertexKey - chunkBorder >= 0
                         && comparedVertexKey - chunkBorder < 0
                         && referenceVertexKey - chunkBorder > overlapRatio * getChunkSize()
                         && chunkBorder - comparedVertexKey > overlapRatio * getChunkSize())
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

void ChunkManager::buildChunks(MeshBufferPtr mesh,
                               float maxChunkOverlap,
                               std::string savePath,
                               std::string layer)
{
    std::vector<ChunkBuilderPtr> chunkBuilders(getChunkAmount().x * getChunkAmount().y
                                               * getChunkAmount().z);

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

    for (int i = getChunkMinChunkIndex().x; i < getChunkMaxChunkIndex().x; i++)
    {
        for (int j = getChunkMinChunkIndex().y; j < getChunkMaxChunkIndex().y; j++)
        {
            for (int k = getChunkMinChunkIndex().z; k < getChunkMaxChunkIndex().z; k++)
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
        currentCenterPoint             = getFaceCenter(halfEdgeMesh, *iterator);
        BaseVector<int> cellCorrdinate = getCellCoordinates(currentCenterPoint);
        size_t hash = hashValue(cellCorrdinate.x, cellCorrdinate.y, cellCorrdinate.z);

        chunkBuilders[hash]->addFace(*iterator);

        ++iterator;
    }

    // save the chunks as .ply
    for (int i = getChunkMinChunkIndex().x; i < getChunkMaxChunkIndex().x; i++)
    {
        for (int j = getChunkMinChunkIndex().y; j < getChunkMaxChunkIndex().y; j++)
        {
            for (int k = getChunkMinChunkIndex().z; k < getChunkMaxChunkIndex().z; k++)
            {
                std::size_t hash = hashValue(i, j, k);

                if (chunkBuilders[hash]->numFaces() > 0)
                {
                    // std::cout << "writing " << i << " " << j << " " << k << std::endl;

                    // get mesh of chunk from chunk builder
                    MeshBufferPtr chunkMeshPtr
                        = chunkBuilders[hash]->buildMesh(mesh, splitVertices, splitFaces);

                    // export chunked meshes for debugging
                    // ModelFactory::saveModel(ModelPtr(new Model(chunkMeshPtr)),
                    //                        savePath + "/" + std::to_string(i) + "-"
                    //                            + std::to_string(j) + "-" + std::to_string(k)
                    //                            + ".ply");
                    // write chunk in hdf5
                    setChunk<MeshBufferPtr>(layer, i, j, k, chunkMeshPtr);

                    chunkBuilders[hash] = nullptr; // deallocate
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

BaseVector<int> ChunkManager::getCellCoordinates(const BaseVector<float>& vec) const
{
    BaseVector<float> tmpVec = vec / getChunkSize();
    BaseVector<int> ret      = BaseVector<int>(
        static_cast<int>(tmpVec.x), static_cast<int>(tmpVec.y), static_cast<int>(tmpVec.z));
    return ret;
}

// std::string ChunkManager::getCellName(const BaseVector<float>& vec) const
//{
//    BaseVector<float> tmpVec = (vec - getBoundingBox().getMin()) / getChunkSize();
//    return std::to_string(static_cast<size_t>(tmpVec.x)) + "_"+
//    std::to_string(static_cast<size_t>(tmpVec.y)) + "_"
//    + std::to_string(static_cast<size_t>(tmpVec.z));
//}

void ChunkManager::loadAllChunks(std::string layer)
{
    int numLoaded = 0;
    for (int i = 0; i < getChunkAmount()[0]; i++)
    {
        for (int j = 0; j < getChunkAmount()[1]; j++)
        {
            for (int k = 0; k < getChunkAmount()[2]; k++)
            {
                if (loadChunk<MeshBufferPtr>(layer, i, j, k))
                {
                    numLoaded++;
                }
            }
        }
    }

    std::cout << "loaded " << numLoaded << " chunks from hdf5-file." << std::endl;
}

} /* namespace lvr2 */
