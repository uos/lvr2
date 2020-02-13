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
 * ChunkManager.hpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 * @author Marcel Wiegand
 * @author Raphael Marx
 */

#ifndef CHUNK_MANAGER_HPP
#define CHUNK_MANAGER_HPP

#include "lvr2/algorithm/ChunkBuilder.hpp"
#include "lvr2/algorithm/ChunkHashGrid.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/io/Model.hpp"
#include "lvr2/types/Channel.hpp"

namespace lvr2
{

class ChunkManager : public ChunkHashGrid
{
  public:
    using FilterFunction = std::function<bool(MultiChannelMap::val_type, size_t)>;

    /**
     * @brief ChunkManager creates chunks from an original mesh
     *
     * Chunks the original model into chunks of given size.
     * Every created chunk has the same length in height, width and depth.
     *
     * @param mesh mesh to be chunked
     * @param chunksize size of a chunk - unit depends on the given mesh
     * @param maxChunkOverlap maximum allowed overlap between chunks relative to the chunk size.
     * Larger triangles will be cut
     * @param savePath JUST FOR TESTING - REMOVE LATER ON
     * @param cacheSize maximum number of chunks loaded in the ChunkHashGrid
     */
    ChunkManager(MeshBufferPtr meshes,
                 float chunksize,
                 float maxChunkOverlap,
                 std::string savePath,
                 std::string layer = std::string("mesh"),
                 size_t cacheSize = 200
                 );


    ChunkManager(std::vector<MeshBufferPtr> meshes,
                 float chunksize,
                 float maxChunkOverlap,
                 std::string savePath,
                 std::vector<std::string> layers,
                 size_t cacheSize = 200);

    /**
     * @brief ChunkManager loads a ChunkManager from a given HDF5-file
     *
     * Creates a ChunkManager from an already chunked HDF5 file and allows loading individual chunks
     * and combining them to partial meshes.
     * Every loaded chunk has the same length in height, width and depth.
     *
     * @param hdf5Path path to the HDF5 file, where chunks and additional information are stored
     * @param cacheSize maximum number of chunks loaded in the ChunkHashGrid
     */
    ChunkManager(std::string hdf5Path, size_t cacheSize = 200, float chunkSize = 10.0f);

    /**
     * @brief getGlobalBoundingBox is a getter for the bounding box of the entire chunked model
     * 
     * @return global bounding box of chunked model
     */
    inline BoundingBox<BaseVector<float>> getGlobalBoundingBox()
    {
        return m_boundingBox;
    }

    /**
     * @brief buildChunks builds chunks from an original mesh
     *
     * Creates chunks from an original mesh and initializes the initial chunk structure
     *
     * @param mesh mesh which is being chunked
     * @param maxChunkOverlap maximum allowed overlap between chunks relative to the chunk size.
     * Larger triangles will be cut
     * @param savePath UST FOR TESTING - REMOVE LATER ON
     */
    void buildChunks(MeshBufferPtr mesh, float maxChunkOverlap, std::string savePath, std::string layer = std::string("mesh"));

    /**
     * @brief extractArea creates and returns MeshBufferPtr of merged chunks for given area.
     *
     * Finds corresponding chunks for given area inside the grid and merges those chunks to a new
     * mesh without duplicated vertices. The new mesh is returned as MeshBufferPtr.
     *
     * @param area
     * @return mesh of the given area
     */
    MeshBufferPtr extractArea(const BoundingBox<BaseVector<float>>& area, std::string layer= std::string("mesh"));

    void extractArea(const BoundingBox<BaseVector<float> >& area,
                    std::unordered_map<std::size_t, MeshBufferPtr>& chunks,
                    std::string layer = std::string("mesh"));
                    

    /**
     * @brief extractArea creates and returns MeshBufferPtr of merged chunks for given area after
     * filtering the resulting mesh.
     *
     * Finds corresponding chunks for given area inside the grid and merges those chunks to a new
     * mesh without duplicated vertices. The new mesh is returned as MeshBufferPtr.
     * After extracting the area, given filters are being applied to allow more precise quaries for
     * use cases like robot navigation while minimizing network traffic.
     *
     * example for a filter:
     * std::map<std::string, lvr2::ChunkManager::FilterFunction> filter
     *    = {{"roughness", [](lvr2::MultiChannelMap::val_type channel, size_t index) {
     *           return channel.dataPtr<float>()[index] < 0.1;
     *       }}};
     * This filter will only use vertices that have a maximum roughness of 0,1. All Faces
     * referencing the filtered vertices will be filtered too.
     *
     * @param area bounding box of the area to request
     * @param filter map of filters with channel names as keys and functions as values
     * @return mesh of the given area
     */
    MeshBufferPtr extractArea(const BoundingBox<BaseVector<float>>& area,
                              const std::map<std::string, FilterFunction> filter,
                              std::string layer = std::string("mesh"));

    /**
     * @brief Get all existing channels from mesh
     * 
     * @return List of all channel names as vector
     */
    std::vector<std::string> getChannelsFromMesh(std::string layer = std::string("mesh"));

    /**
     * @brief Loads all chunks into the ChunkHashGrid.
     * DEBUG -- Only used for testing, but might be useful for smaller meshes.
     */
    void loadAllChunks(std::string layer = std::string("mesh"));

  private:
    /**
     * @brief initBoundingBox calculates a bounding box of the original mesh
     *
     * This calculates the bounding box of the given model and saves it to m_boundingBox.
     *
     * @param mesh mesh whose bounding box shall be calculated
     */
    void initBoundingBox(MeshBufferPtr mesh);

    /**
     * @brief cutLargeFaces cuts a face if it is too large
     *
     * Checks whether or not a triangle is overlapping the chunk borders too much and
     * cuts those faces. The resulting smaller faces will be added as additional vertices
     * and faces to the chunk hat holds their center point.
     *
     * @param halfEdgeMesh mesh  that is being cut
     * @param overlapRatio ration of maximum allowed overlap and the chunks side length
     * @param splitVertices map from new vertex indices to old vertex indices for all faces that
     * will be cut
     * @param splitFaces map from new face indices to old face indices for all faces that will be
     * cut
     */
    void
    cutLargeFaces(std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> halfEdgeMesh,
                  float overlapRatio,
                  std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitVertices,
                  std::shared_ptr<std::unordered_map<unsigned int, unsigned int>> splitFaces);

    /**
     * @brief getFaceCenter gets the center point for a given face
     *
     * @param verticesChannel channel of mesh that holds the vertices
     * @param facesChannel channel of mesh that holds the faces
     * @param faceIndex index of the requested face
     * @return center point of the given face
     */
    BaseVector<float> getFaceCenter(std::shared_ptr<HalfEdgeMesh<BaseVector<float>>> mesh,
                                    const FaceHandle& handle) const;

    //    /**
    //     * @brief find corresponding grid cell of given point
    //     *
    //     * @param vec point of mesh to find cell id for
    //     * @return cell id
    //     */
    //    std::string getCellName(const BaseVector<float>& vec) const;

    /**
     * @brief returns the grid coordinates of a given point
     *
     * @param vec point of which we want the grid coordinates
     * @return the grid coordinates as a BaseVector
     */
    BaseVector<int> getCellCoordinates(const BaseVector<float>& vec) const;
    
    /**
     * @brief reads and combines a channel of multiple chunks
     *
     * @param chunks list of chunks to combine
     * @param channelName name of channel to extract
     * @param staticVertexIndexOffset amount of duplicate vertices in the combined mesh
     * @param numVertices amount of vertices in the combined mesh
     * @param numFaces amount of faces in the combined mesh
     * @param areaVertexIndices mapping from old vertex index to new vertex index per chunk
     */
    template <typename T>
    ChannelPtr<T> extractChannelOfArea(
        std::unordered_map<std::size_t, MeshBufferPtr>& chunks,
        std::string channelName,
        std::size_t staticVertexIndexOffset,
        std::size_t numVertices,
        std::size_t numFaces,
        std::vector<std::unordered_map<std::size_t, std::size_t>>& areaVertexIndices);

    /**
     * @brief applies given filter arrays to one channel
     *
     * @tparam T Type of channels contents
     * @param vertexFilter filter array for vertices
     * @param faceFilter filter array for faces
     * @param numVertices amount of vertices after filtering
     * @param numFaces amount of faces after filtering
     * @param meshBuffer original mesh
     * @param originalChannel channel to filter
     *
     * @return a new filtered channel
     */
    template <typename T>
    MultiChannelMap::val_type
    applyChannelFilter(const std::vector<bool>& vertexFilter,
                       const std::vector<bool>& faceFilter,
                       const size_t numVertices,
                       const size_t numFaces,
                       const MeshBufferPtr meshBuffer,
                       const MultiChannelMap::val_type& originalChannel) const;
};

} /* namespace lvr2 */

#include "ChunkManager.tcc"

#endif // CHUNK_MANAGER_HPP
