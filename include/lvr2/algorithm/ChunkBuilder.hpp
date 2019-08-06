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
 * ChunkBuilder.hpp
 *
 * @date 21.07.2019
 * @author Malte kl. Piening
 */

#ifndef CHUNK_BUILDER_HPP
#define CHUNK_BUILDER_HPP

#include "lvr2/io/Model.hpp"

#include <unordered_map>

namespace lvr2
{

class ChunkBuilder;

using ChunkBuilderPtr = std::shared_ptr<ChunkBuilder>;

class ChunkBuilder : public std::enable_shared_from_this<ChunkBuilder>
{
    public:
        /**
         * @brief ChunkBuilder constructs a chun builder that can create individual chunks
         *
         * @param originalMesh mesh that is being chunked
         * @param vertexUse list of ChunkBuilders that for each vertex of the original mesh
         */
        ChunkBuilder(MeshBufferPtr originalMesh,
                     std::shared_ptr<std::vector<std::vector<ChunkBuilderPtr>>> vertexUse);

        ~ChunkBuilder();

        /**
         * @brief assigns a face to the chunk this builder is generating
         *
         * This adds a face to the face list of this builder.
         * It is used to create a chunk when calling buildMesh.
         *
         * @param index index of face in the original model
         */
        void addFace(unsigned int index);

        /**
         * @brief addDuplicateVertex marks a vertex as duplicate
         *
         * Duplicate vertices will be added to the beginning of the Vertex array of the chunk mesh when
         * calling buildMesh().
         *
         * @param index vertex index of diplicate vertex
         */
        void addDuplicateVertex(unsigned int index);

        /**
         * @brief addAdditionalVertex adds a new vertex that is not part of the original mesh buffer
         *
         * Adds a new vertex that is not part of the original mesh to allow splitting faces during the
         * chunking process.
         *
         * @param x x-coordinate of the new vertex
         * @param y y-coordinate of the new vertex
         * @param z z-coordinate of the new vertex
         * @return index of the newly created vertex
         */
        unsigned int addAdditionalVertex(float x, float y, float z);

        /**
         * @brief addAdditionalFace adds a new face that is not part of the original mesh buffer
         *
         * Adds a new face from three vertices to the resulting mesh.
         * Use negative vertex indices for now to reference vertices that were added as additional vertices.
         * TODO: use a nicer method to reference additional vertices than just using negative indices!
         *
         * @param vertex1 first vertex index
         * @param vertex2 second vertex index
         * @param vertex3 third vertex index
         */
        void addAdditionalFace(int vertex1, int vertex2, int vertex3);

        /**
         * @brief buildMesh builds a chunk by generating a new mesh buffer
         *
         * By calling buildMesh(), the mesh of a new chunk is being created.
         * Before building a chunk, faces need to be added to this builder using the method addFace(index).
         * The vertex buffer of resulting mesh holds the vertices that got duplicated during the chunking
         * process at the first fields of the buffer followed by the normal vertices.
         * If the number of added faces is 0 this function will return an mesh buffer holding no vertices
         * or faces.
         *
         * @return mesh of the newly created chunk
         */
        MeshBufferPtr buildMesh();

        /**
         * @brief numFaces delivers the number of faces for the chunk
         *
         * This delivers the amount of faces currently added to a ChunkBuilder instance.
         * This functionality is especially useful for checking whether or not a generatec mesh would be
         * empty before calling buildMesh.
         *
         * @return number of faces added to this builder
         */
        unsigned int numFaces();

        /**
         * @brief numVertices amount of vertices ot the resulting mesh
         *
         * This delivers the amount of vertices that the resulting mesh would have when calling buildMesh.
         *
         * @return number of vertices added to this builder
         */
        unsigned int numVertices();
        
    private:
        // model that is being chunked
        MeshBufferPtr m_originalMesh = nullptr;

        // amount of added vertcices
        unsigned int m_numVertices = 0;

        // indices of vertices of this chunk that got dupliplicated during the chunking process
        std::vector<unsigned int> m_duplicateVertices;

        // indices of faces in original model
        std::vector<unsigned int> m_faces;

        // vertices that are not included in the original mesh buffer but are required to be added to the resulting mesh
        std::vector<std::shared_ptr<float[]>> m_additionalVertices;

        // faces that are not included in the original mesh buffer
        std::vector<std::shared_ptr<int[]>> m_additionalFaces;

        // one dynamic sized vector with ChunkBuilder ids for all vertices of the original mesh for duplicate detection
        std::shared_ptr<std::vector<std::vector<ChunkBuilderPtr>>> m_vertexUse;
};

} /* namespace lvr2 */

#endif // CHUNK_BUILDER_HPP
