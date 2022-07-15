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

#ifndef MESHBUFFER2_HPP
#define MESHBUFFER2_HPP

#include "lvr2/texture/Material.hpp"
#include "lvr2/texture/Texture.hpp"
#include "lvr2/types/BaseBuffer.hpp"

namespace lvr2
{

////
/// \brief The MeshBuffer Mesh representation for I/O modules.
///
class MeshBuffer : public BaseBuffer
{
    using base = BaseBuffer;
public:

    ///
    /// \brief MeshBuffer      Contructor. Builds an empty buffer. Fill elements
    ///                         with add-Methods.
    ///
    MeshBuffer();

    bool removeVertices(){
        return this->removeFloatChannel("vertices");
    }

    ///
    /// \brief addVertices      Adds the vertex array. Three floats per vertex
    /// \param vertices         The vertex array
    /// \param n                Number of vertices
    ///
    void setVertices(floatArr vertices, size_t n);

    ///
    /// \brief addVertexNormals Adds vertex normals.
    /// \param normals          Normal defintion. Three floats per vertex.
    ///
    void setVertexNormals(floatArr normals);

    ///
    /// \brief addVertexColors  Adds vertex color information.
    /// \param colors           Vertex color array
    /// \param w                Number of bytes per color. (3 for RGB, 4 for RGBA)
    ///
    void setVertexColors(ucharArr colors, size_t w = 3);

    ///
    /// \brief addTextureCoordinates    Adds texture coordinates for vertices
    /// \param coordinates      Texture coordinate definitions (2 floats per vertex)
    ///
    void setTextureCoordinates(floatArr coordinates);

    ///
    /// \brief addFaceIndices   Adds the face index array that references to the
    ///                         vertex array
    /// \param indices          The index array (3 indices per face)
    /// \param n                Number of faces
    ///
    void setFaceIndices(indexArray indices, size_t n);

    ///
    /// \brief addFaceMaterialIndices   Adds face material indices. The array references
    ///                         to material definitions in \ref m_materials.
    ///
    /// \param indices          One material index per face
    ///
    void setFaceMaterialIndices(indexArray indices);

    ///
    /// \brief addFaceNormals   Adds face normal information. The number of normals
    ///                         in the array are exspected to match the number of
    ///                         faces in the mesh
    /// \param                  Normal definitions for all faces
    ///
    void setFaceNormals(floatArr normals);

    ///
    /// \brief addFaceColors    Adds face colors the the buffer
    /// \param colors           An array containing color information
    /// \param w                Bytes per color attribute (3 for RGB, 4 for RGBA)
    ///
    void setFaceColors(ucharArr colors, size_t w = 3);

    void setTextures(std::vector<Texture>& textures)
    {
        m_textures = std::move(textures);
    }

    void setMaterials(std::vector<Material>& materials)
    {
        m_materials = std::move(materials);
    }

    ///
    /// \brief numVertices      Number of vertices in the mesh
    ///
    size_t numVertices() const;

    ///
    /// \brief numFaces         Number of faces in the mesh
    ///
    size_t numFaces() const;


    ///
    /// \brief getVertices      Return the vertex array.
    ///
    floatArr getVertices();

    ///
    /// \brief getVertexColors  Returns vertex color information or an empty array if
    ///                         vertex colors are not available
    /// \param width            Number of bytes per color (3 for RGB, 4 for RGBA)
    /// \return
    ///
    ucharArr getVertexColors(size_t& width);

    ///
    /// \brief getVertexNormals Returns an array with vertex normals or an empty array
    ///                         if no normals are present.
    ///
    floatArr getVertexNormals();

    ///
    /// \brief getTextureCoordinates Returns an array with texture coordinates. Two
    ///                         normalized floats per vertex. Returns an empty array
    ///                         if no texture coordinates were loaded.
    ///
    floatArr getTextureCoordinates();

    ///
    /// \brief getFaceNormas    Returns an array containing face normals, i.e., three
    ///                         float values per face.
    ///
    floatArr getFaceNormals();

    ///
    /// \brief getFaceIndices   Returns an array with face definitions, i.e., three
    ///                         vertex indices per face.
    indexArray getFaceIndices();

    ///
    /// \brief getFaceColors    Returns an array with wrgb colors
    /// \param width            Number of bytes per color (3 for RGB and 4 for RGBA)
    /// \return                 An array containing point data or an nullptr if
    ///                         no colors are present.
    ///
    ucharArr getFaceColors(size_t& width);

    ///
    /// \brief getFaceMaterialIndices   Returns an array with face material indices 
    ///
    indexArray getFaceMaterialIndices();

    ///
    /// \brief getTextures      Returns a vector with textures
    ///
    std::vector<Texture>& getTextures();

    ///
    /// \brief getTextures      Returns a vector with materials
    ///
    std::vector<Material>& getMaterials();

    bool hasVertices() const;

    bool hasFaces() const;

    bool hasFaceColors() const;

    bool hasVertexColors() const;

    bool hasFaceNormals() const;

    bool hasVertexNormals() const;

    /// TODO: CHANNEL BASED SETTER / GETTER!

private:

    /// Vector containing all material definitions
    std::vector<Material>    m_materials;

    /// Vector containing all textures
    std::vector<Texture>     m_textures;
};

using MeshBufferPtr = std::shared_ptr<MeshBuffer>;

}
#endif // MESHBUFFER2_HPP
