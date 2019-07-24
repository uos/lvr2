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

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2
{

MeshBuffer::MeshBuffer()
:base()
{
    m_numFaces = 0;
    m_numVertices = 0;
}

void MeshBuffer::setVertices(floatArr vertices, size_t n)
{
    if(n)
    {
        m_numVertices = n;
        this->addFloatChannel(vertices, "vertices", n, 3);
    }
}

void MeshBuffer::setVertexNormals(floatArr normals)
{
    if(m_numVertices)
    {
        this->addFloatChannel(normals, "vertex_normals", m_numVertices, 3);
    }
    else
    {
        cout << "MeshBuffer::setVertexNormals(): "
             << "Cannot add vertex normals without vertex definitions" << endl;
    }
}

void MeshBuffer::setVertexColors(ucharArr colors, size_t w)
{
    if(m_numVertices)
    {
        this->addUCharChannel(colors, "vertex_colors", m_numVertices, w);
    }
    else
    {
        cout << "MeshBuffer::setVertexColors(): "
             << "Cannot add vertex colors without vertex definitions" << endl;
    }
}

void MeshBuffer::setTextureCoordinates(floatArr coordinates)
{
    if(m_numVertices)
    {
        this->addFloatChannel(coordinates, "texture_coordinates", m_numVertices, 2);
    }
    else
    {
        cout << "MeshBuffer::setTextureCoordinates(): "
             << "Cannot add vertex colors without vertex definitions" << endl;
    }
}

void MeshBuffer::setFaceIndices(indexArray indices, size_t n)
{
    if(n)
    {
        m_numFaces = n;
        this->addIndexChannel(indices, "face_indices", n, 3);
    }
}

void MeshBuffer::setFaceMaterialIndices(indexArray indices)
{
    if(m_numFaces)
    {
        this->addIndexChannel(indices, "face_material_indices", m_numFaces, 1);
    }
    else
    {
        cout << "MeshBuffer::setFaceMaterialIndices(): "
             << "Cannot add material indices without face definitions" << endl;
    }
}

void MeshBuffer::setFaceNormals(floatArr normals)
{
    if(m_numFaces)
    {
        this->addFloatChannel(normals, "face_normals", m_numFaces, 3);
    }
    else
    {
        cout << "MeshBuffer::setFaceMaterialIndices(): "
             << "Cannot add material indices without face definitions" << endl;
    }
}

void MeshBuffer::setFaceColors(ucharArr colors, size_t w)
{
    if(m_numFaces)
    {
        this->addUCharChannel(colors, "face_colors", m_numFaces, w);
    }
    else
    {
        cout << "MeshBuffer::setFaceColors(): "
             << "Cannot add face colors without face definitions" << endl;
    }
}

size_t MeshBuffer::numVertices()
{
    return m_numVertices;
}

size_t MeshBuffer::numFaces()
{
    return m_numFaces;
}

floatArr MeshBuffer::getVertices()
{
    size_t n;
    size_t w;
    return this->getFloatArray("vertices", n, w);
}

ucharArr MeshBuffer::getVertexColors(size_t& w)
{
    size_t n;
    return this->getUCharArray("vertex_colors", n, w);

}

floatArr MeshBuffer::getVertexNormals()
{
    size_t n;
    size_t w;
    return this->getFloatArray("vertex_normals", n, w);
}

floatArr MeshBuffer::getFaceNormals()
{
    size_t n;
    size_t w;
    return this->getFloatArray("face_normals", n, w);
}

floatArr MeshBuffer::getTextureCoordinates()
{
    size_t n;
    size_t w;
    return this->getFloatArray("texture_coordinates", n, w);
}

indexArray MeshBuffer::getFaceIndices()
{
    size_t n;
    size_t w;
    return this->getIndexArray("face_indices", n, w);
}

ucharArr MeshBuffer::getFaceColors(size_t& w)
{
    size_t n;
    ucharArr arr = this->getUCharArray("face_colors", n, w);
    return arr;
}

indexArray MeshBuffer::getFaceMaterialIndices()
{
    size_t n;
    size_t w;
    return this->getIndexArray("face_material_indices", n, w);
}

vector<Texture>& MeshBuffer::getTextures()
{
    return m_textures;
}

vector<Material>& MeshBuffer::getMaterials()
{
    return m_materials;
}

bool MeshBuffer::hasFaceColors()
{
    UCharChannelOptional channel = this->getUCharChannel("face_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasVertexColors()
{
    UCharChannelOptional channel = this->getUCharChannel("vertex_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasFaceNormals()
{
    FloatChannelOptional channel = this->getFloatChannel("face_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasVertexNormals()
{
    FloatChannelOptional channel = this->getFloatChannel("vertex_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

}
