#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2
{

MeshBuffer::MeshBuffer()
{
    m_numFaces = 0;
    m_numVertices = 0;
}

void MeshBuffer::setVertices(floatArr vertices, size_t n)
{
    if(n)
    {
        m_numVertices = n;
        m_channels.addFloatChannel(vertices, "vertices", n, 3);
    }
}

void MeshBuffer::setVertexNormals(floatArr normals)
{
    if(m_numVertices)
    {
        m_channels.addFloatChannel(normals, "vertex_normals", m_numVertices, 3);
    }
    else
    {
        cout << "MeshBuffer::setVertexNormals(): "
             << "Cannot add vertex normals without vertex definitions" << endl;
    }
}

void MeshBuffer::setVertexColors(ucharArr colors, unsigned w)
{
    if(m_numVertices)
    {
        m_channels.addUCharChannel(colors, "vertex_colors", m_numVertices, w);
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
        m_channels.addFloatChannel(coordinates, "texture_coordinates", m_numVertices, 2);
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
        m_channels.addIndexChannel(indices, "face_indices", n, 3);
    }
}

void MeshBuffer::setFaceMaterialIndices(indexArray indices)
{
    if(m_numFaces)
    {
        m_channels.addIndexChannel(indices, "face_material_indices", m_numFaces, 1);
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
        m_channels.addFloatChannel(normals, "face_normals", m_numFaces, 3);
    }
    else
    {
        cout << "MeshBuffer::setFaceMaterialIndices(): "
             << "Cannot add material indices without face definitions" << endl;
    }
}

void MeshBuffer::setFaceColors(ucharArr colors, unsigned w)
{
    if(m_numFaces)
    {
        m_channels.addUCharChannel(colors, "face_colors", m_numFaces, w);
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
    unsigned w;
    return m_channels.getFloatArray(n, w, "vertices");
}

ucharArr MeshBuffer::getVertexColors(unsigned& w)
{
    size_t n;
    return m_channels.getUCharArray(n, w, "vertex_colors");

}

floatArr MeshBuffer::getVertexNormals()
{
    size_t n;
    unsigned w;
    return m_channels.getFloatArray(n, w, "vertex_normals");
}

floatArr MeshBuffer::getTextureCoordinates()
{
    size_t n;
    unsigned w;
    return m_channels.getFloatArray(n, w, "texture_coordinates");
}

indexArray MeshBuffer::getFaceIndices()
{
    size_t n;
    unsigned w;
    return m_channels.getIndexArray(n, w, "face_indices");
}

ucharArr MeshBuffer::getFaceColors(unsigned& w)
{
    size_t n;
    ucharArr arr = m_channels.getUCharArray(n, w, "face_colors");
    return arr;
}

indexArray MeshBuffer::getFaceMaterialIndices()
{
    size_t n;
    unsigned w;
    return m_channels.getIndexArray(n, w, "face_material_indices");
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
    UCharChannelOptional channel = m_channels.getUCharChannel("face_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasVertexColors()
{
    UCharChannelOptional channel = m_channels.getUCharChannel("vertex_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasFaceNormals()
{
    FloatChannelOptional channel = m_channels.getFloatChannel("face_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer::hasVertexNormals()
{
    FloatChannelOptional channel = m_channels.getFloatChannel("vertex_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

}
