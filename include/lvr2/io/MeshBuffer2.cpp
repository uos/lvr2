#include <lvr2/io/MeshBuffer2.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <iostream>
using std::cout;
using std::endl;

namespace lvr2
{

MeshBuffer2::MeshBuffer2()
{
    m_numFaces = 0;
    m_numVertices = 0;
}

void MeshBuffer2::setVertices(floatArr vertices, size_t n)
{
    if(n)
    {
        m_numVertices = n;
        m_channels.addFloatChannel(vertices, "vertices", n, 3);
    }
}

void MeshBuffer2::setVertexNormals(floatArr normals)
{
    if(m_numVertices)
    {
        m_channels.addFloatChannel(normals, "vertex_normals", m_numVertices, 3);
    }
    else
    {
        cout << "MeshBuffer2::setVertexNormals(): "
             << "Cannot add vertex normals without vertex definitions" << endl;
    }
}

void MeshBuffer2::setVertexColors(ucharArr colors, unsigned w)
{
    if(m_numVertices)
    {
        m_channels.addUCharChannel(colors, "vertex_colors", m_numVertices, w);
    }
    else
    {
        cout << "MeshBuffer2::setVertexColors(): "
             << "Cannot add vertex colors without vertex definitions" << endl;
    }
}

void MeshBuffer2::setTextureCoordinates(floatArr coordinates)
{
    if(m_numVertices)
    {
        m_channels.addFloatChannel(coordinates, "texture_coordinates", m_numVertices, 2);
    }
    else
    {
        cout << "MeshBuffer2::setTextureCoordinates(): "
             << "Cannot add vertex colors without vertex definitions" << endl;
    }
}

void MeshBuffer2::setFaceIndices(indexArray indices, size_t n)
{
    if(n)
    {
        m_numFaces = n;
        m_channels.addIndexChannel(indices, "face_indices", n, 3);
    }
}

void MeshBuffer2::setFaceMaterialIndices(indexArray indices)
{
    if(m_numFaces)
    {
        m_channels.addIndexChannel(indices, "face_material_indices", m_numFaces, 1);
    }
    else
    {
        cout << "MeshBuffer2::setFaceMaterialIndices(): "
             << "Cannot add material indices without face definitions" << endl;
    }
}

void MeshBuffer2::setFaceNormals(floatArr normals)
{
    if(m_numFaces)
    {
        m_channels.addFloatChannel(normals, "face_normals", m_numFaces, 3);
    }
    else
    {
        cout << "MeshBuffer2::setFaceMaterialIndices(): "
             << "Cannot add material indices without face definitions" << endl;
    }
}

void MeshBuffer2::setFaceColors(ucharArr colors, unsigned w)
{
    if(m_numFaces)
    {
        m_channels.addUCharChannel(colors, "face_colors", m_numFaces, w);
    }
    else
    {
        cout << "MeshBuffer2::setFaceColors(): "
             << "Cannot add face colors without face definitions" << endl;
    }
}

size_t MeshBuffer2::numVertices()
{
    return m_numVertices;
}

size_t MeshBuffer2::numFaces()
{
    return m_numFaces;
}

floatArr MeshBuffer2::getVertices()
{
    size_t n;
    unsigned w;
    return m_channels.getFloatArray(n, w, "vertices");
}

ucharArr MeshBuffer2::getVertexColors(unsigned& w)
{
    size_t n;
    return m_channels.getUCharArray(n, w, "vertex_colors");

}

floatArr MeshBuffer2::getVertexNormals()
{
    size_t n;
    unsigned w;
    return m_channels.getFloatArray(n, w, "vertex_normals");
}

floatArr MeshBuffer2::getTextureCoordinates()
{
    size_t n;
    unsigned w;
    return m_channels.getFloatArray(n, w, "texture_coordinates");
}

indexArray MeshBuffer2::getFaceIndices()
{
    size_t n;
    unsigned w;
    return m_channels.getIndexArray(n, w, "face_indices");
}

ucharArr MeshBuffer2::getFaceColors(unsigned& w)
{
    size_t n;
    ucharArr arr = m_channels.getUCharArray(n, w, "face_colors");
    return arr;
}

indexArray MeshBuffer2::getFaceMaterialIndices()
{
    size_t n;
    unsigned w;
    return m_channels.getIndexArray(n, w, "face_material_indices");
}

vector<Texture>& MeshBuffer2::getTextures()
{
    return m_textures;
}

vector<Material>& MeshBuffer2::getMaterials()
{
    return m_materials;
}

bool MeshBuffer2::hasFaceColors()
{
    UCharChannelOptional channel = m_channels.getUCharChannel("face_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer2::hasVertexColors()
{
    UCharChannelOptional channel = m_channels.getUCharChannel("vertex_colors");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer2::hasFaceNormals()
{
    FloatChannelOptional channel = m_channels.getFloatChannel("face_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

bool MeshBuffer2::hasVertexNormals()
{
    FloatChannelOptional channel = m_channels.getFloatChannel("vertex_normals");
    if(channel)
    {
        return true;
    }
    return false;
}

}
