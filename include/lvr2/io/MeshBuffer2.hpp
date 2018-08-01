#ifndef MESHBUFFER2_HPP
#define MESHBUFFER2_HPP

#include <lvr2/io/BaseBuffer.hpp>
#include <lvr2/texture/Material.hpp>
#include <lvr2/texture/Texture.hpp>

class MeshBuffer2 : public BaseBuffer
{
public:
    MeshBuffer2();

    void addVertices(floatArr vertices, size_t n);
    void addVertexNormals(floatArr normals);
    void addVertexColors(ucharArr colors, unsigned w = 3);
    void addTextureCoordinates(floatArr coordinates);

    void addFaceIndices(indexArray indices, size_t n);
    void addFaceMaterialIndices(indexArray indices);
    void addFaceNormals(floatArr normals);
    void addFaceColors(ucharArr colors, unsigned w = 3);

    void numVertices();
    void numFaces();

    floatArr getVertices();
    ucharArr getVertexColors(unsigned& width);
    floatArr getVertexNormals();
    floatArr getTextureCoordinates();
    indexArray getFaceIndices();
    ucharArr getFaceColors(unsigned& width);

private:
    AttributeChannel    m_vertexAttributes;
    AttributeChannel    m_faceAttributes;

    vector<Material>    m_materials;
    vector<Texture>

};

#endif // MESHBUFFER2_HPP
