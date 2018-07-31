#ifndef MESHBUFFER2_HPP
#define MESHBUFFER2_HPP

#include <lvr2/io/BaseBuffer.hpp)>

class MeshBuffer2 : public BaseBuffer
{
public:
    MeshBuffer2();

    void addVertexNormals(floatArr normals);
    void addVertexColors(ucharArr colors, unsigned w = 3);

    void addFaceNormals(floatArr normals);
    void addFaceColors(ucharArr colors, unsigned w = 3);


private:
    AttributeChannel    m_vertexAttributes;
    AttributeChannel    m_faceAttributes;
};

#endif // MESHBUFFER2_HPP
