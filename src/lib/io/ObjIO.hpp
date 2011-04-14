/*
 * ObjIO.h
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef OBJIO_HPP_
#define OBJIO_HPP_

namespace lssr
{

/**
 * @brief A basic implementation of the obj file format. Currently
 *        only geometry information is supported. Color / Material
 *        support will follow shortly.
 */
template<typename CoordType, typename IndexType>
class ObjIO
{
public:
    ObjIO();

    void write(string filename);
    void setVertexArray(CoordType* array, size_t count);
    void setNormalArray(CoordType* array, size_t count);
    void setIndexArray(IndexType* array, size_t count);

private:
    CoordType*              m_vertices;
    CoordType*              m_normals;
    IndexType*              m_indices;

    size_t                  m_faceCount;
    size_t                  m_vertexCount;

};

}

#include "ObjIO.tcc"

#endif /* OBJIO_H_ */
