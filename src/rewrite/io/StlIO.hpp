/*
 * StlIO.h
 *
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef STLIO_H_
#define STLIO_H_

#include <string>
using std::string;

#include "Vertex.hpp"
#include "Normal.hpp"

namespace lssr
{

template class Normal<float>;

/***
 * @brief An Import / Export interface for ASCII STL files
 */

/// TODO: Write import for stl files.
template<typename CoordType, typename IndexType>
class StlIO
{
public:
    StlIO();

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


} // namespace lssr

#include "StlIO.tcc"

#endif /* STLIO_H_ */
