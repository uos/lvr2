/**
 * MeshIO.hpp
 *
 *  @date 04.08.2011
 *  @author Thomas Wiemann
 */

#ifndef MESHIO_HPP_
#define MESHIO_HPP_


#include <cstring>

namespace lssr
{

class MeshLoader
{
public:
    /**
     * Returns the vertices of triangle mesh.
     *
     * @param n     The number of loaded points.
     * @return      The loaded vertex array or a null pointer of no vertices could be read.
     */
    virtual float*  getVertexArray(size_t &n) = 0;

    /**
     * Returns the vertex normals for a triangle mesh
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getVertexNormalArray(size_t &n) = 0;

    /**
     * Returns the vertex colors (RGB) for a triangle mesh
     *
     * @param n     The number of loaded color elements.
     * @return      The loaded color array or a null pointer of no vertices could be read
     */
    virtual float*  getVertexColorArray(size_t &n) = 0;

    /**
     * Returns the index buffer of a a triangle mesh
     *
     * @param n     The number of face in the triangle mesh
     * @return      The loaded index array or a null pointer
     */
    virtual unsigned int* getIndexArray(size_t &n) = 0;


protected:

    /**
      * Ctor.
      */
    MeshLoader() : m_vertices(0),
                   m_vertexNormals(0),
                   m_vertexColors(0),
                   m_indices(0),
                   m_numberOfNormals(0),
                   m_numberOfVertices(0),
                   m_numberOfFaces(0)
    {}

    float*                  m_vertices;
    float*                  m_vertexNormals;
    float*                  m_vertexColors;
    unsigned int*           m_indices;

    size_t                  m_numberOfNormals;
    size_t                  m_numberOfVertices;
    size_t                  m_numberOfFaces;

};

} // namespace lssr

#endif /* MESHIO_HPP_ */
