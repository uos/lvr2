/**
 * MeshIO.hpp
 *
 *  @date 04.08.2011
 *  @author Thomas Wiemann
 */

#ifndef MESHIO_HPP_
#define MESHIO_HPP_

class MeshLoader
{
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
};

#endif /* MESHIO_HPP_ */
