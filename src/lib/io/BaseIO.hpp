/**
 * BaseIO.hpp
 *
 *  @date 03.08.2011
 *  @author Thomas Wiemann
 */

#ifndef BASEIO_HPP_
#define BASEIO_HPP_

#include <cstdlib>

/**
 * @brief Interface specification for low-level io. All read
 *        elements are stored in linear arrays.
 */
class BaseIO
{
public:
    BaseIO() {}

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

    /**
     * Returns a point array
     *
     * @param n     The number of loaded points
     * @return      The point data or a null pointer
     */
    virtual float*  getPointArray(size_t &n) = 0;

    /**
         * Returns the point colors (RGB) for a point cloud
         *
         * @param n     The number of loaded color elements.
         * @return      The loaded color array or a null pointer of no vertices could be read
         */
    virtual float*  getPointColorArray(size_t &n) = 0;

    /**
     * Returns the point normals for a point cloud
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getPointNormalArray(size_t &n) = 0;

    /**
     * Returns the remission values for a point cloud (one float per point)
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getPointIntensityArray(size_t &n) = 0;
};

#endif /* BASEIO_HPP_ */
