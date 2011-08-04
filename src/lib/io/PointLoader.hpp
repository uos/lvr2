/**
 * PointIO.hpp
 *
 *  @date 04.08.2011
 *  @author Thomas Wiemann
 */

#ifndef POINTIO_HPP_
#define POINTIO_HPP_

class PointLoader
{
    /**
     * Returns a point array
     *
     * @param n     The number of loaded points
     * @return      The point data or a null pointer
     */
    virtual float**  getPointArray(size_t &n) = 0;

    /**
     * Returns the point colors (RGB) for a point cloud
     *
     * @param n     The number of loaded color elements.
     * @return      The loaded color array or a null pointer of no vertices could be read
     */
    virtual unsigned char**  getPointColorArray(size_t &n) = 0;

    /**
     * Returns the point normals for a point cloud
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float**  getPointNormalArray(size_t &n) = 0;

    /**
     * Returns the remission values for a point cloud (one float per point)
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getPointIntensityArray(size_t &n) = 0;
};


#endif /* POINTIO_HPP_ */
