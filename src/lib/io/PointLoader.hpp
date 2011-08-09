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
public:
    PointLoader()
        : m_points(0), m_pointNormals(0), m_pointColors(0), m_intensities(0), m_numPoints(0) {}


    /**
     * Returns a point array
     *
     * @param n     The number of loaded points
     * @return      The point data or a null pointer
     */
    virtual float**  getPointArray()
    {
        return m_points;
    }

    /**
     * Returns the point colors (RGB) for a point cloud
     *
     * @param n     The number of loaded color elements.
     * @return      The loaded color array or a null pointer of no vertices could be read
     */
    virtual unsigned char**  getPointColorArray()
    {
       return m_pointColors;
    }

    /**
     * Returns the point normals for a point cloud
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float**  getPointNormalArray()
    {
       return m_pointNormals;
    }

    /**
     * Returns the remission values for a point cloud (one float per point)
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getPointIntensityArray()
    {
       return m_intensities;
    }

    /**
     * Returns the number of loaded points
     */
    virtual size_t  getNumPoints()
    {
        return m_numPoints;
    }
protected:

    /// Point cloud data
    float**          m_points;

    /// Point normals
    float**          m_pointNormals;

    /// Color information
    unsigned char**  m_pointColors;

    /// Intensities
    float*           m_intensities;

    /// Number of loaded points
    size_t           m_numPoints;
};


#endif /* POINTIO_HPP_ */
