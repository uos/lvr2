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
        : m_points(0), m_colors(0), m_intensities(0), m_numPoints(0) {}


    /**
     * Returns a point array
     *
     * @param n     The number of loaded points
     * @return      The point data or a null pointer
     */
    virtual float**  getPointArray(size_t &n)
    {
        n = m_numPoints;
        return m_points;
    }

    /**
     * Returns the point colors (RGB) for a point cloud
     *
     * @param n     The number of loaded color elements.
     * @return      The loaded color array or a null pointer of no vertices could be read
     */
    virtual unsigned char**  getPointColorArray(size_t &n)
    {
        if(m_colors)
        {
            return m_colors;
            n = 0;
        }
        else
        {
            n = 0;
            return 0;
        }
    }

    /**
     * Returns the point normals for a point cloud
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float**  getPointNormalArray(size_t &n)
    {
        n = 0;
        return 0;
    }

    /**
     * Returns the remission values for a point cloud (one float per point)
     *
     * @param n     The number of loaded normals.
     * @return      The loaded normal array or a null pointer of no vertices could be read
     */
    virtual float*  getPointIntensityArray(size_t &n)
    {
        if(m_intensities)
        {
            n = m_numPoints;
            return m_intensities;
        }
        else
        {
            n = 0;
            return 0;
        }

    }

protected:

    /// Point cloud data
    float**          m_points;

    /// Color information
    unsigned char**  m_colors;

    /// Intensities
    float*           m_intensities;

    /// Number of loaded points
    size_t           m_numPoints;
};


#endif /* POINTIO_HPP_ */
