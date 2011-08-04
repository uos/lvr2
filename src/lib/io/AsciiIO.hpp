/*
 * AsciiIO.h
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef ASCIIIO_H_
#define ASCIIIO_H_

#include "BaseIO.hpp"
#include "PointLoader.hpp"

namespace lssr
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *        .3d and .pts are supported.
 */
class AsciiIO : public BaseIO, public PointLoader
{
public:

    AsciiIO();

    /**
     * @brief Reads the given file and stores point and normal
     *        information in the given parameters
     *
     * @param filename      The file to read
     */
    void read(string filename);

    /// TODO: Coordinate mapping for ascii files
    static size_t countLines(string filename);

    /**
     * @brief Helper method. Returns the number of columns in the
     *        given file.
     */
    static int getEntriesInLine(string filename);

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

    virtual void save(string filename)
    {

    }

private:

    float**  m_points;
    unsigned char**  m_colors;
    float*   m_intensities;

    size_t  m_numPoints;
};


} // namespace lssr


#endif /* ASCIIIO_H_ */
