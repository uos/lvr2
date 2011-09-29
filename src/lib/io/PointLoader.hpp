/**
 *
 * @file      PointLoader.hpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 11:23:57 PM
 *
 **/

#ifndef POINTIO_HPP_
#define POINTIO_HPP_

#include <stdint.h>
#include <cstddef>
#include <cstdlib>

namespace lssr
{


class PointLoader {

    public:
        PointLoader();

        void setPointArray( float* array, size_t n );
        void setPointColorArray( uint8_t* array, size_t n );
        void setPointNormalArray( float* array, size_t n );
        void setPointIntensityArray( float* array, size_t n );
        void setPointConfidenceArray( float* array, size_t n );

        virtual float* getPointArray( size_t &n );
        virtual uint8_t* getPointColorArray( size_t &n );
        virtual float* getPointNormalArray( size_t &n );
        virtual float* getPointIntensityArray( size_t &n );
        virtual float* getPointConfidenceArray( size_t &n );

        float** getIndexedPointArray( size_t &n );
        uint8_t** getIndexedPointColorArray( size_t &n );
        float** getIndexedPointNormalArray( size_t &n );
        float** getIndexedPointIntensityArray( size_t &n );
        float** getIndexedPointConfidenceArray( size_t &n );

        virtual size_t getNumPoints();

    protected:

        float** getIndexedArrayf( size_t &n, const size_t num, float** arr1d,
                float*** arr2d );

        float*    m_points;
        float*    m_pointNormals;
        uint8_t*  m_pointColors;
        float*    m_pointIntensities;
        float*    m_pointConfidence;

        float**   m_indexedPoints;
        float**   m_indexedPointNormals;
        float**   m_indexedPointIntensities;
        float**   m_indexedPointConfidence;
        uint8_t** m_indexedPointColors;

        size_t    m_numPoints;
        size_t    m_numPointColors;
        size_t    m_numPointNormals;
        size_t    m_numPointIntensities;
        size_t    m_numPointConfidence;

};

} /* namespace lssr */

#endif /* POINTIO_HPP_ */
