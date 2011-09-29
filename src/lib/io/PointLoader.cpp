/**
 *
 * @file      PointLoader.cpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 11:23:57 PM
 *
 **/

#include "PointLoader.hpp"

namespace lssr
{

PointLoader::PointLoader() :
    m_points( NULL ),
    m_pointNormals( NULL ),
    m_pointColors( NULL ),
    m_pointIntensities( NULL ),
    m_pointConfidence( NULL ),
    m_indexedPoints( NULL ),
    m_indexedPointNormals( NULL ),
    m_indexedPointIntensities( NULL ),
    m_indexedPointConfidence( NULL ),
    m_indexedPointColors( NULL ),
    m_numPoints( 0 ),
    m_numPointColors( 0 ),
    m_numPointNormals( 0 ),
    m_numPointIntensities( 0 ),
    m_numPointConfidence( 0 ) {}


float* PointLoader::getPointArray( size_t &n )
{

    n = m_numPoints;
    return m_points;

}


uint8_t* PointLoader::getPointColorArray( size_t &n )
{

    n = m_numPointColors;
    return m_pointColors;

}


float* PointLoader::getPointNormalArray( size_t &n )
{

    n = m_numPointNormals;
    return m_pointNormals;

}


float* PointLoader::getPointIntensityArray( size_t &n )
{

    n = m_numPointIntensities;
    return m_pointIntensities;

}


float* PointLoader::getPointConfidenceArray( size_t &n )
{

    n = m_numPointConfidence;
    return m_pointConfidence;

}


size_t PointLoader::getNumPoints()
{

    return m_numPoints;

}


uint8_t** PointLoader::getIndexedPointColorArray( size_t &n )
{

    n = m_numPointColors;
    if ( !m_pointColors )
    {
        return NULL;
    }

    if ( !m_indexedPointColors )
    {
        m_indexedPointColors = (uint8_t**) 
            malloc( m_numPointColors * sizeof(uint8_t*) );
        for ( size_t i = 0; i < m_numPointColors; i++ )
        {
            m_indexedPointColors[i] = m_pointColors + ( i * 3 );
        }
    }
    return m_indexedPointColors;

}


float** PointLoader::getIndexedPointNormalArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointNormals, &m_pointNormals, 
            &m_indexedPointNormals );

}

float** PointLoader::getIndexedPointArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPoints, &m_points, &m_indexedPoints );

}


float** PointLoader::getIndexedPointIntensityArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointIntensities, &m_pointIntensities,
            &m_indexedPointIntensities, 1 );

}


float** PointLoader::getIndexedPointConfidenceArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointConfidence, &m_pointConfidence,
            &m_indexedPointConfidence, 1 );

}


float** PointLoader::getIndexedArrayf( size_t &n, const size_t num, 
        float** arr1d, float*** arr2d, const int step )
{

    n = num;

    /* Return NULL if we have no data. */
    if ( !(*arr1d) )
    {
        return NULL;
    }

    /* Generate indexed intensity array in not already done. */
    if ( !(*arr2d) )
    {
        *arr2d = (float**) malloc( num * sizeof(float*) );
        for ( size_t i = 0; i < num; i++ )
        {
            (*arr2d)[i] = (*arr1d) + ( i * step );
        }
    }

    /* Return indexed intensity array */
    return *arr2d;

}


void PointLoader::setPointArray( float* array, size_t n )
{

    m_numPoints = n;
    m_points = array;

}


void PointLoader::setPointColorArray( uint8_t* array, size_t n )
{

    m_numPointColors = n;
    m_pointColors = array;

}


void PointLoader::setPointNormalArray( float* array, size_t n )
{

    m_numPointNormals = n;
    m_pointNormals = array;

}


void PointLoader::setPointIntensityArray( float* array, size_t n )
{

    m_numPointIntensities = n;
    m_pointIntensities = array;

}


void PointLoader::setPointConfidenceArray( float* array, size_t n )
{

    m_numPointConfidence = n;
    m_pointConfidence = array;

}

} /* namespace lssr */
