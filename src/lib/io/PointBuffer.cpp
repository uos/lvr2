/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /**
 *
 * @file      PointLoader.cpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @author    Thomas Wiemann
 *
 **/

#include "PointBuffer.hpp"

namespace lssr
{

PointBuffer::PointBuffer() :
    m_points( NULL ),
    m_pointNormals( NULL ),
    m_pointColors( NULL ),
    m_pointIntensities( NULL ),
    m_pointConfidences( NULL ),
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


float* PointBuffer::getPointArray( size_t &n )
{

    n = m_numPoints;
    return m_points;

}


unsigned char* PointBuffer::getPointColorArray( size_t &n )
{

    n = m_numPointColors;
    return m_pointColors;

}


float* PointBuffer::getPointNormalArray( size_t &n )
{

    n = m_numPointNormals;
    return m_pointNormals;

}


float* PointBuffer::getPointIntensityArray( size_t &n )
{

    n = m_numPointIntensities;
    return m_pointIntensities;

}


float* PointBuffer::getPointConfidenceArray( size_t &n )
{

    n = m_numPointConfidence;
    return m_pointConfidences;

}


size_t PointBuffer::getNumPoints()
{

    return m_numPoints;

}


unsigned char** PointBuffer::getIndexedPointColorArray( size_t &n )
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


float** PointBuffer::getIndexedPointNormalArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointNormals, &m_pointNormals, 
            &m_indexedPointNormals );

}

float** PointBuffer::getIndexedPointArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPoints, &m_points, &m_indexedPoints );

}


float** PointBuffer::getIndexedPointIntensityArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointIntensities, &m_pointIntensities,
            &m_indexedPointIntensities, 1 );

}


float** PointBuffer::getIndexedPointConfidenceArray( size_t &n )
{

    return getIndexedArrayf( n, m_numPointConfidence, &m_pointConfidences,
            &m_indexedPointConfidence, 1 );

}


float** PointBuffer::getIndexedArrayf( size_t &n, const size_t num, 
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


void PointBuffer::setPointArray( float* array, size_t n )
{

    m_numPoints = n;
    m_points = array;

}


void PointBuffer::setPointColorArray( uint8_t* array, size_t n )
{

    m_numPointColors = n;
    m_pointColors = array;

}


void PointBuffer::setPointNormalArray( float* array, size_t n )
{

    m_numPointNormals = n;
    m_pointNormals = array;

}


void PointBuffer::setPointIntensityArray( float* array, size_t n )
{

    m_numPointIntensities = n;
    m_pointIntensities = array;

}


void PointBuffer::setPointConfidenceArray( float* array, size_t n )
{

    m_numPointConfidence = n;
    m_pointConfidences = array;

}


void PointBuffer::freeBuffer()
{
    /// TODO: Memory leak in PointBuffer
    m_points = m_pointConfidences = m_pointIntensities = m_pointNormals = NULL;
    m_pointColors = NULL;
    m_numPoints = m_numPointColors = m_numPointIntensities
        = m_numPointConfidence = m_numPointNormals = 0;

}

void PointBuffer::defineSubCloud(indexPair& range)
{
    m_subClouds.push_back(range);
}


} /* namespace lssr */
