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
    m_numPoints( 0 ),
    m_numPointColors( 0 ),
    m_numPointNormals( 0 ),
    m_numPointIntensities( 0 ),
    m_numPointConfidence( 0 )
    {
        /* coordf must be the exact size of three floats to cast the float
         * array to a coordf array. */
        assert( 3 * sizeof(float) == sizeof(coord<float>) );
        assert( 3 * sizeof(uchar) == sizeof(color<uchar>) );
        assert( sizeof(float) == sizeof(idxVal<float>) );

        m_points.reset();
        m_pointConfidences.reset();
        m_pointIntensities.reset();
        m_pointNormals.reset();
		m_pointColors.reset();
    }


floatArr PointBuffer::getPointArray( size_t &n )
{

    n = m_numPoints;
    return m_points;

}


ucharArr PointBuffer::getPointColorArray( size_t &n )
{

    n = m_numPointColors;
    return m_pointColors;

}


floatArr PointBuffer::getPointNormalArray( size_t &n )
{

    n = m_numPointNormals;
    return m_pointNormals;

}


floatArr PointBuffer::getPointIntensityArray( size_t &n )
{

    n = m_numPointIntensities;
    return m_pointIntensities;

}


floatArr PointBuffer::getPointConfidenceArray( size_t &n )
{

    n = m_numPointConfidence;
    return m_pointConfidences;

}


size_t PointBuffer::getNumPoints()
{

    return m_numPoints;

}


color3bArr PointBuffer::getIndexedPointColorArray( size_t &n )
{

    n = m_numPointColors;
    color3bArr p = *((color3bArr*) &m_pointColors);
    return p;

}


coord3fArr PointBuffer::getIndexedPointNormalArray( size_t &n )
{

    n = m_numPointNormals;
    coord3fArr p = *((coord3fArr*) &m_pointNormals);
    return p;

}

coord3fArr PointBuffer::getIndexedPointArray( size_t &n )
{

    n = m_numPoints;
    coord3fArr p = *((coord3fArr*) &m_points);
    return p;

}


idx1fArr PointBuffer::getIndexedPointIntensityArray( size_t &n )
{

    n = m_numPointIntensities;
    idx1fArr p = *((idx1fArr*) &m_pointIntensities);
    return p;

}


idx1fArr PointBuffer::getIndexedPointConfidenceArray( size_t &n )
{

    n = m_numPointConfidence;
    idx1fArr p = *((idx1fArr*) &m_pointConfidences);
    return p;

}


void PointBuffer::setPointArray( floatArr array, size_t n )
{

    m_numPoints = n;
    m_points = array;

}


void PointBuffer::setPointColorArray( ucharArr array, size_t n )
{

    m_numPointColors = n;
    m_pointColors = array;

}

void PointBuffer::setIndexedPointColorArray( color3bArr array, size_t n )
{

    m_numPointColors = n;
    m_pointColors = *((ucharArr *) &array);

}


void PointBuffer::setPointNormalArray( floatArr array, size_t n )
{

    m_numPointNormals = n;
    m_pointNormals = array;

}


void PointBuffer::setPointIntensityArray( floatArr array, size_t n )
{

    m_numPointIntensities = n;
    m_pointIntensities = array;

}


void PointBuffer::setPointConfidenceArray( floatArr array, size_t n )
{

    m_numPointConfidence = n;
    m_pointConfidences = array;

}


void PointBuffer::freeBuffer()
{
    m_pointConfidences.reset();
    m_pointIntensities.reset();
    m_pointNormals.reset();
    m_points.reset();
    m_pointColors.reset();
    m_numPoints = m_numPointColors = m_numPointIntensities
        = m_numPointConfidence = m_numPointNormals = 0;

}

void PointBuffer::defineSubCloud(indexPair& range)
{
    m_subClouds.push_back(range);
}


} /* namespace lssr */
