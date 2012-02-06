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
 * @file      MeshIO.cpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110928
 * @date      09/22/2011 09:16:36 PM
 *
 **/
#include "MeshBuffer.hpp"

namespace lssr
{

MeshBuffer::MeshBuffer() : 
    m_numVertices( 0 ),
    m_numVertexNormals( 0 ),
    m_numVertexColors( 0 ),
    m_numVertexConfidences( 0 ),
    m_numVertexIntensities( 0 ),
    m_numFaces( 0 ),
    m_numVertexTextureCoordinates( 0 ),
    m_numFaceTextureIndices( 0 ),
    m_numFaceColors ( 0 )
{

    /* Make shure we can convert the indexed arrays into interlaced arrays and
     * vice versa. */
	assert( 3 * sizeof(float) == sizeof(coord<float>) );
	assert( 3 * sizeof(uchar) == sizeof(color<uchar>) );
	assert( 3 * sizeof(unsigned int) == sizeof(coord<unsigned int>) );
	assert( sizeof(float) == sizeof(idxVal<float>) );

    /* Reset all internal buffer. */
    m_faceIndices.reset();
    m_faceColors.reset();
    m_faceTextureIndices.reset();
    m_vertexColors.reset();
    m_vertexConfidence.reset();
    m_vertexIntensity.reset();
    m_vertexNormals.reset();
    m_vertexTextureCoordinates.reset();
    m_vertices.reset();

}


floatArr MeshBuffer::getVertexArray( size_t &n )
{
    n = m_numVertices;
    return m_vertices;
}


floatArr MeshBuffer::getVertexNormalArray( size_t &n )
{
    n = m_numVertexNormals;
    return m_vertexNormals;
}


ucharArr MeshBuffer::getVertexColorArray( size_t &n )
{
    n = m_numVertexColors;
    return m_vertexColors;
}


floatArr MeshBuffer::getVertexConfidenceArray( size_t &n )
{
    n = m_numVertexConfidences;
    return m_vertexConfidence;
}


floatArr MeshBuffer::getVertexIntensityArray( size_t &n )
{
    n = m_numVertexIntensities;
    return m_vertexIntensity;
}

uintArr MeshBuffer::getFaceArray( size_t &n )
{
    n = m_numFaces;
    return m_faceIndices;
}

ucharArr MeshBuffer::getFaceColorArray( size_t &n )
{
    n = m_numFaceColors;
    return m_faceColors;
}


uintArr MeshBuffer::getFaceTextureIndexArray( size_t &n )
{
    n = m_numFaceTextureIndices;
    return m_faceTextureIndices;
}


floatArr MeshBuffer::getVertexTextureCoordinateArray( size_t &n )
{
    n = m_numVertexTextureCoordinates;
    return m_vertexTextureCoordinates;
}


coord3fArr MeshBuffer::getIndexedVertexArray( size_t &n )
{

    n = m_numVertices;
    return *((coord3fArr*) &m_vertices);

}

coord3fArr MeshBuffer::getIndexedVertexNormalArray( size_t &n )
{

    n = m_numVertexNormals;
    coord3fArr p = *((coord3fArr*) &m_vertexNormals);
    return p;

}


idx1fArr MeshBuffer::getIndexedVertexConfidenceArray( size_t &n )
{

    n = m_numVertexConfidences;
    idx1fArr p = *((idx1fArr*) &m_vertexConfidence);
    return p;

}


idx1fArr MeshBuffer::getIndexedVertexIntensityArray( size_t &n )
{

    n = m_numVertexIntensities;
    idx1fArr p = *((idx1fArr*) &m_vertexIntensity);
    return p;

}


color3bArr MeshBuffer::getIndexedVertexColorArray( size_t &n )
{

    n = m_numVertexColors;
    color3bArr p = *((color3bArr*) &m_vertexColors);
    return p;

}


idx3uArr MeshBuffer::getIndexedFaceArray( size_t &n )
{

    n = m_numFaces;
    idx3uArr p = *((idx3uArr*) &m_faceIndices);
    return p;

}


color3bArr MeshBuffer::getIndexedFaceColorArray( size_t &n )
{

    n = m_numFaceColors;
    color3bArr p = *((color3bArr*) &m_faceColors);
    return p;

}


idx1uArr MeshBuffer::getIndexedFaceTextureIndexArray( size_t &n )
{

    n = m_numFaceTextureIndices;
    idx1uArr p = *((idx1uArr*) &m_faceTextureIndices);
    return p;

}


void MeshBuffer::setVertexArray( floatArr array, size_t n )
{

    m_vertices   = array;
    m_numVertices = n;

}


void MeshBuffer::setVertexArray( std::vector<float>& array )
{

    m_vertices = floatArr( new float[array.size()] );
    for ( int i(0); i < array.size(); i++ ) {
        m_vertices[i] = array[i];
    }
    m_numVertices = array.size() / 3;

}


void MeshBuffer::setVertexNormalArray( floatArr array, size_t n )
{

    m_vertexNormals    = array;
    m_numVertexNormals = n;

}


void MeshBuffer::setVertexNormalArray( std::vector<float>& array )
{

    m_vertexNormals = floatArr( new float[array.size()] );

    for ( size_t i(0); i < array.size(); i++ )
    {
        m_vertexNormals[i] = array[i];
    }
    m_numVertexNormals = array.size() / 3;

}


void MeshBuffer::setFaceArray( uintArr array, size_t n )
{

    m_faceIndices  = array;
    m_numFaces      = n;

}


void MeshBuffer::setFaceArray( std::vector<unsigned int>& array )
{
    m_faceIndices = uintArr( new unsigned int[ array.size() ] );
    for ( size_t i(0); i < array.size(); i++ )
    {
        m_faceIndices[i] = array[i];
    }
    m_numFaces    = array.size() / 3;

}


void MeshBuffer::setFaceTextureIndexArray( std::vector<unsigned int>& array )
{

    m_faceTextureIndices = uintArr( new unsigned int[array.size()] );
    for ( size_t i(0); i < array.size(); i++ )
    {
        m_faceTextureIndices[i] = array[i];
    }
    m_numFaceTextureIndices = array.size() / 3;

}


void MeshBuffer::setFaceTextureIndexArray( uintArr array, size_t n )
{

    m_faceTextureIndices = array;
    m_numFaceTextureIndices = n;

}


void MeshBuffer::setVertexColorArray( ucharArr array, size_t n )
{

    m_vertexColors    = array;
    m_numVertexColors = n;

}


void MeshBuffer::setVertexColorArray( std::vector<uint8_t>& array )
{

    m_vertexColors = ucharArr( new uchar[array.size()] );
    for ( int i(0); i < array.size(); i++ )
    {
        m_vertexColors[i] = array[i];
    }
    m_numVertexColors = array.size() / 3;

}

void MeshBuffer::setVertexConfidenceArray( floatArr array, size_t n )
{

    m_vertexConfidence     = array;
    m_numVertexConfidences = n;

}


void MeshBuffer::setVertexConfidenceArray( std::vector<float>& array )
{

    m_vertexConfidence = floatArr( new float[array.size()] );
    for ( int i(0); i < array.size(); i++ )
    {
        m_vertexConfidence[i] = array[i];
    }
    m_numVertexConfidences = array.size();

}


void MeshBuffer::setVertexTextureCoordinateArray( std::vector<float>& array )
{

    m_vertexTextureCoordinates = floatArr( new float[array.size()] );
    for ( int i(0); i < array.size(); i++ )
    {
        m_vertexTextureCoordinates[i] = array[i];
    }
    m_numVertexTextureCoordinates = array.size() / 3;

}


void MeshBuffer::setIndexedVertexTextureCoordinateArray( coord3fArr arr, size_t size )
{

    m_vertexTextureCoordinates = *((floatArr*) &arr);
    m_numVertexTextureCoordinates = size;

}


void MeshBuffer::setIndexedFaceArray( idx3uArr arr, size_t size )
{

    m_faceIndices = *((uintArr*) &arr);
    m_numFaces    = size;

}


void MeshBuffer::setIndexedFaceColorArray( color3bArr arr, size_t size )
{

    m_faceColors    = *((ucharArr*) &arr);
    m_numFaceColors = size;

}


void MeshBuffer::setVertexTextureCoordinateArray( floatArr array, size_t n )
{

    m_vertexTextureCoordinates = array;
    m_numVertexTextureCoordinates = n;

}


coord3fArr MeshBuffer::getIndexedVertexTextureCoordinateArray( size_t &n )
{

    n = m_numVertexTextureCoordinates;
    return *((coord3fArr*) &m_vertexTextureCoordinates);

}


void MeshBuffer::setVertexIntensityArray( floatArr array, size_t n )
{

    m_vertexIntensity      = array;
    m_numVertexIntensities = n;

}


void MeshBuffer::setVertexIntensityArray( std::vector<float>& array )
{
    
    m_vertexIntensity = floatArr( new float[array.size()] );

    for ( int i(0); i < array.size(); i++ ) 
    {
        m_vertexIntensity[i] = array[i];
    }
    m_numVertexIntensities = array.size();

}


void MeshBuffer::setFaceColorArray( std::vector<uchar> &array )
{

    m_faceColors = ucharArr( new uchar[array.size()] );

    for ( int i(0); i < array.size(); i++ ) 
    {
        m_faceColors[i] = array[i];
    }
    m_numFaceColors = array.size() / 3;
}


void MeshBuffer::setFaceColorArray( ucharArr array, size_t n )
{

    m_faceColors = array;
    m_numFaceColors = n;

}


void MeshBuffer::setIndexedVertexArray( coord3fArr arr, size_t count )
{

    m_vertices    = *((floatArr*) &arr);
    m_numVertices = count;

}


void MeshBuffer::setIndexedVertexNormalArray( coord3fArr arr, size_t count )
{

    m_vertexNormals    = *((floatArr*) &arr);
    m_numVertexNormals = count;

}


void MeshBuffer::setIndexedVertexColorArray( color3bArr arr, size_t count )
{

    m_vertexColors    = *((ucharArr*) &arr);
    m_numVertexColors = count;

}


void MeshBuffer::freeBuffer()
{

    m_faceIndices.reset();
    m_faceColors.reset();
    m_faceTextureIndices.reset();
    m_vertexColors.reset();
    m_vertexConfidence.reset();
    m_vertexIntensity.reset();
    m_vertexNormals.reset();
    m_vertexTextureCoordinates.reset();
    m_vertices.reset();

    m_numVertices = m_numVertexColors = m_numVertexIntensities
        = m_numVertexConfidences = m_numVertexNormals = m_numFaces = 0;

}


} /* namespace lssr */
