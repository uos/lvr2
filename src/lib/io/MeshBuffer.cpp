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
    m_vertexColors( NULL ),
    m_vertexTextureCoordinates( NULL ),
    m_indexedVertexColors( NULL ),
    m_indexedVertexTextureCoordinates( NULL ),
    m_faceIndices( NULL ),
    m_faceTextureIndices( NULL ),
    m_faceColors (NULL),
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
    m_vertices.reset();
    m_vertexConfidence.reset();
    m_vertexIntensity.reset();
    m_vertexNormals.reset();
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


uint8_t* MeshBuffer::getVertexColorArray( size_t &n )
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

unsigned int* MeshBuffer::getFaceArray( size_t &n )
{
    n = m_numFaces;
    return m_faceIndices;
}

uchar* MeshBuffer::getFaceColorArray( size_t &n )
{
    n = m_numFaceColors;
    return m_faceColors;
}


unsigned int* MeshBuffer::getFaceTextureIndexArray( size_t &n )
{
    n = m_numFaceTextureIndices;
    return m_faceTextureIndices;
}


coordfArr MeshBuffer::getIndexedVertexArray( size_t &n )
{

    n = m_numVertices;
    return *((coordfArr*) &m_vertices);

}

coordfArr MeshBuffer::getIndexedVertexNormalArray( size_t &n )
{

    n = m_numVertexNormals;
    coordfArr p = *((coordfArr*) &m_vertexNormals);
    return p;

}


idxFloatArr MeshBuffer::getIndexedVertexConfidenceArray( size_t &n )
{

    n = m_numVertexConfidences;
    idxFloatArr p = *((idxFloatArr*) &m_vertexConfidence);
    return p;

}


idxFloatArr MeshBuffer::getIndexedVertexIntensityArray( size_t &n )
{

    n = m_numVertexIntensities;
    idxFloatArr p = *((idxFloatArr*) &m_vertexIntensity);
    return p;

}


uint8_t** MeshBuffer::getIndexedVertexColorArray( size_t &n )
{

    n = m_numVertexColors;
    if ( !m_vertexColors )
    {
        return NULL;
    }

    if ( !m_indexedVertexColors )
    {
        m_indexedVertexColors = (uint8_t**) 
            malloc( m_numVertexColors * sizeof(uint8_t**) );
        for ( size_t i = 0; i < m_numVertexColors; i++ )
        {
            m_indexedVertexColors[i] = m_vertexColors + ( i * 3 );
        }
    }
    return m_indexedVertexColors;

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

void MeshBuffer::setFaceArray( unsigned int* array, size_t n )
{

    m_faceIndices  = array;
    m_numFaces      = n;

}

void MeshBuffer::setFaceArray( std::vector<unsigned int>& array )
{
    if(m_faceIndices)
    {
        delete m_faceIndices;
    }
    m_faceIndices = new unsigned int[array.size()];
    std::copy(array.begin(), array.end(), m_faceIndices);
    m_numFaces      = array.size() / 3;

}

void MeshBuffer::setFaceTextureIndexArray( std::vector<unsigned int>& array )
{
    if(m_faceTextureIndices)
    {
        delete m_faceTextureIndices;
    }
    m_faceTextureIndices = new unsigned int[array.size()];
    std::copy(array.begin(), array.end(), m_faceTextureIndices);
    m_numFaceTextureIndices = array.size() / 3;
}

//void MeshLoader::setVertexColorArray( float* array, size_t n )
//{
//
//    m_vertexColors = (uint8_t*) malloc( n * 3 * sizeof(uint8_t) );
//    for ( int i = 0; i < ( 3 * n ); i++ )
//    {
//        m_vertexColors[i] = (uint8_t) ( array[i] * 255 );
//    }
//    m_numVertexColors = n;
//
//}

void MeshBuffer::setVertexColorArray( uint8_t* array, size_t n )
{

    m_vertexColors     = array;
    m_numVertexColors = n;

}

void MeshBuffer::setVertexColorArray( std::vector<uint8_t>& array )
{
    if(m_vertexColors)
    {
        delete m_vertexColors;
    }

    m_vertexColors = new uchar[array.size()];
    std::copy(array.begin(), array.end(), m_vertexColors);
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
    if(m_vertexTextureCoordinates) 
    {
         delete m_vertexTextureCoordinates;
    }

    m_vertexTextureCoordinates = new float[array.size()];

    std::copy(array.begin(), array.end(), m_vertexTextureCoordinates);
    m_numVertexTextureCoordinates = array.size() / 3;
}


float** MeshBuffer::getIndexedVertexTextureCoordinateArray( size_t &n )
{

    n = m_numVertexTextureCoordinates;

  /* Return NULL if we have no vertices. */
    if ( !m_vertexTextureCoordinates )
    {
        n = 0;
        return NULL;
    }


  /* Generate indexed vertex array in not already done. */
    if ( !m_indexedVertexTextureCoordinates )
    {
          m_indexedVertexTextureCoordinates = (float**) malloc( m_numVertexTextureCoordinates * sizeof(float**) );
          if( !m_indexedVertexTextureCoordinates )
          {
              std::cerr << "Could not Allocate memory. Lets just gracefully die now my dear :)" << std::endl;
              n = 0;
              return NULL;
          }
        for ( size_t i = 0; i < m_numVertexTextureCoordinates; i++ )
        {
            m_indexedVertexTextureCoordinates[i] = m_vertexTextureCoordinates + ( i * 3 );
        }
    }

    /* Return indexed vertex array */
    return m_indexedVertexTextureCoordinates;
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
    if( m_faceColors)
    {
        delete m_faceColors;
    }

    m_faceColors = new uchar[array.size()];

    std::copy(array.begin(), array.end(), m_faceColors);
    m_numFaceColors = array.size();
}


void MeshBuffer::setIndexedVertexArray( coordfArr arr, size_t count )
{

    m_numVertices = count;
    m_vertices    = *((floatArr*) &arr);

}


void MeshBuffer::setIndexedVertexNormalArray( coordfArr arr, size_t count )
{

    m_vertexNormals    = *((floatArr*) &m_vertexNormals);
    m_numVertexNormals = count;

}


void MeshBuffer::freeBuffer()
{

    m_vertexConfidence.reset();
    m_vertexIntensity.reset();
    m_vertexNormals.reset();
    m_vertices.reset();
    m_vertexColors = NULL;
    m_faceIndices = NULL;
    m_numVertices= m_numVertexColors = m_numVertexIntensities
        = m_numVertexConfidences = m_numVertexNormals = m_numFaces = 0;

}


} /* namespace lssr */
