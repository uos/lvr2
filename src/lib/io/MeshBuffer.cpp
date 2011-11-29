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
    m_vertices( NULL ),
    m_vertexColors( NULL ),
    m_vertexIntensity( NULL ),
    m_vertexNormals( NULL ),
    m_vertexTextureCoordinates( NULL ),
    m_indexedVertices( NULL ),
    m_indexedVertexColors( NULL ),
    m_indexedVertexIntensity( NULL ),
    m_indexedVertexNormals( NULL ),
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
    m_vertexConfidence.reset();
}


float* MeshBuffer::getVertexArray( size_t &n )
{

    n = m_numVertices;
    return m_vertices;

}

float* MeshBuffer::getVertexNormalArray( size_t &n )
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


float* MeshBuffer::getVertexIntensityArray( size_t &n )
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


float** MeshBuffer::getIndexedVertexArray( size_t &n )
{

    n = m_numVertices;

    /* Return NULL if we have no vertices. */
    if ( !m_vertices )
    {
        n = 0;
        return NULL;
    }


    /* Generate indexed vertex array in not already done. */
    if ( !m_indexedVertices )
    {
        m_indexedVertices = (float**) malloc( m_numVertices * sizeof(float**) );
        if( !m_indexedVertices )
        {
            std::cerr << "Could not Allocate memory. Lets just gracefully die now my dear :)" << std::endl;
            n = 0;
            return NULL;
        }
        for ( size_t i = 0; i < m_numVertices; i++ )
        {
            m_indexedVertices[i] = m_vertices + ( i * 3 );
        }
    }

    /* Return indexed vertex array */
    return m_indexedVertices;

}

float** MeshBuffer::getIndexedVertexNormalArray( size_t &n )
{

    n = m_numVertexNormals;

    /* Return NULL if we have no normals. */
    if ( !m_vertexNormals )
    {
        n = 0;
        return NULL;
    }


    /* Generate indexed normal array in not already done. */
    if ( !m_indexedVertexNormals )
    {
        m_indexedVertexNormals = (float**) malloc( m_numVertexNormals * sizeof(float**) );
        if( !m_indexedVertexNormals )
        {
            std::cerr << "Could not Allocate memory. Lets just gracefully die now my dear :)" << std::endl;
            n = 0;
            return NULL;
        }
        for ( size_t i = 0; i < m_numVertexNormals; i++ )
        {
            m_indexedVertexNormals[i] = m_vertexNormals + ( i * 3 );
        }
    }

    /* Return indexed normals array */
    return m_indexedVertexNormals;


}


idxFloatArr MeshBuffer::getIndexedVertexConfidenceArray( size_t &n )
{

    n = m_numVertexConfidences;
    idxFloatArr p = *((idxFloatArr*) &m_vertexConfidence);
    return p;

}


float** MeshBuffer::getIndexedVertexIntensityArray( size_t &n )
{

    n = m_numVertexIntensities;

    /* Return NULL if we have no intensity information. */
    if ( !m_vertexIntensity )
    {
        return NULL;
    }

    /* Generate indexed intensity array in not already done. */
    if ( !m_indexedVertexIntensity )
    {
        m_indexedVertexIntensity = (float**) malloc( m_numVertexIntensities * sizeof(float**) );
        for ( size_t i = 0; i < m_numVertexIntensities; i++ )
        {
            m_indexedVertexIntensity[i] = m_vertexIntensity + i;
        }
    }

    /* Return indexed intensity array */
    return m_indexedVertexIntensity;

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

void MeshBuffer::setVertexArray( float* array, size_t n )
{

    m_vertices   = array;
    m_numVertices = n;

}


void MeshBuffer::setVertexArray( std::vector<float>& array )
{

    if(m_vertices) delete m_vertices;
    m_vertices = new float[array.size()];
    std::copy(array.begin(), array.end(), m_vertices);
    m_numVertices = array.size() / 3;

}

void MeshBuffer::setVertexNormalArray( float* array, size_t n )
{

    m_vertexNormals    = array;
    m_numVertexNormals = n;

}

void MeshBuffer::setVertexNormalArray( std::vector<float>& array )
{

    if(m_vertexNormals)
    {
        delete m_vertexNormals;
    }

    m_vertexNormals = new float[array.size()];

    std::copy(array.begin(), array.end(), m_vertexNormals);
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


void MeshBuffer::setVertexIntensityArray( float* array, size_t n )
{

    m_vertexIntensity     = array;
    m_numVertexIntensities = n;

}


void MeshBuffer::setVertexIntensityArray( std::vector<float>& array )
{
    
    if( m_vertexIntensity )
    {
        delete m_vertexIntensity;
    }

    m_vertexIntensity = new float[array.size()];

    std::copy(array.begin(), array.end(), m_vertexIntensity);
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


void MeshBuffer::setIndexedVertexArray( float** arr, size_t count )
{

    m_vertices = (float*) malloc( count * 3 * sizeof(float) );
    for ( size_t i = 0; i < count; i++ )
    {
        m_vertices[ i * 3     ] = arr[i][0];
        m_vertices[ i * 3 + 1 ] = arr[i][1];
        m_vertices[ i * 3 + 2 ] = arr[i][2];
    }

}

void MeshBuffer::setIndexedVertexNormalArray( float** arr, size_t count )
{

    m_vertexNormals = (float*) malloc( count * 3 * sizeof(float) );
    for ( size_t i = 0; i < count; i++ )
    {
        m_vertexNormals[ i * 3     ] = arr[i][0];
        m_vertexNormals[ i * 3 + 1 ] = arr[i][1];
        m_vertexNormals[ i * 3 + 2 ] = arr[i][2];
    }

}

void MeshBuffer::freeBuffer()
{

    m_vertexConfidence.reset();
    m_vertices = m_vertexIntensity = m_vertexNormals = NULL;
    m_vertexColors = NULL;
    m_faceIndices = NULL;
    m_numVertices= m_numVertexColors = m_numVertexIntensities
        = m_numVertexConfidences = m_numVertexNormals = m_numFaces = 0;

}

} /* namespace lssr */
