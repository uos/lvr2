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
#include "BufferedMesh.hpp"

namespace lssr
{

BufferedMesh::BufferedMesh() : 
    m_vertices( NULL ),
    m_vertexColors( NULL ),
    m_vertexConfidence( NULL ),
    m_vertexIntensity( NULL ),
    m_vertexNormals( NULL ),
    m_indexedVertices( NULL ),
    m_indexedVertexColors( NULL ),
    m_indexedVertexConfidence( NULL ),
    m_indexedVertexIntensity( NULL ),
    m_indexedVertexNormals( NULL ),
    m_faceIndices( NULL ),
    m_numVertices( 0 ),
    m_numVertexNormals( 0 ),
    m_numVertexColors( 0 ),
    m_numVertexConfidences( 0 ),
    m_numVertexIntensities( 0 ),
    m_numFaces( 0 )
{
}


float* BufferedMesh::getVertexArray( size_t &n )
{

    n = m_numVertices;
    return m_vertices;

}

float* BufferedMesh::getVertexNormalArray( size_t &n )
{

    n = m_numVertexNormals;
    return m_vertexNormals;

}


uint8_t* BufferedMesh::getVertexColorArray( size_t &n )
{

    n = m_numVertexColors;
    return m_vertexColors;

}


float* BufferedMesh::getVertexConfidenceArray( size_t &n )
{

    n = m_numVertexConfidences;
    return m_vertexConfidence;

}


float* BufferedMesh::getVertexIntensityArray( size_t &n )
{

    n = m_numVertexIntensities;
    return m_vertexIntensity;

}

unsigned int* BufferedMesh::getFaceArray( size_t &n )
{

    n = m_numFaces;
    return m_faceIndices;

}


float** BufferedMesh::getIndexedVertexArray( size_t &n )
{

    n = m_numVertices;

    /* Return NULL if we have no vertices. */
    if ( !m_vertices )
    {
        return NULL;
    }


    /* Generate indexed vertex array in not already done. */
    if ( !m_indexedVertices )
    {
        m_indexedVertices = (float**) malloc( m_numVertices * sizeof(float**) );
        for ( size_t i = 0; i < m_numVertices; i++ )
        {
            m_indexedVertices[i] = m_vertices + ( i * 3 );
        }
    }

    /* Return indexed vertex array */
    return m_indexedVertices;

}

float** BufferedMesh::getIndexedVertexNormalArray( size_t &n )
{

    n = m_numVertexNormals;

    /* Return NULL if we have no normals. */
    if ( !m_vertexNormals )
    {
        return NULL;
    }


    /* Generate indexed normal array in not already done. */
    if ( !m_indexedVertexNormals )
    {
        m_indexedVertexNormals = (float**) 
            malloc( m_numVertexNormals * sizeof(float**) );
        for ( size_t i = 0; i < m_numVertexNormals; i++ )
        {
            m_indexedVertexNormals[i] = m_vertexNormals + ( i * 3 );
        }
    }

    /* Return indexed normals array */
    return m_indexedVertexNormals;


}


float** BufferedMesh::getIndexedVertexConfidenceArray( size_t &n )
{

    n = m_numVertexConfidences;

    /* Return NULL if we have no confidence information. */
    if ( !m_vertexConfidence )
    {
        return NULL;
    }


    /* Generate indexed confidence array in not already done. */
    if ( !m_indexedVertexConfidence )
    {
        m_indexedVertexConfidence = (float**) 
            malloc( m_numVertexConfidences * sizeof(float**) );
        for ( size_t i = 0; i < m_numVertexConfidences; i++ )
        {
            m_indexedVertexConfidence[i] = m_vertexConfidence + i;
        }
    }

    /* Return indexed confidence array */
    return m_indexedVertexConfidence;

}


float** BufferedMesh::getIndexedVertexIntensityArray( size_t &n )
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
        m_indexedVertexIntensity = (float**) 
            malloc( m_numVertexIntensities * sizeof(float**) );
        for ( size_t i = 0; i < m_numVertexIntensities; i++ )
        {
            m_indexedVertexIntensity[i] = m_vertexIntensity + i;
        }
    }

    /* Return indexed intensity array */
    return m_indexedVertexIntensity;

}


uint8_t** BufferedMesh::getIndexedVertexColorArray( size_t &n )
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


void BufferedMesh::setVertexArray( float* array, size_t n )
{

    m_vertices   = array;
    m_numVertices = n;

}

void BufferedMesh::setVertexNormalArray( float* array, size_t n )
{

    m_vertexNormals    = array;
    m_numVertexNormals = n;

}

void BufferedMesh::setFaceArray( unsigned int* array, size_t n )
{

    m_faceIndices  = array;
    m_numFaces      = n;

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

void BufferedMesh::setVertexColorArray( uint8_t* array, size_t n )
{

    m_vertexColors     = array;
    m_numVertexColors = n;

}


void BufferedMesh::setVertexConfidenceArray( float* array, size_t n )
{

    m_vertexConfidence     = array;
    m_numVertexConfidences = n;

}


void BufferedMesh::setVertexIntensityArray( float* array, size_t n )
{

    m_vertexIntensity     = array;
    m_numVertexIntensities = n;

}


void BufferedMesh::setIndexedVertexArray( float** arr, size_t count )
{

    m_vertices = (float*) malloc( count * 3 * sizeof(float) );
    for ( size_t i = 0; i < count; i++ )
    {
        m_vertices[ i * 3     ] = arr[i][0];
        m_vertices[ i * 3 + 1 ] = arr[i][1];
        m_vertices[ i * 3 + 2 ] = arr[i][2];
    }

}


void BufferedMesh::setIndexedVertexNormalArray( float** arr, size_t count )
{

    m_vertexNormals = (float*) malloc( count * 3 * sizeof(float) );
    for ( size_t i = 0; i < count; i++ )
    {
        m_vertexNormals[ i * 3     ] = arr[i][0];
        m_vertexNormals[ i * 3 + 1 ] = arr[i][1];
        m_vertexNormals[ i * 3 + 2 ] = arr[i][2];
    }

}


void BufferedMesh::freeBuffer()
{

    m_vertices = m_vertexConfidence = m_vertexIntensity = m_vertexNormals = NULL;
    m_vertexColors = NULL;
    m_faceIndices = NULL;
    m_numVertices= m_numVertexColors = m_numVertexIntensities
        = m_numVertexConfidences = m_numVertexNormals = m_numFaces = 0;

}

} /* namespace lssr */
