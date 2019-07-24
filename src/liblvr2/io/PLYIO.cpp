/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /**
 * @file       PLYIO.hpp
 * @brief      I/O support for PLY files (implementation).
 * @details    I/O support for PLY files: Reading and writing meshes and
 *             pointclouds, including color information, confidence, intensity
 *             and normals.
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @author     Thomas Wiemann
 * @version   110929
 * @date       Created:       2011-09-16 17:28:28
 * @date       Last modified: 2011-09-29 14:23:36
 */


#include "lvr2/io/PLYIO.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <cstring>
#include <ctime>
#include <sstream>
#include <fstream>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

namespace lvr2
{


void PLYIO::save( string filename )
{
    if ( !m_model )
    {
        std::cerr << timestamp << "No data to save." << std::endl;
        return;
    }

    /* Handle options. */
    e_ply_storage_mode mode( PLY_LITTLE_ENDIAN );

    // Local buffer shortcuts
    floatArr m_vertices;
    floatArr m_vertexConfidence;
    floatArr m_vertexIntensity;
    floatArr m_vertexNormals;
    floatArr m_points;
    floatArr m_pointConfidences;
    floatArr m_pointIntensities;
    floatArr m_pointNormals;

    size_t dummy                    = 0;
    size_t w_point_color            = 0;
    size_t w_vertex_color            = 0;
    size_t m_numVertices              = 0;
    size_t m_numVertexColors          = 0;
    size_t m_numVertexConfidences     = 0;
    size_t m_numVertexIntensities     = 0;
    size_t m_numVertexNormals         = 0;

    size_t m_numPoints                = 0;
    size_t m_numPointColors           = 0;
    size_t m_numPointConfidence       = 0;
    size_t m_numPointIntensities      = 0;
    size_t m_numPointNormals          = 0;
    size_t m_numFaces                 = 0;

    ucharArr m_vertexColors;
    ucharArr m_pointColors;
    uintArr  m_faceIndices;

    // Get buffers
    if ( m_model->m_pointCloud )
    {
        PointBufferPtr pc( m_model->m_pointCloud );

        m_numPoints = pc->numPoints();
        m_numPointNormals = m_numPoints;
        m_numPointColors = m_numPoints;

        m_points                = pc->getPointArray();
        m_pointConfidences      = pc->getFloatArray("confidences", m_numPointConfidence, dummy);
        m_pointColors           = pc->getColorArray(w_point_color);
        m_pointIntensities      = pc->getFloatArray("intensities", m_numPointIntensities, dummy);
        m_pointNormals          = pc->getNormalArray();
    }

    if ( m_model->m_mesh )
    {
        MeshBufferPtr mesh( m_model->m_mesh );
        m_numVertices = mesh->numVertices();
        m_numFaces = mesh->numFaces();
        m_numVertexColors = m_numVertices;
        m_numVertexNormals = m_numVertices;

        m_vertices         = mesh->getVertices();
        m_vertexColors     = mesh->getVertexColors(w_vertex_color);
        m_vertexConfidence = mesh->getFloatArray("vertex_confidences", m_numVertexConfidences, dummy);
        m_vertexIntensity  = mesh->getFloatArray("vertex_intensities", m_numVertexIntensities, dummy);
        m_vertexNormals    = mesh->getVertexNormals();
        m_faceIndices      = mesh->getFaceIndices();
    }


    p_ply oply = ply_create( filename.c_str(), mode, NULL, 0, NULL );
    if ( !oply )
    {
        std::cerr << timestamp << "Could not create »" << filename << "«" << std::endl;
        return;
    }

    /* Check if we have vertex information. */
    if ( !( m_vertices || m_points ) )
    {
        std::cout << timestamp << "Neither vertices nor points to write." << std::endl;
        if ( !ply_close( oply ) )
        {
            std::cerr << timestamp << "Could not close file." << std::endl;
        }
        return;
    }

    /* First: Write Header information according to data. */

    bool vertex_color      = false;
    bool vertex_intensity  = false;
    bool vertex_confidence = false;
    bool vertex_normal     = false;
    bool point_color       = false;
    bool point_intensity   = false;
    bool point_confidence  = false;
    bool point_normal      = false;


    /* Add vertex element. */
    if ( m_vertices )
    {
        ply_add_element( oply, "vertex", m_numVertices );

        /* Add vertex properties: x, y, z, (r, g, b) */
        ply_add_scalar_property( oply, "x", PLY_FLOAT );
        ply_add_scalar_property( oply, "y", PLY_FLOAT );
        ply_add_scalar_property( oply, "z", PLY_FLOAT );

        /* Add color information if there is any. */
        if ( m_vertexColors )
        {
            if ( m_numVertexColors != m_numVertices )
            {
                std::cerr << timestamp << "Amount of vertices and color information is"
                    << " not equal. Color information won't be written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "red",   PLY_UCHAR );
                ply_add_scalar_property( oply, "green", PLY_UCHAR );
                ply_add_scalar_property( oply, "blue",  PLY_UCHAR );
                vertex_color = true;
            }
        }

        /* Add intensity. */
        if ( m_vertexIntensity )
        {
            if ( m_numVertexIntensities != m_numVertices )
            {
                std::cout << timestamp << "Amount of vertices and intensity"
                    << " information is not equal. Intensity information won't be"
                    << " written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "intensity",  PLY_FLOAT );
                vertex_intensity = true;
            }
        }

        /* Add confidence. */
        if ( m_vertexConfidence )
        {
            if ( m_numVertexConfidences != m_numVertices )
            {
                std::cout << timestamp << "Amount of vertices and confidence"
                    << " information is not equal. Confidence information won't be"
                    << " written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "confidence",  PLY_FLOAT );
                vertex_confidence = true;
            }
        }

        /* Add normals if there are any. */
        if ( m_vertexNormals )
        {
            if ( m_numVertexNormals != m_numVertices )
            {
                std::cout << timestamp << "Amount of vertices and normals"
                    << " does not match. Normals won't be written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "nx", PLY_FLOAT );
                ply_add_scalar_property( oply, "ny", PLY_FLOAT );
                ply_add_scalar_property( oply, "nz", PLY_FLOAT );
                vertex_normal = true;
            }
        }

        /* Add faces. */
        if ( m_faceIndices )
        {
            ply_add_element( oply, "face", m_numFaces );
            ply_add_list_property( oply, "vertex_indices", PLY_UCHAR, PLY_INT );
        }
    }

    /* Add point element */
    if ( m_points )
    {
        ply_add_element( oply, "point", m_numPoints );

        /* Add point properties: x, y, z, (r, g, b) */
        ply_add_scalar_property( oply, "x", PLY_FLOAT );
        ply_add_scalar_property( oply, "y", PLY_FLOAT );
        ply_add_scalar_property( oply, "z", PLY_FLOAT );

        /* Add color information if there is any. */
        if ( m_pointColors )
        {
            if ( m_numPointColors != m_numPoints )
            {
                std::cout << timestamp << "Amount of points and color information is"
                    << " not equal. Color information won't be written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "red",   PLY_UCHAR );
                ply_add_scalar_property( oply, "green", PLY_UCHAR );
                ply_add_scalar_property( oply, "blue",  PLY_UCHAR );
                point_color = true;
            }
        }

        /* Add intensity. */
        if ( m_pointIntensities )
        {
            if ( m_numPointIntensities != m_numPoints )
            {
                std::cout << timestamp << "Amount of points and intensity"
                    << " information is not equal. Intensity information won't be"
                    << " written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "intensity",  PLY_FLOAT );
                point_intensity = true;
            }
        }

        /* Add confidence. */
        if ( m_pointConfidences )
        {
            if ( m_numPointConfidence != m_numPoints )
            {
                std::cout << timestamp << "Amount of point and confidence"
                    << " information is not equal. Confidence information won't be"
                    << " written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "confidence",  PLY_FLOAT );
                point_confidence = true;
            }
        }

        /* Add normals if there are any. */
        if ( m_pointNormals )
        {
            if ( m_numPointNormals != m_numPoints )
            {
                std::cout << timestamp << "Amount of point and normals does"
                    << " not match. Normals won't be written." << std::endl;
            }
            else
            {
                ply_add_scalar_property( oply, "nx", PLY_FLOAT );
                ply_add_scalar_property( oply, "ny", PLY_FLOAT );
                ply_add_scalar_property( oply, "nz", PLY_FLOAT );
                point_normal = true;
            }
        }
    }

    /* Write header to file. */
    if ( !ply_write_header( oply ) )
    {
        std::cerr << timestamp << "Could not write header." << std::endl;
        return;
    }

    /* Second: Write data. */

    for (size_t i = 0; i < m_numVertices; i++ )
    {
        ply_write( oply, (double) m_vertices[ i * 3     ] ); /* x */
        ply_write( oply, (double) m_vertices[ i * 3 + 1 ] ); /* y */
        ply_write( oply, (double) m_vertices[ i * 3 + 2 ] ); /* z */
        if ( vertex_color )
        {
            ply_write( oply, m_vertexColors[ i * w_vertex_color     ] ); /* red */
            ply_write( oply, m_vertexColors[ i * w_vertex_color + 1 ] ); /* green */
            ply_write( oply, m_vertexColors[ i * w_vertex_color + 2 ] ); /* blue */
        }
        if ( vertex_intensity )
        {
            ply_write( oply, m_vertexIntensity[ i ] );
        }
        if ( vertex_confidence )
        {
            ply_write( oply, m_vertexConfidence[ i ] );
        }
        if ( vertex_normal )
        {
            ply_write( oply, (double) m_vertexNormals[ i * 3     ] ); /* nx */
            ply_write( oply, (double) m_vertexNormals[ i * 3 + 1 ] ); /* ny */
            ply_write( oply, (double) m_vertexNormals[ i * 3 + 2 ] ); /* nz */
        }
    }

    /* Write faces (Only if we also have vertices). */
    if ( m_vertices )
    {
        for ( size_t i = 0; i < m_numFaces; i++ )
        {
            ply_write( oply, 3.0 ); /* Indices per face. */
            ply_write( oply, (double) m_faceIndices[ i * 3     ] );
            ply_write( oply, (double) m_faceIndices[ i * 3 + 1 ] );
            ply_write( oply, (double) m_faceIndices[ i * 3 + 2 ] );
        }
    }

    for ( size_t i = 0; i < m_numPoints; i++ )
    {
        ply_write( oply, (double) m_points[ i * 3     ] ); /* x */
        ply_write( oply, (double) m_points[ i * 3 + 1 ] ); /* y */
        ply_write( oply, (double) m_points[ i * 3 + 2 ] ); /* z */
        if ( point_color )
        {
            ply_write( oply, m_pointColors[ i * w_point_color     ] ); /* red */
            ply_write( oply, m_pointColors[ i * w_point_color + 1 ] ); /* green */
            ply_write( oply, m_pointColors[ i * w_point_color + 2 ] ); /* blue */
        }
        if ( point_intensity )
        {
            ply_write( oply, m_pointIntensities[ i ] );
        }
        if ( point_confidence )
        {
            ply_write( oply, m_pointConfidences[ i ] );
        }
        if ( point_normal )
        {
            ply_write( oply, (double) m_pointNormals[ i * 3     ] ); /* nx */
            ply_write( oply, (double) m_pointNormals[ i * 3 + 1 ] ); /* ny */
            ply_write( oply, (double) m_pointNormals[ i * 3 + 2 ] ); /* nz */
        }
    }

    if ( !ply_close( oply ) )
    {
       std::cerr << timestamp << "Could not close file." << std::endl;
    }

}


ModelPtr PLYIO::read( string filename )
{
   return read( filename, true );
}

/**
 * swaps n elements from index i1 in arr with n elements from index i2 in arr
 */
template <typename T>
void swap(T*& arr, size_t i1, size_t i2, size_t n)
{
    std::swap_ranges(arr + i1, arr + i1 + n, arr + i2);
}

ModelPtr PLYIO::read( string filename, bool readColor, bool readConfidence,
        bool readIntensity, bool readNormals, bool readFaces, bool readPanoramaCoords )
{

    /* Start reading new PLY */
    p_ply ply = ply_open( filename.c_str(), NULL, 0, NULL );

    if ( !ply )
    {
        std::cerr << timestamp << "Could not open »" << filename << "«."
           << std::endl;
        return ModelPtr();
    }
    if ( !ply_read_header( ply ) )
    {
        std::cerr << timestamp << "Could not read header." << std::endl;
        return ModelPtr();
    }
    //std::cout << timestamp << "Loading »" << filename << "«." << std::endl;

    /* Check if there are vertices and get the amount of vertices. */
    char buf[256] = "";
    const char * name = buf;
    long int n;
    p_ply_element elem  = NULL;

    // Buffer count variables
    size_t numVertices              = 0;
    size_t numVertexColors          = 0;
    size_t numVertexConfidences     = 0;
    size_t numVertexIntensities     = 0;
    size_t numVertexNormals         = 0;
    size_t numVertexPanoramaCoords  = 0;

    size_t numPoints                = 0;
    size_t numPointColors           = 0;
    size_t numPointConfidence       = 0;
    size_t numPointIntensities      = 0;
    size_t numPointNormals          = 0;
    size_t numPointPanoramaCoords   = 0;
    size_t numPointSpectralChannels = 0;
    size_t numFaces                 = 0;

    size_t n_channels               = 0; // Number of spectral channels


    while ( ( elem = ply_get_next_element( ply, elem ) ) )
    {
        ply_get_element_info( elem, &name, &n );
        if ( !strcmp( name, "vertex" ) )
        {
            numVertices = n;
            p_ply_property prop = NULL;
            while ( ( prop = ply_get_next_property( elem, prop ) ) )
            {
                ply_get_property_info( prop, &name, NULL, NULL, NULL );
                if ( !strcmp( name, "red" ) && readColor )
                {
                    /* We have color information */
                    numVertexColors = n;
                }
                else if ( !strcmp( name, "confidence" ) && readConfidence )
                {
                    /* We have confidence information */
                    numVertexConfidences = n;
                }
                else if ( !strcmp( name, "intensity" ) && readIntensity )
                {
                    /* We have intensity information */
                    numVertexIntensities = n;
                }
                else if ( !strcmp( name, "nx" ) && readNormals )
                {
                    /* We have normals */
                    numVertexNormals = n;
                }
                else if ( !strcmp( name, "x_coords" ) && readPanoramaCoords )
                {
                    /* We have panorama coordinates */
                    numVertexPanoramaCoords = n;
                }
            }
        }
        else if ( !strcmp( name, "point" ) )
        {
            numPoints = n;
            p_ply_property prop = NULL;
            while ( ( prop = ply_get_next_property( elem, prop ) ) )
            {
                ply_get_property_info( prop, &name, NULL, NULL, NULL );
                if ( !strcmp( name, "red" ) && readColor )
                {
                    /* We have color information */
                    numPointColors = n;
                }
                else if ( !strcmp( name, "confidence" ) && readConfidence )
                {
                    /* We have confidence information */
                    numPointConfidence = n;
                }
                else if ( !strcmp( name, "intensity" ) && readIntensity )
                {
                    /* We have intensity information */
                    numPointIntensities = n;
                }
                else if ( !strcmp( name, "nx" ) && readNormals )
                {
                    /* We have normals */
                    numPointNormals = n;
                }
                else if ( !strcmp( name, "x_coords" ) && readPanoramaCoords )
                {
                    /* We have panorama coordinates */
                    numPointPanoramaCoords = n;
                }
            }
        }
        else if ( !strcmp( name, "face" ) && readFaces )
        {
            numFaces = n;
        }
    }

    if ( !( numVertices || numPoints ) )
    {
        std::cout << timestamp << "Neither vertices nor points in ply."
            << std::endl;
        return ModelPtr();
    }

    // Buffers
    floatArr vertices;
    floatArr vertexConfidence;
    floatArr vertexIntensity;
    floatArr vertexNormals;
    floatArr points;
    floatArr pointConfidences;
    floatArr pointIntensities;
    floatArr pointNormals;
    ucharArr pointSpectralChannels;

    ucharArr vertexColors;
    ucharArr pointColors;

    shortArr vertexPanoramaCoords;
    shortArr pointPanoramaCoords;

    uintArr  faceIndices;


    /* Allocate memory. */
    if ( numVertices )
    {
        vertices = floatArr( new float[ numVertices * 3 ] );
    }
    if ( numVertexColors )
    {
        vertexColors = ucharArr( new unsigned char[ numVertices * 3 ] );
    }
    if ( numVertexConfidences )
    {
        vertexConfidence = floatArr( new float[ numVertices ] );
    }
    if ( numVertexIntensities )
    {
        vertexIntensity = floatArr( new float[ numVertices ] );
    }
    if ( numVertexNormals )
    {
        vertexNormals = floatArr( new float[ numVertices * 3 ] );
    }
    if ( numVertexPanoramaCoords )
    {
        vertexPanoramaCoords = shortArr( new short[ numVertices * 2 ] );
    }
    if ( numFaces )
    {
        faceIndices = indexArray( new unsigned int[ numFaces * 3 ] );
    }
    if ( numPoints )
    {
        points = floatArr( new float[ numPoints * 3 ] );
    }
    if ( numPointColors )
    {
        pointColors = ucharArr( new unsigned char[ numPoints * 3 ] );
    }
    if ( numPointConfidence )
    {
        pointConfidences = floatArr( new float[numPoints] );
    }
    if ( numPointIntensities )
    {
        pointIntensities = floatArr( new float[numPoints] );
    }
    if ( numPointNormals )
    {
        pointNormals = floatArr( new float[ numPoints * 3 ] );
    }
    if ( numPointPanoramaCoords )
    {
        pointPanoramaCoords = shortArr( new short[ numPoints * 2 ] );
    }


    float*          vertex                   = vertices.get();
    uint8_t*        vertex_color             = vertexColors.get();
    float*          vertex_confidence        = vertexConfidence.get();
    float*          vertex_intensity         = vertexIntensity.get();
    float*          vertex_normal            = vertexNormals.get();
    short*          vertex_panorama_coords   = vertexPanoramaCoords.get();
    unsigned int*   face                     = faceIndices.get();
    float*          point                    = points.get();
    uint8_t*        point_color              = pointColors.get();
    float*          point_confidence         = pointConfidences.get();
    float*          point_intensity          = pointIntensities.get();
    float*          point_normal             = pointNormals.get();
    short*          point_panorama_coords    = pointPanoramaCoords.get();


    /* Set callbacks. */
    if ( vertex )
    {
        ply_set_read_cb( ply, "vertex", "x", readVertexCb, &vertex, 0 );
        ply_set_read_cb( ply, "vertex", "y", readVertexCb, &vertex, 0 );
        ply_set_read_cb( ply, "vertex", "z", readVertexCb, &vertex, 1 );
    }
    if ( vertex_color )
    {
        ply_set_read_cb( ply, "vertex", "red",   readColorCb,  &vertex_color,  0 );
        ply_set_read_cb( ply, "vertex", "green", readColorCb,  &vertex_color,  0 );
        ply_set_read_cb( ply, "vertex", "blue",  readColorCb,  &vertex_color,  1 );
    }
    if ( vertex_confidence )
    {
        ply_set_read_cb( ply, "vertex", "confidence", readVertexCb, &vertex_confidence, 1 );
    }
    if ( vertex_intensity )
    {
        ply_set_read_cb( ply, "vertex", "intensity", readVertexCb, &vertex_intensity, 1 );
    }
    if ( vertex_normal )
    {
        ply_set_read_cb( ply, "vertex", "nx", readVertexCb, &vertex_normal, 0 );
        ply_set_read_cb( ply, "vertex", "ny", readVertexCb, &vertex_normal, 0 );
        ply_set_read_cb( ply, "vertex", "nz", readVertexCb, &vertex_normal, 1 );
    }
    if ( vertex_panorama_coords )
    {
        ply_set_read_cb( ply, "vertex", "x_coords", readPanoramaCoordCB, &vertex_panorama_coords, 0 );
        ply_set_read_cb( ply, "vertex", "y_coords", readPanoramaCoordCB, &vertex_panorama_coords, 1 );
    }

    if ( face )
    {
        ply_set_read_cb( ply, "face", "vertex_indices", readFaceCb, &face, 0 );
        ply_set_read_cb( ply, "face", "vertex_index", readFaceCb, &face, 0 );
    }

    if ( point )
    {
        ply_set_read_cb( ply, "point", "x", readVertexCb, &point, 0 );
        ply_set_read_cb( ply, "point", "y", readVertexCb, &point, 0 );
        ply_set_read_cb( ply, "point", "z", readVertexCb, &point, 1 );
    }
    if ( point_color )
    {
        ply_set_read_cb( ply, "point", "red",   readColorCb,  &point_color,  0 );
        ply_set_read_cb( ply, "point", "green", readColorCb,  &point_color,  0 );
        ply_set_read_cb( ply, "point", "blue",  readColorCb,  &point_color,  1 );
    }
    if ( point_confidence )
    {
        ply_set_read_cb( ply, "point", "confidence", readVertexCb, &point_confidence, 1 );
    }
    if ( point_intensity )
    {
        ply_set_read_cb( ply, "point", "intensity", readVertexCb, &point_intensity, 1 );
    }
    if ( point_normal )
    {
        ply_set_read_cb( ply, "point", "nx", readVertexCb, &point_normal, 0 );
        ply_set_read_cb( ply, "point", "ny", readVertexCb, &point_normal, 0 );
        ply_set_read_cb( ply, "point", "nz", readVertexCb, &point_normal, 1 );
    }
    if ( point_panorama_coords )
    {
        ply_set_read_cb( ply, "point", "x_coords", readPanoramaCoordCB, &point_panorama_coords, 0 );
        ply_set_read_cb( ply, "point", "y_coords", readPanoramaCoordCB, &point_panorama_coords, 1 );
    }

    /* Read ply file. */
    if ( !ply_read( ply ) )
    {
        std::cerr << timestamp << "Could not read »" << filename << "«."
            << std::endl;
    }

    /* Check if we got only vertices and neither points nor faces. If that is
     * the case then use the vertices as points. */
    if ( vertices && !points && !faceIndices )
    {
        std::cout << timestamp << "PLY contains neither faces nor points. "
            << "Assuming that vertices are meant to be points." << std::endl;
        points                  = vertices;
        pointColors             = vertexColors;
        pointConfidences        = vertexConfidence;
        pointIntensities        = vertexIntensity;
        pointNormals            = vertexNormals;
        pointPanoramaCoords     = vertexPanoramaCoords;
        point                   = points.get();
        point_color             = pointColors.get();
        point_confidence        = pointConfidences.get();
        point_intensity         = pointIntensities.get();
        point_normal            = pointNormals.get();
        point_panorama_coords   = pointPanoramaCoords.get();
        numPoints               = numVertices;
        numPointColors          = numVertexColors;
        numPointConfidence      = numVertexConfidences;
        numPointIntensities     = numVertexIntensities;
        numPointNormals         = numVertexNormals;
        numPointPanoramaCoords  = numVertexPanoramaCoords;
        numVertices             = 0;
        numVertexColors         = 0;
        numVertexConfidences    = 0;
        numVertexIntensities    = 0;
        numVertexNormals        = 0;
        numVertexPanoramaCoords = 0;
        vertices.reset();
        vertexColors.reset();
        vertexConfidence.reset();
        vertexIntensity.reset();
        vertexNormals.reset();
        vertexPanoramaCoords.reset();
    }

    ply_close( ply );

    // read Panorama Images if we have annotated data
    if (numPointPanoramaCoords)
    {
        // move all the Points that don't have spectral information to the end
        for (int i = 0; i < numPointPanoramaCoords; i++)
        {
            if (point_panorama_coords[2 * i] == -1)
            {
                // swap with last element
                numPointPanoramaCoords--;

                const int n = numPointPanoramaCoords;
                swap(point_panorama_coords, 2 * i, 2 * numPointPanoramaCoords, 2);
                swap(point, 3 * i, 3 * numPointPanoramaCoords, 3);
                if (numPointColors)      swap(point_color,      3*i, 3*numPointPanoramaCoords, 3);
                if (numPointConfidence)  swap(point_confidence,   i,   numPointPanoramaCoords, 1);
                if (numPointIntensities) swap(point_intensity,    i,   numPointPanoramaCoords, 1);
                if (numPointNormals)     swap(point_normal,     3*i, 3*numPointPanoramaCoords, 3);

                i--;
            }
        }

        std::cout << timestamp << numPoints << "Found " << (numPoints - numPointPanoramaCoords) << " without spectral data. Reodering..." << std::endl;


        size_t pos_underscore = filename.find_last_of("_");
        size_t pos_extension = filename.find_last_of(".");
        string scanNr = filename.substr(pos_underscore + 1, pos_extension - pos_underscore - 1);

        string channelDirName = string("panorama_channels_") + scanNr;

        boost::filesystem::path dir(filename);
        dir = dir.parent_path() / "panoramas_fixed" / channelDirName;

        if (!boost::filesystem::exists(dir / "channel0.png"))
        {
            std::cerr << timestamp << "Annotated Data given, but " + dir.string() + " does not contain channel files" << std::endl;
        }
        else
        {
            std::cout << timestamp << "Found Annotated Data. Loading spectral channel images from: " << dir.string() << "/" << std::endl;
            std::cout << timestamp << "This may take a while depending on data size" << std::endl;

            std::vector<cv::Mat> imgs;
            std::vector<unsigned char*> pixels;

            cv::VideoCapture images(dir.string() + "/channel%d.png");
            cv::Mat img, imgFlipped;
            while (images.read(img))
            {
                imgs.push_back(cv::Mat());
                cv::flip(img, imgFlipped, 0); // TODO: FIXME: Data is currently stored mirrored and offset
                cv::cvtColor(imgFlipped, imgs.back(), cv::COLOR_RGB2GRAY);
                pixels.push_back(imgs.back().data);
            }

            n_channels = imgs.size();
            int width = imgs[0].cols;
            int height = imgs[0].rows;

            numPointSpectralChannels = numPointPanoramaCoords;
            pointSpectralChannels = ucharArr(new unsigned char[numPointPanoramaCoords * n_channels]);

            unsigned char* point_spectral_channels = pointSpectralChannels.get();

            std::cout << timestamp << "Finished loading " << n_channels << " channel images" << std::endl;

            #pragma omp parallel for
            for (int i = 0; i < numPointPanoramaCoords; i++)
            {
                int pc_index = 2 * i; // x_coords, y_coords
                short x = point_panorama_coords[pc_index];
                short y = point_panorama_coords[pc_index + 1];
                x = (x + width / 2) % width; // TODO: FIXME: Data is currently stored mirrored and offset

                int panoramaPosition = y * width + x;
                unsigned char* pixel = point_spectral_channels + n_channels * i;

                for (int channel = 0; channel < n_channels; channel++)
                {
                    pixel[channel] = pixels[channel][panoramaPosition];
                }
            }
            std::cout << timestamp << "Finished extracting channel information" << std::endl;
        }
    }


    // Save buffers in model
    PointBufferPtr pc;
    MeshBufferPtr mesh;
    if(points)
    {
        pc = PointBufferPtr( new PointBuffer );
        pc->setPointArray(points, numPoints);

        if (pointColors)
        {
            pc->setColorArray(pointColors, numPointColors);
        }

        if (pointIntensities)
        {
            pc->addFloatChannel(pointIntensities, "intensities", numPointIntensities, 1);
        }

        if (pointConfidences)
        {
            pc->addFloatChannel(pointConfidences, "confidences", numPointConfidence, 1);
        }

        if (pointNormals)
        {
            pc->setNormalArray(pointNormals, numPointNormals);
        }

        // only add spectral data if we really have some...
        if (pointSpectralChannels)
        {
            pc->addUCharChannel(pointSpectralChannels, "spectral_channels", numPointSpectralChannels, n_channels);

            // there is no way to read min-, maxchannel from ply file => assume default 400-1000nm
            pc->addIntAtomic(400, "spectral_wavelength_min");
            pc->addIntAtomic(400 + 4 * n_channels, "spectral_wavelength_max");
            pc->addIntAtomic(n_channels, "num_spectral_channels");
        }
    }

    if(vertices)
    {
        mesh = MeshBufferPtr( new MeshBuffer );
        mesh->setVertices(vertices, numVertices );

        if (faceIndices)
        {
            mesh->setFaceIndices(faceIndices, numFaces);
        }

        if (vertexNormals)
        {
            mesh->setVertexNormals(vertexNormals);
        }

        if (vertexColors)
        {
            mesh->setVertexColors(vertexColors);
        }

        if (vertexIntensity)
        {
            mesh->addFloatChannel(vertexIntensity, "vertex_intensities", numVertexIntensities, 1);
        }

        if (vertexConfidence)
        {
            mesh->addFloatChannel(vertexConfidence, "vertex_confidences",  numVertexConfidences, 1);
        }
    }

    ModelPtr m( new Model( mesh, pc ) );
    m_model = m;
    return m;

}


int PLYIO::readVertexCb( p_ply_argument argument )
{
    float ** ptr;
    ply_get_argument_user_data( argument, (void **) &ptr, NULL );
    **ptr = ply_get_argument_value( argument );
    (*ptr)++;
    return 1;

}


int PLYIO::readColorCb( p_ply_argument argument )
{

    uint8_t ** color;
    ply_get_argument_user_data( argument, (void **) &color, NULL );
    **color = ply_get_argument_value( argument );
    (*color)++;
    return 1;

}


int PLYIO::readFaceCb( p_ply_argument argument )
{

    unsigned int ** face;
    long int length, value_index;
    ply_get_argument_user_data( argument, (void **) &face, NULL );
    ply_get_argument_property( argument, NULL, &length, &value_index );
    if ( value_index < 0 )
    {
        /* We got info about amount of face vertices. */
        if ( ply_get_argument_value( argument ) == 3 )
        {
            return 1;
        }
        std::cerr << timestamp << "Mesh is not a triangle mesh." << std::endl;
        return 0;
    }
    **face = ply_get_argument_value( argument );
    (*face)++;

    return 1;

}

int PLYIO::readPanoramaCoordCB( p_ply_argument argument )
{

    short ** ptr;
    ply_get_argument_user_data( argument, (void **) &ptr, NULL );
    **ptr = ply_get_argument_value( argument );
    (*ptr)++;
    return 1;

}

} // namespace lvr2
