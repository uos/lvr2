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
 * @file       PLYIO.hpp
 * @brief      I/O support for PLY files (implementation).
 * @details    I/O support for PLY files: Reading and writing meshes and
 *             pointclouds, including color information, confidence, intensity
 *             and normals.
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version   110929
 * @date       Created:       2011-09-16 17:28:28
 * @date       Last modified: 2011-09-29 14:23:36
 */


#include "PLYIO.hpp"

#include <cstring>
#include <ctime>
#include <sstream>
#include <cassert>
#include <iostream>
#include "Message.hpp"


namespace lssr
{


void PLYIO::save( string filename )
{
    save( filename, PLY_LITTLE_ENDIAN );

}


void PLYIO::save( string filename, e_ply_storage_mode mode, 
        std::vector<string> obj_info, std::vector<string> comment )
{
    assert(m_model);

    // Local buffer shortcuts
    floatArr m_vertices;
    floatArr m_vertexConfidence;
    floatArr m_vertexIntensity;
    floatArr m_vertexNormals;
    floatArr m_points;
    floatArr m_pointConfidences;
    floatArr m_pointIntensities;
    floatArr m_pointNormals;

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

        m_points                = pc->getPointArray(m_numPoints);
        m_pointConfidences      = pc->getPointConfidenceArray(m_numPointConfidence);
        m_pointColors           = pc->getPointColorArray(m_numPointColors);
        m_pointIntensities      = pc->getPointIntensityArray(m_numPointIntensities);
        m_pointNormals          = pc->getPointNormalArray(m_numPointNormals);
    }

    if ( m_model->m_mesh )
    {
        MeshBufferPtr mesh( m_model->m_mesh );

        m_vertices         = mesh->getVertexArray(m_numVertices);
        m_vertexColors     = mesh->getVertexColorArray(m_numVertexColors);
        m_vertexConfidence = mesh->getVertexConfidenceArray(m_numVertexConfidences);
        m_vertexIntensity  = mesh->getVertexIntensityArray(m_numVertexIntensities);
        m_vertexNormals    = mesh->getVertexNormalArray(m_numVertexNormals);
        m_faceIndices      = mesh->getFaceArray(m_numFaces);
    }



    p_ply oply = ply_create( filename.c_str(), mode, NULL, 0, NULL );
    if ( !oply )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not create »%s«\n", filename.c_str() );
        return;
    }

    /* Add object infos and comments */
    std::vector<string>::iterator it; 
    for ( it = obj_info.begin(); it < obj_info.end(); it++ )
    {
        if ( !ply_add_obj_info( oply, it->c_str() ) )
        {
            g_msg.print( MSG_TYPE_ERROR, "Could not add object info.\n" );
        }
    }
    for ( it = comment.begin(); it < comment.end(); it++ )
    {
        if ( !ply_add_comment( oply, it->c_str() ) )
        {
            g_msg.print( MSG_TYPE_ERROR, "Could not add comment.\n" );
        }
    }

    /* Check if we have vertex information. */
    if ( !( m_vertices || m_points ) )
    {
        g_msg.print( MSG_TYPE_WARNING, "Neither vertices nor points to write.\n" );
        if ( !ply_close( oply ) )
        {
            g_msg.print( MSG_TYPE_ERROR, "Could not close file.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of vertices and color information is"
                        " not equal. Color information won't be written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of vertices and intensity"
                        " information is not equal. Intensity information won't be"
                        " written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of vertices and confidence"
                        " information is not equal. Confidence information won't be"
                        " written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of vertices and normals"
                        " does not match. Normals won't be written.\n" );
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
        if ( m_numFaces )
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of points and color information is"
                        " not equal. Color information won't be written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of points and intensity"
                        " information is not equal. Intensity information won't be"
                        " written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of point and confidence"
                        " information is not equal. Confidence information won't be"
                        " written.\n" );
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
                g_msg.print( MSG_TYPE_WARNING, "Amount of point and normals"
                        " does not match. Normals won't be written.\n" );
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
        g_msg.print( MSG_TYPE_ERROR, "Could not write header.\n" );
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
            ply_write( oply, m_vertexColors[ i * 3     ] ); /* red */
            ply_write( oply, m_vertexColors[ i * 3 + 1 ] ); /* green */
            ply_write( oply, m_vertexColors[ i * 3 + 2 ] ); /* blue */
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
            ply_write( oply, m_pointColors[ i * 3     ] ); /* red */
            ply_write( oply, m_pointColors[ i * 3 + 1 ] ); /* green */
            ply_write( oply, m_pointColors[ i * 3 + 2 ] ); /* blue */
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
        g_msg.print( MSG_TYPE_ERROR, "Could not close file.\n" );
    }

}


ModelPtr PLYIO::read( string filename )
{
   return read( filename, true );
}


ModelPtr PLYIO::read( string filename, bool readColor, bool readConfidence,
        bool readIntensity, bool readNormals, bool readFaces )
{

    /* Start reading new PLY */
    p_ply ply = ply_open( filename.c_str(), NULL, 0, NULL );

    if ( !ply )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not open »%s«.\n", filename.c_str() );
        return ModelPtr();
    }
    if ( !ply_read_header( ply ) )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not read header.\n" );
        return ModelPtr();
    }
    g_msg.print( MSG_TYPE_MESSGAE, "Loading »%s«…\n", filename.c_str() );

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

    size_t numPoints                = 0;
    size_t numPointColors           = 0;
    size_t numPointConfidence       = 0;
    size_t numPointIntensities      = 0;
    size_t numPointNormals          = 0;
    size_t numFaces                 = 0;

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
            }
        }
        else if ( !strcmp( name, "face" ) && readFaces )
        {
            numFaces = n;
        }
    }

    if ( !( numVertices || numPoints ) )
    {
        g_msg.print( MSG_TYPE_WARNING, "Neither vertices nor points in ply.\n" );
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

    ucharArr pointColors;
    ucharArr vertexColors;

    uintArr  faceIndices;

    std::cout << numVertices << std::endl;

    /* Allocate memory. */
    if ( numVertices )
    {
        vertices = floatArr( new float[ numVertices * 3 ] );
    }
    if ( numVertexColors )
    {
        vertexColors = ucharArr( new uchar[ numVertices * 3 ] );
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
        vertexNormals = floatArr( new float[ 3 * numVertices ] );
    }
    if ( numFaces )
    {
        faceIndices = uintArr( new unsigned int[ numFaces * 3 ] );
    }
    if ( numPoints )
    {
        points = floatArr( new float[ numPoints * 3 ] );
    }
    if ( numPointColors )
    {
        pointColors = ucharArr( new uchar[ numPoints * 3 ] );
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
        pointNormals = floatArr( new float[ 3 * numPoints ] );
    }


    float*        vertex            = vertices.get();
    uint8_t*      vertex_color      = vertexColors.get();
    float*        vertex_confidence = vertexConfidence.get();
    float*        vertex_intensity  = vertexIntensity.get();
    float*        vertex_normal     = vertexNormals.get();
    unsigned int* face              = faceIndices.get();
    float*        point             = points.get();
    uint8_t*      point_color       = pointColors.get();
    float*        point_confidence  = pointConfidences.get();
    float*        point_intensity   = pointIntensities.get();
    float*        point_normal      = pointNormals.get();


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
        ply_set_read_cb( ply, "point", "nx", readVertexCb, &point_intensity, 0 );
        ply_set_read_cb( ply, "point", "ny", readVertexCb, &point_intensity, 0 );
        ply_set_read_cb( ply, "point", "nz", readVertexCb, &point_intensity, 1 );
    }

    /* Read ply file. */
    if ( !ply_read( ply ) )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not read »%s«.\n", filename.c_str() );
    }

    /* Check if we got only vertices and neither points nor faces. If that is
     * the case then use the vertices as points. */
    if ( vertices && !points && !faceIndices )
    {
        g_msg.print( MSG_TYPE_HINT, "PLY contains neither faces nor points. "
                "Assuming that vertices are ment to be points.\n" );
        points               = vertices;
        pointColors          = vertexColors;
        pointConfidences     = vertexConfidence;
        pointIntensities     = vertexIntensity;
        pointNormals         = vertexNormals;
        numPoints            = numVertices;
        numPointColors       = numVertexColors;
        numPointConfidence   = numVertexConfidences;
        numPointIntensities  = numVertexIntensities;
        numPointNormals      = numVertexNormals;
        numVertices          = 0;
        numVertexColors      = 0;
        numVertexConfidences = 0;
        numVertexIntensities = 0;
        numVertexNormals     = 0;
        vertices.reset();
        vertexColors.reset();
        vertexConfidence.reset();
        vertexIntensity.reset();
        vertexNormals.reset();
    }

    ply_close( ply );


    // Save buffers in model
    PointBufferPtr pc;
    MeshBufferPtr mesh;
    if(points)
    {
        pc = PointBufferPtr( new PointBuffer );
        pc->setPointArray(           points,           numPoints );
        pc->setPointColorArray(      pointColors,      numPointColors );
        pc->setPointIntensityArray(  pointIntensities, numPointIntensities );
        pc->setPointConfidenceArray( pointConfidences, numPointConfidence );
    }

    if(vertices)
    {
        mesh = MeshBufferPtr( new MeshBuffer );
        mesh->setVertexArray(           vertices,         numVertices );
        mesh->setVertexColorArray(      vertexColors,     numVertexColors );
        mesh->setVertexIntensityArray(  vertexIntensity,  numVertexIntensities );
        mesh->setVertexNormalArray(     vertexNormals,    numVertexNormals );
        mesh->setVertexConfidenceArray( vertexConfidence, numVertexConfidences );
        mesh->setFaceArray(             faceIndices,      numFaces );
    }

    ModelPtr m( new Model( mesh, pc ) );
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
        g_msg.print( MSG_TYPE_ERROR, "Mesh is not a triangle mesh.\n" );
        return 0;
    }
    **face = ply_get_argument_value( argument );
    (*face)++;

    return 1;

}


} // namespace lssr
