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
 * @author     Thomas Wiemann
 * @version   110929
 * @date       Created:       2011-09-16 17:28:28
 * @date       Last modified: 2011-09-29 14:23:36
 */


#include <lvr2/io/PLYIO.hpp>
#include <lvr2/io/Timestamp.hpp>

#include <cstring>
#include <ctime>
#include <sstream>
#include <fstream>



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
    floatArr m_vertexNormals;
    ucharArr m_pointColors;

    floatArr m_points;
    floatArr m_pointNormals;

    size_t m_numPoints                = 0;
    size_t m_numVertices              = 0;
    size_t m_numFaces                 = 0;

    unsigned vertexColorWidth;
    unsigned pointColorWidth;

    ucharArr    m_faceColors;
    ucharArr    m_vertexColors;
    indexArray  m_faceIndices;

    // Get buffers
    if ( m_model->m_pointCloud )
    {
        PointBuffer2Ptr pc( m_model->m_pointCloud );
        m_numPoints = pc->numPoints();

        m_points                = pc->getPointArray();
        m_pointColors           = pc->getColorArray(pointColorWidth);
        m_pointNormals          = pc->getNormalArray();
    }

    if ( m_model->m_mesh )
    {


        MeshBuffer2Ptr mesh( m_model->m_mesh );
        m_numFaces = mesh->numFaces();
        m_numVertices = mesh->numVertices();

        m_vertices         = mesh->getVertices();
        m_vertexColors     = mesh->getVertexColors(vertexColorWidth);
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
    bool vertex_normal     = false;
    bool point_color       = false;
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
            ply_add_scalar_property( oply, "red",   PLY_UCHAR );
            ply_add_scalar_property( oply, "green", PLY_UCHAR );
            ply_add_scalar_property( oply, "blue",  PLY_UCHAR );
            if(vertexColorWidth == 4)
            {
                ply_add_scalar_property( oply, "alpha",  PLY_UCHAR );
            }
            vertex_color = true;
        }

        /* Add normals if there are any. */
        if ( m_vertexNormals )
        {
            ply_add_scalar_property( oply, "nx", PLY_FLOAT );
            ply_add_scalar_property( oply, "ny", PLY_FLOAT );
            ply_add_scalar_property( oply, "nz", PLY_FLOAT );
            vertex_normal = true;
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
            ply_add_scalar_property( oply, "red",   PLY_UCHAR );
            ply_add_scalar_property( oply, "green", PLY_UCHAR );
            ply_add_scalar_property( oply, "blue",  PLY_UCHAR );
            if(pointColorWidth == 4)
            {
                ply_add_scalar_property( oply, "alpha",  PLY_UCHAR );
            }
            point_color = true;
        }

        /* Add normals if there are any. */
        if ( m_pointNormals )
        {
            ply_add_scalar_property( oply, "nx", PLY_FLOAT );
            ply_add_scalar_property( oply, "ny", PLY_FLOAT );
            ply_add_scalar_property( oply, "nz", PLY_FLOAT );
            point_normal = true;
        }
    }

    /* Write header to file. */
    if ( !ply_write_header( oply ) )
    {
        std::cerr << timestamp << "Could not write header." << std::endl;
        return;
    }

    /* Second: Write data. */
    bool vertex_color_alpha = (vertexColorWidth == 4);
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
            if(vertex_color_alpha)
            {
                ply_write( oply, m_vertexColors[ i * 3 + 3 ] ); // alpha
            }
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

    bool point_color_alpha = (pointColorWidth == 4);
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
            if(point_color_alpha)
            {
                ply_write( oply, m_pointColors[ i * 3 + 3 ] ); /* alpha */
            }
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


ModelPtr PLYIO::read( string filename, bool readColor, bool readConfidence,
        bool readIntensity, bool readNormals, bool readFaces )
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
    size_t numVertexNormals         = 0;

    size_t numPoints                = 0;
    size_t numPointColors           = 0;
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
        std::cout << timestamp << "Neither vertices nor points in ply."
            << std::endl;
        return ModelPtr();
    }

    // Buffers
    floatArr vertices;
    floatArr vertexNormals;
    floatArr points;
    floatArr pointNormals;

    ucharArr pointColors;
    ucharArr vertexColors;

    uintArr faceIndices;


    /* Allocate memory. */
    if ( numVertices )
    {
        vertices = floatArr( new float[ numVertices * 3 ] );
    }
    if ( numVertexColors )
    {
        vertexColors = ucharArr( new unsigned char[ numVertices * 3 ] );
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
        pointColors = ucharArr( new unsigned char[ numPoints * 3 ] );
    }
    if ( numPointNormals )
    {
        pointNormals = floatArr( new float[ 3 * numPoints ] );
    }


    float*        vertex            = vertices.get();
    uint8_t*      vertex_color      = vertexColors.get();
    float*        vertex_normal     = vertexNormals.get();
    unsigned int* face              = faceIndices.get();
    float*        point             = points.get();
    uint8_t*      point_color       = pointColors.get();
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
    if ( point_normal )
    {
        ply_set_read_cb( ply, "point", "nx", readVertexCb, &point_normal, 0 );
        ply_set_read_cb( ply, "point", "ny", readVertexCb, &point_normal, 0 );
        ply_set_read_cb( ply, "point", "nz", readVertexCb, &point_normal, 1 );
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
        points               = vertices;
        pointColors          = vertexColors;
        pointNormals         = vertexNormals;
        numPoints            = numVertices;
        numPointColors       = numVertexColors;
        numPointNormals      = numVertexNormals;
        numVertices          = 0;
        numVertexColors      = 0;
        numVertexNormals     = 0;
        vertices.reset();
        vertexColors.reset();
        vertexNormals.reset();
    }

    ply_close( ply );


    // Save buffers in model
    PointBuffer2Ptr pc;
    MeshBuffer2Ptr mesh;
    if(points)
    {
        pc = PointBuffer2Ptr(new PointBuffer2);
        pc->setPointArray(points, numPoints);
        pc->setColorArray(pointColors, numPointColors, 3);
        pc->setNormalArray(pointNormals, numPointNormals);
    }

    if(vertices)
    {
        mesh = MeshBuffer2Ptr(new MeshBuffer2);
        mesh->setVertices(vertices, numVertices);
        mesh->setVertexColors(vertexColors, 3);
        mesh->setVertexNormals(vertexNormals);
        mesh->setFaceIndices(faceIndices, numFaces);
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
    //cout << ply_get_argument_value( argument ) << endl;
    **face = ply_get_argument_value( argument );
    (*face)++;
    return 1;

}


} // namespace lvr
