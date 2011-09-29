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
#include "Message.hpp"


namespace lssr
{


PLYIO::PLYIO()
    : MeshLoader(), PointLoader()
{
}


PLYIO::~PLYIO()
{
}


void PLYIO::save( string filename )
{

    save( filename, PLY_LITTLE_ENDIAN );

}


void PLYIO::save( string filename, e_ply_storage_mode mode, 
        std::vector<string> obj_info, std::vector<string> comment )
{

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
        ply_add_element( oply, "vertex", m_numVertex );

        /* Add vertex properties: x, y, z, (r, g, b) */
        ply_add_scalar_property( oply, "x", PLY_FLOAT );
        ply_add_scalar_property( oply, "y", PLY_FLOAT );
        ply_add_scalar_property( oply, "z", PLY_FLOAT );

        /* Add color information if there is any. */
        if ( m_vertexColors )
        {
            if ( m_numVertexColors != m_numVertex )
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
            if ( m_numVertexIntensity != m_numVertex )
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
            if ( m_numVertexConfidence != m_numVertex )
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
            if ( m_numVertexNormals != m_numVertex )
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
        if ( m_numFace )
        {
            ply_add_element( oply, "face", m_numFace );
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
        if ( m_pointConfidence )
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

    for ( uint32_t i = 0; i < m_numVertex; i++ )
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
        for ( uint32_t i = 0; i < m_numFace; i++ )
        {
            ply_write( oply, 3.0 ); /* Indices per face. */
            ply_write( oply, (double) m_faceIndices[ i * 3     ] );
            ply_write( oply, (double) m_faceIndices[ i * 3 + 1 ] );
            ply_write( oply, (double) m_faceIndices[ i * 3 + 2 ] );
        }
    }

    for ( uint32_t i = 0; i < m_numPoints; i++ )
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
            ply_write( oply, m_pointConfidence[ i ] );
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


void PLYIO::read( string filename )
{

    read( filename, true );

}


void PLYIO::read( string filename, bool readColor, bool readConfidence,
        bool readIntensity, bool readNormals, bool readFaces )
{

    freeBuffer();

    /* Start reading new PLY */
    p_ply ply = ply_open( filename.c_str(), NULL, 0, NULL );

    if ( !ply )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not open »%s«.\n", filename.c_str() );
        return;
    }
    if ( !ply_read_header( ply ) )
    {
        g_msg.print( MSG_TYPE_ERROR, "Could not read header.\n" );
        return;
    }
    g_msg.print( MSG_TYPE_MESSGAE, "Loading »%s«…\n", filename.c_str() );

    /* Check if there are vertices and get the amount of vertices. */
    char buf[256] = "";
    const char * name = buf;
    long int n;
    p_ply_element elem  = NULL;
    while ( ( elem = ply_get_next_element( ply, elem ) ) )
    {
        ply_get_element_info( elem, &name, &n );
        if ( !strcmp( name, "vertex" ) )
        {
            m_numVertex = n;
            p_ply_property prop = NULL;
            while ( ( prop = ply_get_next_property( elem, prop ) ) )
            {
                ply_get_property_info( prop, &name, NULL, NULL, NULL );
                if ( !strcmp( name, "red" ) && readColor )
                {
                    /* We have color information */
                    m_numVertexColors = n;
                }
                else if ( !strcmp( name, "confidence" ) && readConfidence )
                {
                    /* We have confidence information */
                    m_numVertexConfidence = n;
                }
                else if ( !strcmp( name, "intensity" ) && readIntensity )
                {
                    /* We have intensity information */
                    m_numVertexIntensity = n;
                }
                else if ( !strcmp( name, "nx" ) && readNormals )
                {
                    /* We have normals */
                    m_numVertexNormals = n;
                }
            }
        }
        else if ( !strcmp( name, "point" ) )
        {
            m_numPoints = n;
            p_ply_property prop = NULL;
            while ( ( prop = ply_get_next_property( elem, prop ) ) )
            {
                ply_get_property_info( prop, &name, NULL, NULL, NULL );
                if ( !strcmp( name, "red" ) && readColor )
                {
                    /* We have color information */
                    m_numPointColors = n;
                }
                else if ( !strcmp( name, "confidence" ) && readConfidence )
                {
                    /* We have confidence information */
                    m_numPointConfidence = n;
                }
                else if ( !strcmp( name, "intensity" ) && readIntensity )
                {
                    /* We have intensity information */
                    m_numPointIntensities = n;
                }
                else if ( !strcmp( name, "nx" ) && readNormals )
                {
                    /* We have normals */
                    m_numPointNormals = n;
                }
            }
        }
        else if ( !strcmp( name, "face" ) && readFaces )
        {
            m_numFace = n;
        }
    }
    if ( !( m_numVertex || m_numPoints ) )
    {
        g_msg.print( MSG_TYPE_WARNING, "Neither vertices nor points in ply.\n" );
        return;
    }

    /* Allocate memory. */
    if ( m_numVertex )
    {
        m_vertices = ( float * ) malloc( m_numVertex * 3 * sizeof(float) );
    }
    if ( m_numVertexColors )
    {
        m_vertexColors = ( uint8_t * ) malloc( m_numVertex * 3 * sizeof(uint8_t) );
    }
    if ( m_numVertexConfidence )
    {
        m_vertexConfidence = ( float * ) malloc( m_numVertex * sizeof(float) );
    }
    if ( m_numVertexIntensity )
    {
        m_vertexIntensity = ( float * ) malloc( m_numVertex * sizeof(float) );
    }
    if ( m_numVertexNormals )
    {
        m_vertexNormals = ( float * ) malloc( m_numVertex * 3 * sizeof(float) );
    }
    if ( m_numFace )
    {
        m_faceIndices = ( unsigned int * ) malloc( m_numFace * 3 * sizeof(unsigned int) );
    }
    if ( m_numPoints )
    {
        m_points = ( float * ) malloc( m_numPoints * 3 * sizeof(float) );
    }
    if ( m_numPointColors )
    {
        m_pointColors = ( uint8_t * ) malloc( m_numPoints * 3 * sizeof(uint8_t) );
    }
    if ( m_numPointConfidence )
    {
        m_pointConfidence = ( float * ) malloc( m_numPoints * sizeof(float) );
    }
    if ( m_numPointIntensities )
    {
        m_pointIntensities = ( float * ) malloc( m_numPoints * sizeof(float) );
    }
    if ( m_numPointNormals )
    {
        m_pointNormals = ( float * ) malloc( m_numPoints * 3 * sizeof(float) );
    }


    float        * vertex            = m_vertices;
    uint8_t      * vertex_color      = m_vertexColors;
    float        * vertex_confidence = m_vertexConfidence;
    float        * vertex_intensity  = m_vertexIntensity;
    float        * vertex_normal     = m_vertexNormals;
    unsigned int * face              = m_faceIndices;
    float        * point             = m_points;
    uint8_t      * point_color       = m_pointColors;
    float        * point_confidence  = m_pointConfidence;
    float        * point_intensity   = m_pointIntensities;
    float        * point_normal      = m_pointNormals;


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
        ply_set_read_cb( ply, "vertex", "nx", readVertexCb, &vertex_intensity, 0 );
        ply_set_read_cb( ply, "vertex", "ny", readVertexCb, &vertex_intensity, 0 );
        ply_set_read_cb( ply, "vertex", "nz", readVertexCb, &vertex_intensity, 1 );
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
        freeBuffer();
    }

    /* Check if we got only vertices and neither points nor faces. If that is
     * the case then use the vertices as points. */
    if ( m_vertices && !m_points && !m_faceIndices )
    {
        g_msg.print( MSG_TYPE_HINT, "PLY contains neither faces nor points. "
                "Assuming that vertices are ment to be points.\n" );
        m_points              = m_vertices;
        m_pointColors         = m_vertexColors;
        m_pointConfidence     = m_vertexConfidence;
        m_pointIntensities    = m_vertexIntensity;
        m_pointNormals        = m_vertexNormals;
        m_numPoints           = m_numVertex;
        m_numPointColors      = m_numVertexColors;
        m_numPointConfidence  = m_numVertexConfidence;
        m_numPointIntensities = m_numVertexIntensity;
        m_numPointNormals     = m_numVertexNormals;
        m_numVertex           = 0;
        m_numVertexColors     = 0;
        m_numVertexConfidence = 0;
        m_numVertexIntensity  = 0;
        m_numVertexNormals    = 0;
        m_vertices            = NULL;
        m_vertexColors        = NULL;
        m_vertexConfidence    = NULL;
        m_vertexIntensity     = NULL;
        m_vertexNormals       = NULL;
    }

    ply_close( ply );

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

    float ** face;
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
