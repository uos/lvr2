#include "PLYIO.hpp"

#include <cstring>
#include <sstream>
#include <cassert>

//using std::stringstream;

//#include <boost/algorithm/string.hpp>
//#include <boost/progress.hpp>


//using boost::algorithm::to_lower;
//using boost::algorithm::is_equal;

namespace lssr
{


PLYIO::~PLYIO() { }


void PLYIO::save( string filename ) {

	 save( filename, PLY_ASCII );

}


void PLYIO::save( string filename, e_ply_storage_mode mode, 
		vector<string> obj_info, vector<string> comment ) {

#warning Code for saving pointclouds still missing. »lkiesow«

	p_ply oply = ply_create( filename.c_str(), mode, NULL, 0, NULL );
	if ( !oply ) {
		fprintf( stderr, "ERROR: Could not create »%s«\n", filename.c_str() );
		return;
	}

	/* Add object infos and comments */
	vector<string>::iterator it; 
	for ( it = obj_info.begin(); it < obj_info.end(); it++ ) {
		if ( !ply_add_obj_info( oply, it->c_str() ) ) {
			fprintf( stderr, "ERROR: Could not add object info.\n" );
		}
	}
	for ( it = comment.begin(); it < comment.end(); it++ ) {
		if ( !ply_add_comment( oply, it->c_str() ) ) {
			fprintf( stderr, "ERROR: Could not add comment.\n" );
		}
	}

	/* Check if we have vertex information. */
	if ( !m_vertices ) {
		fprintf( stderr, "WARNING: No vertices to write.\n" );
		if ( !ply_close( oply ) ) {
			fprintf( stderr, "ERROR: Could not close file.\n" );
		}
		return;
	}

	/* First: Write Header information according to data. */

	/* Add vertex element. */
	ply_add_element( oply, "vertex", m_num_vertex );

	/* Add vertex properties: x, y, z, (r, g, b) */
	ply_add_scalar_property( oply, "x", PLY_FLOAT );
	ply_add_scalar_property( oply, "y", PLY_FLOAT );
	ply_add_scalar_property( oply, "z", PLY_FLOAT );

	/* Add color information if there is any. */
	bool color = false;
	if ( m_vertex_colors ) {
		if ( m_num_vertex_colors != m_num_vertex ) {
			fprintf( stderr, "WARNING: Amount of vertices and color information is"
					" not equal. Color information won't be written.\n" );
		} else {
			ply_add_scalar_property( oply, "red",   PLY_UCHAR );
			ply_add_scalar_property( oply, "green", PLY_UCHAR );
			ply_add_scalar_property( oply, "blue",  PLY_UCHAR );
			color = true;
		}
	}

	/* Add intensity. */
	bool intensity = false;
	if ( m_vertex_intensity ) {
		if ( m_num_vertex_intensity != m_num_vertex ) {
			fprintf( stderr, "WARNING: Amount of vertices and intensity"
					" information is not equal. Intensity information won't be"
					" written.\n" );
		} else {
			ply_add_scalar_property( oply, "intensity",  PLY_FLOAT );
			intensity = true;
		}
	}

	/* Add confidence. */
	bool confidence = false;
	if ( m_vertex_confidence ) {
		if ( m_num_vertex_confidence != m_num_vertex ) {
			fprintf( stderr, "WARNING: Amount of vertices and confidence"
					" information is not equal. Confidence information won't be"
					" written.\n" );
		} else {
			ply_add_scalar_property( oply, "confidence",  PLY_FLOAT );
			confidence = true;
		}
	}

	/* Add normals if there are any. */
	bool normal = false;
	if ( m_vertex_normals ) {
		if ( m_num_vertex_normals != m_num_vertex ) {
			fprintf( stderr, "WARNING: Amount of vertices and normals"
					" does not match. Normals won't be written.\n" );
		} else {
			ply_add_scalar_property( oply, "nx", PLY_FLOAT );
			ply_add_scalar_property( oply, "ny", PLY_FLOAT );
			ply_add_scalar_property( oply, "nz", PLY_FLOAT );
			normal = true;
		}
	}

	/* Add faces. */
	if ( m_num_face ) {
		ply_add_element( oply, "face", m_num_face );
		ply_add_list_property( oply, "vertex_indices", PLY_UCHAR, PLY_INT );
	}

	/* Write header to file. */
	if ( !ply_write_header( oply ) ) {
		fprintf( stderr, "ERROR: Could not write header.\n" );
		return;
	}

	/* Second: Write data. */

	for ( int i = 0; i < m_num_vertex; i++ ) {
		ply_write( oply, (double) m_vertices[ i * 3     ] ); /* x */
		ply_write( oply, (double) m_vertices[ i * 3 + 1 ] ); /* y */
		ply_write( oply, (double) m_vertices[ i * 3 + 2 ] ); /* z */
		if ( color ) {
			ply_write( oply, m_vertex_colors[ i * 3     ] ); /* red */
			ply_write( oply, m_vertex_colors[ i * 3 + 1 ] ); /* green */
			ply_write( oply, m_vertex_colors[ i * 3 + 2 ] ); /* blue */
		}
		if ( intensity ) {
			ply_write( oply, m_vertex_intensity[ i ] );
		}
		if ( confidence ) {
			ply_write( oply, m_vertex_confidence[ i ] );
		}
		if ( normal ) {
			ply_write( oply, (double) m_vertex_normals[ i * 3     ] ); /* x */
			ply_write( oply, (double) m_vertex_normals[ i * 3 + 1 ] ); /* y */
			ply_write( oply, (double) m_vertex_normals[ i * 3 + 2 ] ); /* z */
		}
	}

	/* Write faces. */
	for ( int i = 0; i < m_num_face; i++ ) {
		ply_write( oply, 3.0 ); /* Indices per face. */
		ply_write( oply, (double) m_face_indices[ i * 3     ] );
		ply_write( oply, (double) m_face_indices[ i * 3 + 1 ] );
		ply_write( oply, (double) m_face_indices[ i * 3 + 2 ] );
	}

	if ( !ply_close( oply ) ) {
		fprintf( stderr, "ERROR: Could not close file.\n" );
	}

}


void PLYIO::read( string filename ) {

	read( filename, true );

}


void PLYIO::read( string filename, bool readColor, bool readConfidence,
		bool readIntensity, bool readNormals, bool readFaces ) {

	/* Cleanup members. */
	freeBuffer();

	/* Start reading new PLY */
	p_ply ply = ply_open( filename.c_str(), NULL, 0, NULL );

	if ( !ply ) {
		fprintf( stderr, "error: Could not open »%s«.\n", filename.c_str() );
		return;
	}
	if ( !ply_read_header( ply ) ) {
		fprintf( stderr, "error: Could not read header.\n" );
		return;
	}
	printf( "Loading »%s«…\n", filename.c_str() );

	/* Check if there are vertices and get the amount of vertices. */
	char buf[256] = "";
	const char * name = buf;
	long int n;
	p_ply_element elem  = NULL;
	while ( ( elem = ply_get_next_element( ply, elem ) ) ) {
		ply_get_element_info( elem, &name, &n );
		if ( !strcmp( name, "vertex" ) ) {
			m_num_vertex = n;
			p_ply_property prop = NULL;
			while ( ( prop = ply_get_next_property( elem, prop ) ) ) {
				ply_get_property_info( prop, &name, NULL, NULL, NULL );
				if ( !strcmp( name, "red" ) && readColor ) {
					/* We have color information */
					m_num_vertex_colors = n;
				} else if ( !strcmp( name, "confidence" ) && readConfidence ) {
					/* We have confidence information */
					m_num_vertex_confidence = n;
				} else if ( !strcmp( name, "intensity" ) && readIntensity ) {
					/* We have intensity information */
					m_num_vertex_intensity = n;
				} else if ( !strcmp( name, "nx" ) && readNormals ) {
					/* We have normals */
					m_num_vertex_normals = n;
				}
			}
		} else if ( !strcmp( name, "point" ) ) {
			m_num_points = n;
			p_ply_property prop = NULL;
			while ( ( prop = ply_get_next_property( elem, prop ) ) ) {
				ply_get_property_info( prop, &name, NULL, NULL, NULL );
				if ( !strcmp( name, "red" ) && readColor ) {
					/* We have color information */
					m_num_point_colors = n;
				} else if ( !strcmp( name, "confidence" ) && readConfidence ) {
					/* We have confidence information */
					m_num_point_confidence = n;
				} else if ( !strcmp( name, "intensity" ) && readIntensity ) {
					/* We have intensity information */
					m_num_point_intensities = n;
				} else if ( !strcmp( name, "nx" ) && readNormals ) {
					/* We have normals */
					m_num_point_normals = n;
				}
			}
		} else if ( !strcmp( name, "face" ) && readFaces ) {
			m_num_face = n;
		}
	}
	if ( !( m_num_vertex || m_num_points ) ) {
		fprintf( stderr, "warning: Neither vertices nor points in ply.\n" );
		return;
	}

	/* Allocate memory. */
	if ( m_num_vertex ) {
		m_vertices = ( float * ) malloc( m_num_vertex * 3 * sizeof(float) );
	}
	if ( m_num_vertex_colors ) {
		m_vertex_colors = ( uint8_t * ) malloc( m_num_vertex * 3 * sizeof(uint8_t) );
	}
	if ( m_num_vertex_confidence ) {
		m_vertex_confidence = ( float * ) malloc( m_num_vertex * sizeof(float) );
	}
	if ( m_num_vertex_intensity ) {
		m_vertex_intensity = ( float * ) malloc( m_num_vertex * sizeof(float) );
	}
	if ( m_num_vertex_normals ) {
		m_vertex_normals = ( float * ) malloc( m_num_vertex * 3 * sizeof(float) );
	}
	if ( m_num_face ) {
		m_face_indices = ( unsigned int * ) malloc( m_num_face * 3 * sizeof(unsigned int) );
	}
	if ( m_num_points ) {
		m_points = ( float * ) malloc( m_num_points * 3 * sizeof(float) );
	}
	if ( m_num_point_colors ) {
		m_point_colors = ( uint8_t * ) malloc( m_num_points * 3 * sizeof(uint8_t) );
	}
	if ( m_num_point_confidence ) {
		m_point_confidence = ( float * ) malloc( m_num_points * sizeof(float) );
	}
	if ( m_num_point_intensities ) {
		m_point_intensities = ( float * ) malloc( m_num_points * sizeof(float) );
	}
	if ( m_num_point_normals ) {
		m_point_normals = ( float * ) malloc( m_num_points * 3 * sizeof(float) );
	}
	
	
	float        * vertex            = m_vertices;
	uint8_t      * vertex_color      = m_vertex_colors;
	float        * vertex_confidence = m_vertex_confidence;
	float        * vertex_intensity  = m_vertex_intensity;
	float        * vertex_normal     = m_vertex_normals;
	unsigned int * face              = m_face_indices;
	float        * point             = m_points;
	uint8_t      * point_color       = m_point_colors;
	float        * point_confidence  = m_point_confidence;
	float        * point_intensity   = m_point_intensities;
	float        * point_normal      = m_point_normals;


	/* Set callbacks. */
	if ( vertex ) {
		ply_set_read_cb( ply, "vertex", "x", readVertexCb, &vertex, 0 );
		ply_set_read_cb( ply, "vertex", "y", readVertexCb, &vertex, 0 );
		ply_set_read_cb( ply, "vertex", "z", readVertexCb, &vertex, 1 );
	}
	if ( vertex_color ) {
		ply_set_read_cb( ply, "vertex", "red",   readColorCb,  &vertex_color,  0 );
		ply_set_read_cb( ply, "vertex", "green", readColorCb,  &vertex_color,  0 );
		ply_set_read_cb( ply, "vertex", "blue",  readColorCb,  &vertex_color,  1 );
	}
	if ( vertex_confidence ) {
		ply_set_read_cb( ply, "vertex", "confidence", readVertexCb, &vertex_confidence, 1 );
	}
	if ( vertex_intensity ) {
		ply_set_read_cb( ply, "vertex", "intensity", readVertexCb, &vertex_intensity, 1 );
	}
	if ( vertex_normal ) {
		ply_set_read_cb( ply, "vertex", "nx", readVertexCb, &vertex_intensity, 0 );
		ply_set_read_cb( ply, "vertex", "ny", readVertexCb, &vertex_intensity, 0 );
		ply_set_read_cb( ply, "vertex", "nz", readVertexCb, &vertex_intensity, 1 );
	}

	if ( face ) {
		ply_set_read_cb( ply, "face", "vertex_indices", readFaceCb, &face, 0 );
		ply_set_read_cb( ply, "face", "vertex_index", readFaceCb, &face, 0 );
	}

	if ( point ) {
		ply_set_read_cb( ply, "point", "x", readVertexCb, &point, 0 );
		ply_set_read_cb( ply, "point", "y", readVertexCb, &point, 0 );
		ply_set_read_cb( ply, "point", "z", readVertexCb, &point, 1 );
	}
	if ( point_color ) {
		ply_set_read_cb( ply, "point", "red",   readColorCb,  &point_color,  0 );
		ply_set_read_cb( ply, "point", "green", readColorCb,  &point_color,  0 );
		ply_set_read_cb( ply, "point", "blue",  readColorCb,  &point_color,  1 );
	}
	if ( point_confidence ) {
		ply_set_read_cb( ply, "point", "confidence", readVertexCb, &point_confidence, 1 );
	}
	if ( point_intensity ) {
		ply_set_read_cb( ply, "point", "intensity", readVertexCb, &point_intensity, 1 );
	}
	if ( point_normal ) {
		ply_set_read_cb( ply, "point", "nx", readVertexCb, &point_intensity, 0 );
		ply_set_read_cb( ply, "point", "ny", readVertexCb, &point_intensity, 0 );
		ply_set_read_cb( ply, "point", "nz", readVertexCb, &point_intensity, 1 );
	}

	/* Read ply file. */
	if ( !ply_read( ply ) ) {
		fprintf( stderr, "error: could not read »%s«.\n", filename.c_str() );
		freeBuffer();
	}
	ply_close( ply );
	
}


int PLYIO::readVertexCb( p_ply_argument argument ) {

	float ** ptr;
	ply_get_argument_user_data( argument, (void **) &ptr, NULL );
	**ptr = ply_get_argument_value( argument );
	(*ptr)++;
	return 1;

}


int PLYIO::readColorCb( p_ply_argument argument ) {

	uint8_t ** color;
	ply_get_argument_user_data( argument, (void **) &color, NULL );
	**color = ply_get_argument_value( argument );
	(*color)++;
	return 1;

}


int PLYIO::readFaceCb( p_ply_argument argument ) {

	float ** face;
	long int length, value_index;
	ply_get_argument_user_data( argument, (void **) &face, NULL );
	ply_get_argument_property( argument, NULL, &length, &value_index );
	if ( value_index < 0 ) {
		/* We got info about amount of face vertices. */
		if ( ply_get_argument_value( argument ) == 3 ) {
			return 1;
		}
		fprintf( stderr, "error: Mesh is not a triangle mesh.\n" );
		return 0;
	}
	**face = ply_get_argument_value( argument );
	(*face)++;

	return 1;

}


} // namespace lssr
