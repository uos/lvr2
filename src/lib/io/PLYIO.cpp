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

PLYIO::PLYIO() {

	m_vertices       = NULL;
	m_color          = NULL;
	m_intensity      = NULL;
	m_confidence     = NULL;
	m_normals        = NULL;
	m_face_indices   = NULL;
	m_num_face       = 0;
	m_num_normal     = 0;
	m_num_color      = 0;
	m_num_intensity  = 0;
	m_num_vertex     = 0;
	m_num_confidence = 0;

 }


PLYIO::~PLYIO() { }


float * PLYIO::getVertexArray( size_t &n ) {

	n = m_num_vertex;
	return m_vertices;

}

float * PLYIO::getVertexNormalArray( size_t &n ) {
	
	return getNormalArray( n );

};

float * PLYIO::getNormalArray( size_t &n ) {

	n = m_num_normal;
	return m_normals;

}

float * PLYIO::getVertexColorArray( size_t &n ) {

	getColorArray3f( n );

}

unsigned char * PLYIO::getColorArray( size_t &n ) {

	n = m_num_color;
	return m_color;

}

float * PLYIO::getColorArray3f( size_t &n ) {

	n = m_num_color;
	if ( !m_color ) {
		return NULL;
	}
	float * color3fv = (float *) malloc( m_num_color * 3 * sizeof(float) );
	for ( int i = 0; i < ( 3 * m_num_color ); i++ ) {
		color3fv[i] = m_color[i] / 255.0;
	}
	return color3fv;

}


float * PLYIO::getConfidenceArray( size_t &n ) {

	n = m_num_confidence;
	return m_confidence;

}


float * PLYIO::getIntensityArray( size_t &n ) {

	n = m_num_intensity;
	return m_intensity;

}

unsigned int * PLYIO::getIndexArray( size_t &n ) {

	n = m_num_face;
	return m_indices;

}

float ** PLYIO::getIndexedVertexArray( size_t &n ) {

	n = m_num_vertex;
	if ( !m_vertices ) {
		return NULL;
	}

	float ** ivert = (float **) malloc( m_num_vertex * sizeof(float **) );
	for ( int i = 0; i < m_num_vertex; i++ ) {
		ivert[i] = m_vertices + ( i * 3 );
	}
	return ivert;

}

float ** PLYIO::getIndexedNormalArray( size_t &n ) {

	n = m_num_normal;
	if ( !m_normals ) {
		return NULL;
	}

	float ** inorm = (float **) malloc( m_num_normal * sizeof(float **) );
	for ( int i = 0; i < m_num_normal; i++ ) {
		inorm[i] = m_normals + ( i * 3 );
	}
	return inorm;

}


unsigned char ** PLYIO::getIndexedColorArray( size_t &n ) {

	n = m_num_color;
	if ( !m_color ) {
		return NULL;
	}

	unsigned char ** icol = (unsigned char **) malloc( 
			m_num_color * sizeof(unsigned char **) );
	for ( int i = 0; i < m_num_color; i++ ) {
		icol[i] = m_color + ( i * 3 );
	}
	return icol;

}


void PLYIO::setVertexArray( float * array, size_t n ) {

	m_vertices   = array;
	m_num_vertex = n;

}

void PLYIO::setNormalArray( float * array, size_t n ) {

	m_normals    = array;
	m_num_normal = n;

}

void PLYIO::setIndexArray( unsigned int * array, size_t n ) {

	m_face_indices  = array;
	m_num_face      = n;

}


void PLYIO::setColorArray( float * array, size_t n ) {

	m_color = (unsigned char *) malloc( n * 3 * sizeof(unsigned char) );
	for ( int i = 0; i < ( 3 * n ); i++ ) {
		m_color[i] = (unsigned char) ( array[i] * 255 );
	}
	m_num_color = n;

}


void PLYIO::setColorArray( unsigned char * array, size_t n ) {

	m_color     = array;
	m_num_color = n;

}


void PLYIO::setConfidenceArray( float * array, size_t n ) {

	m_confidence     = array;
	m_num_confidence = n;

}


void PLYIO::setIntensityArray( float * array, size_t n ) {

	m_intensity     = array;
	m_num_intensity = n;

}


void PLYIO::setIndexedVertexArray( float ** arr, size_t count ) {

	m_vertices = (float *) realloc( m_vertices, count * 3 * sizeof(float) );
	for ( int i = 0; i < count; i++ ) {
		m_vertices[ i * 3     ] = arr[i][0];
		m_vertices[ i * 3 + 1 ] = arr[i][1];
		m_vertices[ i * 3 + 2 ] = arr[i][2];
	}

}

void PLYIO::setIndexedNormalArray( float ** arr, size_t count ) {

	m_normals = (float *) realloc( m_normals, count * 3 * sizeof(float) );
	for ( int i = 0; i < count; i++ ) {
		m_normals[ i * 3     ] = arr[i][0];
		m_normals[ i * 3 + 1 ] = arr[i][1];
		m_normals[ i * 3 + 2 ] = arr[i][2];
	}

}


void PLYIO::save( string filename, e_ply_storage_mode mode, 
		vector<string> obj_info, vector<string> comment ) {


	p_ply oply = ply_create( filename.c_str(), mode, NULL );
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
	if ( m_color ) {
		if ( m_num_color != m_num_vertex ) {
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
	if ( m_intensity ) {
		if ( m_num_intensity != m_num_vertex ) {
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
	if ( m_confidence ) {
		if ( m_num_confidence != m_num_vertex ) {
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
	if ( m_normals ) {
		if ( m_num_normal != m_num_vertex ) {
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
			ply_write( oply, m_color[ i * 3     ] ); /* red */
			ply_write( oply, m_color[ i * 3 + 1 ] ); /* green */
			ply_write( oply, m_color[ i * 3 + 2 ] ); /* blue */
		}
		if ( intensity ) {
			ply_write( oply, m_intensity[ i ] );
		}
		if ( confidence ) {
			ply_write( oply, m_confidence[ i ] );
		}
		if ( normal ) {
			ply_write( oply, (double) m_normals[ i * 3     ] ); /* x */
			ply_write( oply, (double) m_normals[ i * 3 + 1 ] ); /* y */
			ply_write( oply, (double) m_normals[ i * 3 + 2 ] ); /* z */
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
	p_ply ply = ply_open( filename.c_str(), NULL );

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
	int32_t n;
	p_ply_element elem  = NULL;
	while ( ( elem = ply_get_next_element( ply, elem ) ) ) {
		ply_get_element_info( elem, &name, &n );
		if ( !strcmp( name, "vertex" ) ) {
			m_num_vertex = n;
			p_ply_property prop = NULL;
			while ( ( prop = ply_get_next_property( elem, prop ) ) ) {
				ply_get_property_info( prop, (const char **) &name, NULL, NULL, NULL );
				if ( !strcmp( name, "red" ) && readColor ) {
					/* We have color information */
					m_num_color = n;
				} else if ( !strcmp( name, "confidence" ) && readConfidence ) {
					/* We have confidence information */
					m_num_confidence = n;
				} else if ( !strcmp( name, "intensity" ) && readIntensity ) {
					/* We have intensity information */
					m_num_intensity = n;
				} else if ( !strcmp( name, "nx" ) && readNormals ) {
					/* We have normals */
					m_num_normal = n;
				}
			}
		} else if ( !strcmp( name, "face" ) && readFaces ) {
			m_num_face = n;
		}
	}
	if ( !m_num_vertex ) {
		fprintf( stderr, "warning: No vertices in ply.\n" );
		return;
	}

	/* Allocate memory. */
	m_vertices = ( float * ) malloc( m_num_vertex * 3 * sizeof(float) );
	if ( m_num_color ) {
		m_color = ( unsigned char * ) malloc( m_num_vertex * 3 * sizeof(unsigned char) );
	}
	if ( m_num_confidence ) {
		m_confidence = ( float * ) malloc( m_num_vertex * sizeof(float) );
	}
	if ( m_num_intensity ) {
		m_intensity = ( float * ) malloc( m_num_vertex * sizeof(float) );
	}
	if ( m_num_normal ) {
		m_normals = ( float * ) malloc( m_num_vertex * 3 * sizeof(float) );
	}
	if ( m_num_face ) {
		m_face_indices = ( unsigned int * ) malloc( m_num_face * 3 * sizeof(unsigned int) );
	}
	
	float * vertex        = m_vertices;
	unsigned char * color = m_color;
	float * confidence    = m_confidence;
	float * intensity     = m_intensity;
	float * normal        = m_normals;
	unsigned int * face   = m_face_indices;

	/* Set callbacks. */
	ply_set_read_cb( ply, "vertex", "x", readVertexCb, &vertex, 0 );
	ply_set_read_cb( ply, "vertex", "y", readVertexCb, &vertex, 0 );
	ply_set_read_cb( ply, "vertex", "z", readVertexCb, &vertex, 1 );

	if ( color ) {
		ply_set_read_cb( ply, "vertex", "red",   readColorCb,  &color,  0 );
		ply_set_read_cb( ply, "vertex", "green", readColorCb,  &color,  0 );
		ply_set_read_cb( ply, "vertex", "blue",  readColorCb,  &color,  1 );
	}
	if ( confidence ) {
		ply_set_read_cb( ply, "vertex", "confidence", readVertexCb, &confidence, 1 );
	}
	if ( intensity ) {
		ply_set_read_cb( ply, "vertex", "intensity", readVertexCb, &intensity, 1 );
	}
	if ( normal ) {
		ply_set_read_cb( ply, "vertex", "nx", readVertexCb, &intensity, 0 );
		ply_set_read_cb( ply, "vertex", "ny", readVertexCb, &intensity, 0 );
		ply_set_read_cb( ply, "vertex", "nz", readVertexCb, &intensity, 1 );
	}

	if ( face ) {
		ply_set_read_cb( ply, "face", "vertex_indices", readFaceCb, &face, 0 );
		ply_set_read_cb( ply, "face", "vertex_index", readFaceCb, &face, 0 );
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

	unsigned char ** color;
	ply_get_argument_user_data( argument, (void **) &color, NULL );
	**color = ply_get_argument_value( argument );
	(*color)++;
	return 1;

}


int PLYIO::readFaceCb( p_ply_argument argument ) {

	float ** face;
	int32_t length, value_index;
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


void PLYIO::freeBuffer() {

	if ( m_vertices ) {
		free( m_vertices );
	}
	if ( m_color ) {
		free( m_color );
	}
	if ( m_intensity ) {
		free( m_intensity );
	}
	if ( m_confidence ) {
		free( m_confidence );
	}
	if ( m_normals ) {
		free( m_normals );
	}
	if ( m_face_indices ) {
		free( m_face_indices );
	}
	m_vertices = m_confidence = m_intensity = m_normals = NULL;
	m_color =  NULL;
	m_face_indices = NULL;
	m_num_vertex = m_num_color = m_num_intensity = m_num_confidence
		= m_num_normal = m_num_face = 0;

}


} // namespace lssr
