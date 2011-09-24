/**
 *
 * @file      MeshLoader.cpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 09:16:36 PM
 *
 **/
#include "MeshLoader.hpp"


MeshLoader::MeshLoader() : 
	m_vertices( NULL ),
	m_vertex_colors( NULL ),
	m_vertex_intensity( NULL ),
	m_vertex_confidence( NULL ),
	m_vertex_normals( NULL ),
	m_indexed_vertices( NULL ),
	m_indexed_vertex_colors( NULL ),
	m_indexed_vertex_confidence( NULL ),
	m_indexed_vertex_intensity( NULL ),
	m_indexed_vertex_normals( NULL ),
	m_face_indices( NULL ),
	m_num_face( 0 ),
	m_num_vertex_normals( 0 ),
	m_num_vertex_colors( 0 ),
	m_num_vertex_intensity( 0 ),
	m_num_vertex( 0 ),
	m_num_vertex_confidence( 0 ) {}


float * MeshLoader::getVertexArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex;
	}
	return m_vertices;

}

float * MeshLoader::getVertexNormalArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_normals;
	}
	return m_vertex_normals;

}


uint8_t * MeshLoader::getVertexColorArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_colors;
	}
	return m_vertex_colors;

}


float * MeshLoader::getVertexConfidenceArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_confidence;
	}
	return m_vertex_confidence;

}


float * MeshLoader::getVertexIntensityArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_intensity;
	}
	return m_vertex_intensity;

}

unsigned int * MeshLoader::getFaceArray( size_t * n ) {

	if ( n ) {
		*n = m_num_face;
	}
	return m_face_indices;

}


float ** MeshLoader::getIndexedVertexArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex;
	}

	/* Return NULL if we have no vertices. */
	if ( !m_vertices ) {
		return NULL;
	}


	/* Generate indexed vertex array in not already done. */
	if ( !m_indexed_vertices ) {
		m_indexed_vertices = (float **) malloc( m_num_vertex * sizeof(float **) );
		for ( int i = 0; i < m_num_vertex; i++ ) {
			m_indexed_vertices[i] = m_vertices + ( i * 3 );
		}
	}

	/* Return indexed vertex array */
	return m_indexed_vertices;

}

float ** MeshLoader::getIndexedVertexNormalArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_normals;
	}

	/* Return NULL if we have no normals. */
	if ( !m_vertex_normals ) {
		return NULL;
	}


	/* Generate indexed normal array in not already done. */
	if ( !m_indexed_vertex_normals ) {
		m_indexed_vertex_normals = (float **) 
			malloc( m_num_vertex_normals * sizeof(float **) );
		for ( int i = 0; i < m_num_vertex_normals; i++ ) {
			m_indexed_vertex_normals[i] = m_vertex_normals + ( i * 3 );
		}
	}

	/* Return indexed normals array */
	return m_indexed_vertex_normals;


}


float ** MeshLoader::getIndexedVertexConfidenceArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_confidence;
	}

	/* Return NULL if we have no confidence information. */
	if ( !m_vertex_confidence ) {
		return NULL;
	}


	/* Generate indexed confidence array in not already done. */
	if ( !m_indexed_vertex_confidence ) {
		m_indexed_vertex_confidence = (float **) 
			malloc( m_num_vertex_confidence * sizeof(float **) );
		for ( int i = 0; i < m_num_vertex_confidence; i++ ) {
			m_indexed_vertex_confidence[i] = m_vertex_confidence + ( i * 3 );
		}
	}

	/* Return indexed confidence array */
	return m_indexed_vertex_confidence;

}


float ** MeshLoader::getIndexedVertexIntensityArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_intensity;
	}

	/* Return NULL if we have no intensity information. */
	if ( !m_vertex_intensity ) {
		return NULL;
	}

	/* Generate indexed intensity array in not already done. */
	if ( !m_indexed_vertex_intensity ) {
		m_indexed_vertex_intensity = (float **) 
			malloc( m_num_vertex_intensity * sizeof(float **) );
		for ( int i = 0; i < m_num_vertex_intensity; i++ ) {
			m_indexed_vertex_intensity[i] = m_vertex_intensity + ( i * 3 );
		}
	}

	/* Return indexed intensity array */
	return m_indexed_vertex_intensity;

}


uint8_t ** MeshLoader::getIndexedVertexColorArray( size_t * n ) {

	if ( n ) {
		*n = m_num_vertex_colors;
	}
	if ( !m_vertex_colors ) {
		return NULL;
	}

	if ( !m_indexed_vertex_colors ) {
		m_indexed_vertex_colors = (uint8_t **) 
			malloc( m_num_vertex_colors * sizeof(uint8_t **) );
		for ( int i = 0; i < m_num_vertex_colors; i++ ) {
			m_indexed_vertex_colors[i] = m_vertex_colors + ( i * 3 );
		}
	}
	return m_indexed_vertex_colors;

}


void MeshLoader::setVertexArray( float * array, size_t n ) {

	m_vertices   = array;
	m_num_vertex = n;

}

void MeshLoader::setVertexNormalArray( float * array, size_t n ) {

	m_vertex_normals    = array;
	m_num_vertex_normals = n;

}

void MeshLoader::setFaceArray( unsigned int * array, size_t n ) {

	m_face_indices  = array;
	m_num_face      = n;

}

void MeshLoader::setVertexColorArray( float * array, size_t n ) {

	m_vertex_colors = (uint8_t *) malloc( n * 3 * sizeof(uint8_t) );
	for ( int i = 0; i < ( 3 * n ); i++ ) {
		m_vertex_colors[i] = (uint8_t) ( array[i] * 255 );
	}
	m_num_vertex_colors = n;

}

void MeshLoader::setVertexColorArray( uint8_t * array, size_t n ) {

	m_vertex_colors     = array;
	m_num_vertex_colors = n;

}


void MeshLoader::setVertexConfidenceArray( float * array, size_t n ) {

	m_vertex_confidence     = array;
	m_num_vertex_confidence = n;

}


void MeshLoader::setVertexIntensityArray( float * array, size_t n ) {

	m_vertex_intensity     = array;
	m_num_vertex_intensity = n;

}


void MeshLoader::setIndexedVertexArray( float ** arr, size_t count ) {

	m_vertices = (float *) malloc( count * 3 * sizeof(float) );
	for ( int i = 0; i < count; i++ ) {
		m_vertices[ i * 3     ] = arr[i][0];
		m_vertices[ i * 3 + 1 ] = arr[i][1];
		m_vertices[ i * 3 + 2 ] = arr[i][2];
	}

}


void MeshLoader::setIndexedVertexNormalArray( float ** arr, size_t count ) {

	m_vertex_normals = (float *) malloc( count * 3 * sizeof(float) );
	for ( int i = 0; i < count; i++ ) {
		m_vertex_normals[ i * 3     ] = arr[i][0];
		m_vertex_normals[ i * 3 + 1 ] = arr[i][1];
		m_vertex_normals[ i * 3 + 2 ] = arr[i][2];
	}

}


void MeshLoader::freeBuffer() {

	m_vertices = m_vertex_confidence = m_vertex_intensity = m_vertex_normals = NULL;
	m_vertex_colors = NULL;
	m_face_indices = NULL;
	m_num_vertex = m_num_vertex_colors = m_num_vertex_intensity = m_num_vertex_confidence
		= m_num_vertex_normals = m_num_face = 0;

}
