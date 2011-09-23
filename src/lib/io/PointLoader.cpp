/**
 *
 * @file      PointLoader.cpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 11:23:57 PM
 *
 **/

#include "PointLoader.hpp"


PointLoader::PointLoader() :
	m_points( NULL ),
	m_point_normals( NULL ),
	m_point_colors( NULL ),
	m_point_intensities( NULL ),
	m_point_confidence( NULL ),
	m_indexed_points( NULL ),
	m_indexed_point_normals( NULL ),
	m_indexed_point_intensities( NULL ),
	m_indexed_point_confidence( NULL ),
	m_indexed_point_colors( NULL ),
	m_num_points( 0 ),
	m_num_point_colors( 0 ),
	m_num_point_normals( 0 ),
	m_num_point_intensities( 0 ),
	m_num_point_confidence( 0 ) {}


float * PointLoader::getPointArray( size_t * n ) {

	*n = m_num_points;
	return m_points;

}


uint8_t * PointLoader::getPointColorArray( size_t *n ) {

	*n = m_num_point_colors;
	return m_point_colors;

}


float * PointLoader::getPointNormalArray( size_t *n ) {

	*n = m_num_point_normals;
	return m_point_normals;

}


float * PointLoader::getPointIntensityArray( size_t *n ) {

	*n = m_num_point_intensities;
	return m_point_intensities;

}


float * PointLoader::getPointConfidenceArray( size_t *n ) {

	*n = m_num_point_confidence;
	return m_point_confidence;

}


size_t PointLoader::getNumPoints() {

	return m_num_points;

}


uint8_t ** PointLoader::getIndexedPointColorArray( size_t * n ) {

	if ( n ) {
		*n = m_num_point_colors;
	}
	if ( !m_point_colors ) {
		return NULL;
	}

	if ( !m_indexed_point_colors ) {
		m_indexed_point_colors = (uint8_t **) 
			malloc( m_num_point_colors * sizeof(uint8_t **) );
		for ( int i = 0; i < m_num_point_colors; i++ ) {
			m_indexed_point_colors[i] = m_point_colors + ( i * 3 );
		}
	}
	return m_indexed_point_colors;

}


float ** PointLoader::getIndexedPointNormalArray( size_t * n ) {

	return getIndexedArrayf( n, m_num_point_normals, &m_point_normals, 
			&m_indexed_point_normals );

}


float ** PointLoader::getIndexedPointArray( size_t * n ) {

	return getIndexedArrayf( n, m_num_points, &m_points, &m_indexed_points );

}


float ** PointLoader::getIndexedPointIntensityArray( size_t * n ) {

	return getIndexedArrayf( n, m_num_point_intensities, &m_point_intensities,
			&m_indexed_point_intensities );

}


float ** PointLoader::getIndexedPointConfidenceArray( size_t * n ) {

	return getIndexedArrayf( n, m_num_point_confidence, &m_point_confidence,
			&m_indexed_point_confidence );

}


float ** PointLoader::getIndexedArrayf( size_t * n, const size_t num, 
		float ** arr1d, float *** arr2d ) {

	if ( n ) {
		*n = num;
	}

	/* Return NULL if we have no data. */
	if ( !(*arr1d) ) {
		return NULL;
	}

	/* Generate indexed intensity array in not already done. */
	if ( !(*arr2d) ) {
		*arr2d = (float **) malloc( num * sizeof(float **) );
		for ( int i = 0; i < num; i++ ) {
			arr2d[i] = arr1d + ( i * 3 );
		}
	}

	/* Return indexed intensity array */
	return *arr2d;

}


void PointLoader::setPointArray( float * array, size_t n ) {

	m_num_points = n;
	m_points = array;

}


void PointLoader::setPointColorArray( uint8_t * array, size_t n ) {

	m_num_point_colors = n;
	m_point_colors = array;

}


void PointLoader::setPointNormalArray( float * array, size_t n ) {

	m_num_point_normals = n;
	m_point_normals = array;

}


void PointLoader::setPointIntensityArray( float * array, size_t n ) {

	m_num_point_intensities = n;
	m_point_intensities = array;

}


void PointLoader::setPointConfidenceArray( float * array, size_t n ) {

	m_num_point_confidence = n;
	m_point_confidence = array;

}
