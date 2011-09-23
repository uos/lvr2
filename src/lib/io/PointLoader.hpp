/**
 *
 * @file      PointLoader.hpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 11:23:57 PM
 *
 **/

#ifndef POINTIO_HPP_
#define POINTIO_HPP_

#include <stdint.h>
#include <cstddef>
#include <cstdlib>

class PointLoader {

	public:
		PointLoader();

		void setPointArray( float * array, size_t n );
		void setPointColorArray( uint8_t * array, size_t n );
		void setPointNormalArray( float * array, size_t n );
		void setPointIntensityArray( float * array, size_t n );
		void setPointConfidenceArray( float * array, size_t n );

		virtual float * getPointArray( size_t * n = NULL );
		virtual uint8_t * getPointColorArray( size_t *n = NULL );
		virtual float * getPointNormalArray( size_t *n = NULL );
		virtual float * getPointIntensityArray( size_t *n = NULL );
		virtual float * getPointConfidenceArray( size_t *n = NULL );

		float ** getIndexedPointArray( size_t * n = NULL );
		uint8_t ** getIndexedPointColorArray( size_t * n = NULL );
		float ** getIndexedPointNormalArray( size_t * n = NULL );
		float ** getIndexedPointIntensityArray( size_t * n = NULL );
		float ** getIndexedPointConfidenceArray( size_t * n = NULL );

		virtual size_t getNumPoints();

		float ** getIndexedArrayf( size_t * n, const size_t num, float ** arr1d,
				float *** arr2d );

	protected:

		float    * m_points;
		float    * m_point_normals;
		uint8_t  * m_point_colors;
		float    * m_point_intensities;
		float    * m_point_confidence;

		float   ** m_indexed_points;
		float   ** m_indexed_point_normals;
		float   ** m_indexed_point_intensities;
		float   ** m_indexed_point_confidence;
		uint8_t ** m_indexed_point_colors;

		size_t     m_num_points;
		size_t     m_num_point_colors;
		size_t     m_num_point_normals;
		size_t     m_num_point_intensities;
		size_t     m_num_point_confidence;

};


#endif /* POINTIO_HPP_ */
