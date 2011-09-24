/**
 *
 * @file      MeshLoader.hpp
 * @brief     
 * @details   
 * 
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   110922
 * @date      09/22/2011 09:16:36 PM
 *
 **/

#ifndef MESHIO_HPP_
#define MESHIO_HPP_

#include <stdint.h>
#include <cstddef>
#include <cstdlib>

class MeshLoader {

	public:
		MeshLoader();
		void setVertexArray( float * array, size_t n );
		void setVertexConfidenceArray( float * array, size_t n );
		void setVertexIntensityArray( float * array, size_t n );
		void setVertexNormalArray( float * array, size_t n );
		void setVertexColorArray( float * array, size_t n );
		void setVertexColorArray( uint8_t * array, size_t n );

		void setIndexedVertexArray( float ** arr, size_t size );
		void setIndexedVertexNormalArray( float ** arr, size_t size );

		float *    getVertexArray( size_t * n = NULL );
		float *    getVertexNormalArray( size_t * n = NULL );
		float *    getVertexConfidenceArray( size_t * n = NULL );
		float *    getVertexIntensityArray( size_t * n = NULL );
		uint8_t *  getVertexColorArray( size_t * n = NULL );

		float **   getIndexedVertexArray( size_t * n = NULL );
		float **   getIndexedVertexNormalArray( size_t * n = NULL );
		float **   getIndexedVertexConfidenceArray( size_t * n = NULL );
		float **   getIndexedVertexIntensityArray( size_t * n = NULL );
		uint8_t ** getIndexedVertexColorArray( size_t * n = NULL );

		unsigned int * getFaceArray( size_t * n = NULL );
		void setFaceArray( unsigned int * array, size_t n );

		void freeBuffer();

	protected:

		float        * m_vertices;
		uint8_t      * m_vertex_colors;
		float        * m_vertex_confidence;
		float        * m_vertex_intensity;
		float        * m_vertex_normals;

		float       ** m_indexed_vertices;
		uint8_t     ** m_indexed_vertex_colors;
		float       ** m_indexed_vertex_confidence;
		float       ** m_indexed_vertex_intensity;
		float       ** m_indexed_vertex_normals;

		unsigned int * m_face_indices;

		uint32_t       m_num_vertex;
		uint32_t       m_num_vertex_normals;
		uint32_t       m_num_vertex_colors;
		uint32_t       m_num_vertex_confidence;
		uint32_t       m_num_vertex_intensity;
		uint32_t       m_num_face;

};

#endif /* MESHIO_HPP_ */
