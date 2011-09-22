///*
// * PLYWriter.h
// *
// *  Created on: 11.11.2009
// *      Author: Thomas Wiemann
// */
//
//

#ifndef __PLY_IO_H__
#define __PLY_IO_H__

#include "BaseIO.hpp"
#include "MeshLoader.hpp"
#include "PointLoader.hpp"
#include <rply.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cassert>

using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;

namespace lssr
{

/**
 * @brief A class for input and output to ply files.
 */
class PLYIO : public BaseIO, public MeshLoader
// ,  public PointLoader
{

	float         * m_vertices;
	unsigned char * m_color;
	float         * m_confidence;
	float         * m_intensity;
	float         * m_normals;
	unsigned int  * m_face_indices;
	uint32_t        m_num_vertex;
	uint32_t        m_num_normal;
	uint32_t        m_num_color;
	uint32_t        m_num_confidence;
	uint32_t        m_num_intensity;
	uint32_t        m_num_face;

public:

	/**
	 * @brief Ctor.
	 */
	PLYIO();

	/**
	 * @brief Sets the vertex array (for meshes)
	 * This version uses an interlaced array. Hence the number of floats in the
	 * array is 3 * \ref{n}.
	 *
	 * @param array  Vertex array.
	 * @param n      Number of vertices in the array.
	 */
	void setVertexArray(float* array, size_t n);
	float * getConfidenceArray( size_t &n );
	float * getIntensityArray( size_t &n );
	void setConfidenceArray( float * array, size_t n );
	void setIntensityArray( float * array, size_t n );
	static int readVertexCb( p_ply_argument argument );
	static int readColorCb( p_ply_argument argument );
	static int readFaceCb( p_ply_argument argument );
	unsigned char ** getIndexedColorArray( size_t &n );

	/**
	 * @brief Sets the normal array (for meshes)
	 *
	 * @param array			A set of normal coordinated
	 * @param n				The number of normals in the array
	 *
     * This version uses an interlaced array. Hence the number of floats
	 * in the array is 3 * \ref{n}.
	 */
	void setNormalArray(float* array, size_t n);

	/**
	 * @brief Sets the color array (for vertex colors)
	 *
	 * @param array			An array containing color information
	 * @param n				The number of elements in the array
	 */
	void setColorArray(float* array, size_t n);
	void setColorArray( unsigned char * array, size_t n );

	/**
	 * @brief Sets the index buffer
	 *
	 * @param array 		A index buffer
	 * @param n				The number of faces encoded in the buffer
	 *
	 * The number of face in the buffer is \ref{n} / 3 since each face consists
	 * of three vertices that are referenced in the buffer.
	 */
	void setIndexArray(unsigned int* array, size_t n);

	/**
	 * @brief Save the currently present information to the given file
	 *
	 * @param filename		The output file
	 * @param binary		If, the data is writen in binary format (default). Set
	 * 						this param to false to create an ASCII ply file
	 */
	void save( string filename, e_ply_storage_mode mode, 
			vector<string> obj_info = vector<string>(), 
			vector<string> comment = vector<string>() );

    /**
     * @brief Save the currently present information to the given file
     *
     * @param filename      The output file. The data is writte in ASCII format.
     */
	void save(string filename)
	{
	    save( filename, PLY_ASCII );
	}

	void read( string filename, bool readColor, bool readConfidence = true, 
			bool readIntensity = true, bool readNormals = true, 
			bool readFaces = true );

	/**
	 * @brief Reads all supported information from the given file
	 * @param filename		A ply file
	 */
	void read( string filename );

	/**
	 * @brief Dtor.
	 */
	virtual ~PLYIO();

	/**
	 * @brief Returns the interlaced vertex array (or a null pointer if
	 * 		  not set).
	 * @param n				Contains the number of Vertices in the array
	 * @return				A pointer to vertex data
	 */
	float* getVertexArray(size_t &n);

	/**
	 * @brief Returns the interlaced normal array (or a null pointer if
	 * 		  not set).
	 * @param n				Contains the number of Vertices in the array
	 * @return				A pointer to normal data
	 */
	float* getNormalArray(size_t &n);

	/**
	 * @brief Returns the interlaced color array (or a null pointer if
	 * 		  not set).
	 * @param n				Contains the number of Vertices in the array
	 * @return				A pointer to color data
	 */
	unsigned char * getColorArray(size_t &);

	/**
	 * @brief Returns an index accessible representation (2D array) of
	 * 		  the vertex data.
	 *
	 * @param n				Contains the number of vertices
	 * @return				A pointer to 2D vertex data.
	 *
	 * Using this method, the preferred interlaced representation is
	 * converted into a 2D array. Be careful with large data sets since
	 * the information is duplicated.
	 */
	float** getIndexedVertexArray(size_t &n);


	/**
	 * @brief Returns an index accessible representation (2D array) of
	 * 		  the vertex data.
	 *
	 * @param n				Contains the number of vertices
	 * @return				A pointer to 2D vertex data.
	 *
	 * Using this method, the preferred interlaced representation is
	 * converted into a 2D array. Be careful with large data sets since
	 * the information is duplicated.
	 */
	float** getIndexedNormalArray(size_t &n);

	/**
	 * @brief Adds indexed vertex data.
	 *
	 * @param arr			Indexed vertex data
	 * @param size			The number of vertices in the provided 2D array
	 *
	 * The provided data is converted. Beware of memory overhead.
	 */
	void setIndexedVertexArray(float** arr, size_t size);

	/**
	 * @brief Adds indexed vertex data.
	 *
	 * @param arr			Indexed vertex data
	 * @param size			The number of vertices in the provided 2D array
	 *
	 * The provided data is converted. Beware of memory overhead.
	 */
	void setIndexedNormalArray(float** arr, size_t size);

	/**
	 * @brief Returns the index array of a mesh
	 * @param n 			The number of faces in the mesh
	 * @return				A pointer to the index data
	 */
	unsigned int* getIndexArray(size_t &n);

	void freeBuffer();

	float * getVertexNormalArray( size_t &n );
	float * getVertexColorArray( size_t &n );
	float * getColorArray3f( size_t &n );

};



}

#endif
