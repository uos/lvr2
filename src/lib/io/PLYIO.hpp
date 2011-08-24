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

#include "PLYProperty.hpp"
#include "PLYElement.hpp"

#include <iostream>
#include <fstream>
#include <vector>

#include <cassert>

using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;

#define BUFFER_SIZE 1024
#define OUT_BUFFER_SIZE 20000

namespace lssr
{

/**
 * @brief A class for input and output to ply files.
 */
class PLYIO : public BaseIO,  public PointLoader, public MeshLoader
{


public:

	/**
	 * @brief Ctor.
	 */
	PLYIO();

	/**
	 * @brief Adds the given element to the file
	 *
	 * @param element 		A ply element description
	 */
	void addElement(PLYElement* e);

	/**
	 * @brief Sets the vertex array (for meshes)
	 *
	 * @param array			A vertex array
	 * @param n				The number of vertices in the array
	 *
	 * This version uses an interlaced array. Hence the number of floats
	 * in the array is 3 * \ref{n}.
	 */
	void setVertexArray(float* array, size_t n);

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
	void save(string filename, bool binary);

    /**
     * @brief Save the currently present information to the given file
     *
     * @param filename      The output file. The data is writte in ASCII format.
     */
	void save(string filename)
	{
	    save(filename, false);
	}


	/**
	 * @brief Reads all supported information from the given file
	 * @param filename		A ply file
	 */
	void read(string filename);

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
	float* getColorArray(size_t &);

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

	/**
	 * @brief Returns true if the current element contains the provided
	 * 		  element
	 * @param e				A ply element description object
	 */
	bool containsElement(PLYElement& e);

	/**
	 * @brief Returns true if the current element lists contains an
	 *        element with the given name.
	 */
	bool containsElement(string elementName);

	/**
	 * @brief Checks if \ref{e} has property \ref{p}
	 */
	bool hasProperty(PLYElement& e, Property& p);

	/**
	 * @brief Prints all elements and properties to stdout.
	 */
	void printElementsInHeader();


	float* getVertexNormalArray(size_t &n) { return getNormalArray(n); };
	float* getVertexColorArray(size_t &n) { n = 0; return 0;}

private:

	float** interlacedBufferToIndexedBuffer(float* src, size_t n);
	float*	indexedBufferToInterlacedBuffer(float** src, size_t n);

	void writeHeader(ofstream& str);
	void writeElements(ofstream &str);
	void writeFacesBinary(ofstream &str, PLYElement* e);
	void writeFacesASCII(ofstream &str, PLYElement* e);
	void writePointsBinary(ofstream &str, PLYElement* e);
	void writePointsASCII(ofstream &str, PLYElement* e);
	void writeVerticesBinary(ofstream &str, PLYElement* e);
	void writeVerticesASCII(ofstream &str, PLYElement* e);
	void writeNormalsBinary(ofstream &out, PLYElement* e);
	void writeNormalsASCII(ofstream &out, PLYElement* e);

	void readVerticesBinary(ifstream &in, PLYElement* descr);
	void readFacesBinary(ifstream &in, PLYElement* descr);
	void readNormalsBinary(ifstream &in, PLYElement* descr);
	void readPointsBinary(ifstream &in, PLYElement* descr);

	void readVerticesASCII(ifstream &in, PLYElement* descr);
	void readFacesASCII(ifstream &in, PLYElement* descr);
	void readNormalsASCII(ifstream &in, PLYElement* descr);
	void readPointsASCII(ifstream &in, PLYElement* descr);

	void readHeader(ifstream& str);

	char* putElementInBuffer(char* buffer, string s,  float value);

	bool isSupported(string element_name);
	bool parseHeaderLine(const char* line);

	void loadElements(ifstream& in);

	void deleteBuffers();
	void allocVertexBuffers(PLYElement* dscr);
	void allocPointBuffers(PLYElement* descr);

	template<typename T>
	void copyElementToBuffer(ifstream &str, Property*, T* buffer, size_t position);

	template<typename T>
	void copyElementToIndexedBuffer(ifstream &str, Property*, T** buffer, size_t position, size_t index);

	bool 					m_binary;
	vector<PLYElement*> 	m_elements;

};



}

#endif
