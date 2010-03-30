///*
// * PLYWriter.h
// *
// *  Created on: 11.11.2009
// *      Author: Thomas Wiemann
// */
//
//

#ifndef __PLY_WRITER_H__
#define __PLY_WRITER_H__

#include "PLYProperty.h"
#include "PLYElement.h"

#include <iostream>
#include <fstream>
#include <vector>

#include <cassert>

using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;

class PLYIO {

public:
	PLYIO();

	void addElement(PLYElement* e);
	void setVertexArray(float* array, size_t n);
	void setNormalArray(float* array, size_t n);
	void setColorArray(float* array, size_t n);
	void setIndexArray(unsigned int* array, size_t n);

	void save(string filename, bool binary = true);
	void read(string filename);

	virtual ~PLYIO();

	float* getVertexArray(size_t &n);
	float* getNormalArray(size_t &n);
	float* getColorArray(size_t &);

	unsigned int* getIndexArray(size_t &n);

	bool containsElement(PLYElement& e);
	bool hasProperty(PLYElement& e, Property& p);

	void printElementsInHeader();

private:

	void writeHeader(ofstream& str);
	void writeElements(ofstream &str);
	void writeFacesBinary(ofstream &str, PLYElement* e);
	void writeFacesASCII(ofstream &str, PLYElement* e);
	void writeVerticesBinary(ofstream &str, PLYElement* e);
	void writeVerticesASCII(ofstream &str, PLYElement* e);

	void readVerticesBinary(ifstream &in, PLYElement* descr);
	void readFacesBinary(ifstream &in, PLYElement* descr);
	void readVerticesASCII(ifstream &in, PLYElement* descr);
	void readFacesASCII(ifstream &in, PLYElement* descr);

	void readHeader(ifstream& str);

	char* putElementInBuffer(char* buffer, string s,  float value);

	bool isSupported(string element_name);
	bool parseHeaderLine(const char* line);

	void loadElements(ifstream& in);

	void deleteBuffers();
	void allocVertexBuffers(PLYElement* dscr);

	void copyElementToVertexBuffer(ifstream &str, Property*, float* buffer, size_t position);

	float*					m_vertices;
	float*					m_normals;
	float*					m_colors;
	unsigned int*			m_indices;

	size_t					m_numberOfVertices;
	size_t					m_numberOfFaces;

	bool 					m_binary;

	vector<PLYElement*> 	m_elements;



};

#endif
