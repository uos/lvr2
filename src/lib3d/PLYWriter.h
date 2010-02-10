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

private:
	void writeHeader(ofstream& str);
	void writeElements(ofstream &str);
	void writeFacesBinary(ofstream &str, PLYElement* e);
	void writeFacesASCII(ofstream &str, PLYElement* e);
	void writeVerticesBinary(ofstream &str, PLYElement* e);
	void writeVerticesASCII(ofstream &str, PLYElement* e);

	char* putElementInBuffer(char* buffer, string s,  float value);

	bool isSupported(string element_name);

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
