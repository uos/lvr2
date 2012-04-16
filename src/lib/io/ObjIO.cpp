/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * ObjIO.cpp
 *
 *  @date 07.11.2011
 *  @author Florian Otte (fotte@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Lars Kiesow (lkiesow@uos.de)
 *  @author Denis Meyer (denmeyer@uos.de)
 */

#include "ObjIO.hpp"
#include "objLoader.h"

#include <climits>
#include <iostream>
#include <fstream>
#include "Timestamp.hpp"
#include "boost/tuple/tuple.hpp"
#include "../geometry/Vertex.hpp"
#include "../display/GlTexture.hpp"
#include "../display/TextureFactory.hpp"
#include <string.h>
#include <locale.h>
#include <sstream>

#include "PLYIO.hpp"

namespace lssr
{
using namespace std; // Bitte vergebt mir....
// Meinst du wirklich, dass ich dir so etwas durchgehen lassen kann?

/*ModelPtr ObjIO::read( string filename ) // TODO: Format correctly
{
	setlocale(LC_NUMERIC, "en_US");
	ifstream f (filename.c_str());
	if (!f.is_open())
	{
		cerr << timestamp << "File '" << filename << "' could not be opened." << endl;
		return ModelPtr( new Model );
	}
	f.close();

	char *fn = const_cast<char *>(filename.c_str());
	objLoader *objData = new objLoader();
	objData->load(fn);

	// Buffer count variables
	size_t numVertices      = objData->vertexCount;
	size_t numVertexNormals = objData->normalCount;
	size_t numFaces         = objData->faceCount;
	size_t numTextures      = objData->textureCount;
	size_t numSpheres       = objData->sphereCount;
	size_t numPlanes        = objData->planeCount;
	size_t numPointLights   = objData->lightPointCount;
	size_t numDiscLights    = objData->lightDiscCount;
	size_t numQuadLights    = objData->lightQuadCount;
	size_t numMaterials     = objData->materialCount;
	size_t numVertexColors  = 0; // TODO: Set

	// Some output
	cout << timestamp << endl;
	cout << "[obj_io]" << "Number of vertices: "            << numVertices      << endl;
	cout << "[obj_io]" << "Number of vertex normals: "      << numVertexNormals << endl;
	cout << "[obj_io]" << "Number of vertex colors: "       << numVertexColors  << endl;
	cout << "[obj_io]" << "Number of faces: "               << numFaces         << endl;
	cout << "[obj_io]" << "Number of texture coordinates: " << numTextures      << endl;
	cout << "[obj_io]" << "Number of spheres: "             << numSpheres       << endl;
	cout << "[obj_io]" << "Number of planes: "              << numPlanes        << endl;
	cout << "[obj_io]" << "Number of point lights: "        << numPointLights   << endl;
	cout << "[obj_io]" << "Number of disc lights: "         << numDiscLights    << endl;
	cout << "[obj_io]" << "Number of quad lights: "         << numQuadLights    << endl;
	cout << "[obj_io]" << "Number of materials: "           << numMaterials     << endl;
	if(objData->camera != NULL)
	{
		cout << "[obj_io]" << "Found a camera" << endl;
	}
	cout << endl;

	// Buffers
	floatArr 	vertices;
	floatArr 	vertexNormals;
	ucharArr 	vertexColors;
	uintArr 	faceIndices;
	uintArr 	materialIndexBuffer;
	materialArr materialBuffer;
	floatArr 	textureCoordBuffer;

	vector<GlTexture*> textures;

	// Allocate memory
	if ( numVertices )
	{
		vertices = floatArr( new float[ numVertices * 3 ] );
	}
	if ( numVertexNormals )
	{
		vertexNormals = floatArr( new float[ numVertexNormals * 3 ] );
	}
	if ( numVertexColors )
	{
		vertexColors = ucharArr( new uchar[ numVertexColors * 3 ] );
	}
	if ( numFaces )
	{
		faceIndices = uintArr( new unsigned int[ numFaces * 3 ] );
		materialIndexBuffer = uintArr( new unsigned int[ numFaces ] );
	}
	if( numVertices )
	{
		textureCoordBuffer = floatArr( new float[ numVertices * 3 ] );
	}

	if(numMaterials)
	{
		materialBuffer = materialArr( new Material*[numMaterials]);
	}


	// vertices
	for(int i = 0; i < numVertices; ++i)
	{
		obj_vector *o = objData->vertexList[i];
		vertices[ i * 3 ]     = o->e[ 0 ];
		vertices[ i * 3 + 1 ] = o->e[ 1 ];
		vertices[ i * 3 + 2 ] = o->e[ 2 ];
	}

	// vertex normals
	for(int i = 0; i < numVertexNormals; ++i)
	{
		obj_vector *o = objData->normalList[i];
		vertexNormals[ i * 3 ]     = o->e[ 0 ];
		vertexNormals[ i * 3 + 1 ] = o->e[ 1 ];
		vertexNormals[ i * 3 + 2 ] = o->e[ 2 ];
	}

	// vertex colors
	for(int i = 0; i < numVertexColors; ++i)
	{
		vertexColors[ i * 3 ]    = 255;
		vertexColors[ i * 3 + 1] = 0;
		vertexColors[ i * 3 + 2] = 0;
	}

	// faces
	for(int i = 0; i < numFaces; ++i)
	{
		obj_face *o              = objData->faceList[ i ];
		faceIndices[ i * 3 ]     = o->vertex_index[ 0 ];
		faceIndices[ i * 3 + 1 ] = o->vertex_index[ 1 ];
		faceIndices[ i * 3 + 2 ] = o->vertex_index[ 2 ];

		materialIndexBuffer[ i ]     = o->material_index;
	}

	// texture coordinates
	for(int i = 0; i < numVertices; ++i)
	{
		obj_vector *o = objData->textureList[i];
		textureCoordBuffer[ i * 3 ]     = o->e[ 0 ];
		textureCoordBuffer[ i * 3 + 1 ] = 1 - o->e[ 1 ];
		textureCoordBuffer[ i * 3 + 2 ] = o->e[ 2 ];
 	}

	// Parse materials...
	map<string, int> textureNameMap;
	map<string, int>::iterator it;
	int textureIndex = 0;
	for(int i = 0; i < numMaterials; ++i)
	{
		obj_material *o = objData->materialList[i];

		materialBuffer[i] = new Material;
		materialBuffer[i]->r = o->amb[0];
		materialBuffer[i]->g = o->amb[1];
		materialBuffer[i]->b = o->amb[2];

		string texname(o->texture_filename);

		if(texname == "")
		{
			materialBuffer[i]->texture_index = -1;
		}
		else
		{
			// Test if texture is already loaded
			it = textureNameMap.find(texname);

			if(it != textureNameMap.end())
			{
				materialBuffer[i]->texture_index = it->second;
			}
			else
			{
 				GlTexture* texture = TextureFactory::instance().getTexture(texname);
				if(texture == 0)
				{
					materialBuffer[i]->texture_index = -1;
				}
				else
				{
					materialBuffer[i]->texture_index = textureIndex;
					textures.push_back(texture);
					textureNameMap[texname] = textureIndex;
					textureIndex++;
				}
			}
		}

	}

	// delete obj data object
	delete(objData);

	// Save buffers in model
	MeshBufferPtr mesh;
	if(vertices)
	{
		mesh = MeshBufferPtr( new MeshBuffer );
		mesh->setVertexArray(                  vertices,           numVertices );
		mesh->setVertexColorArray(             vertexColors,       numVertexColors );
		mesh->setVertexNormalArray(            vertexNormals,      numVertexNormals );
		mesh->setFaceArray(                    faceIndices,        numFaces );
		mesh->setFaceMaterialIndexArray(        materialIndexBuffer, numFaces );
		mesh->setMaterialArray ( materialBuffer, numMaterials);
		mesh->setVertexTextureCoordinateArray( textureCoordBuffer, numTextures );
		mesh->setTextureArray(textures);
	}

	ModelPtr m( new Model( mesh ) );
	m_model = m;
	return m;
};
*/
void tokenize(const string& str,
                      vector<string>& tokens,
                      const string& delimiters = " ")
{
    // Skip delimiters at beginning.
    string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (string::npos != pos || string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

void ObjIO::parseMtlFile(
		map<string, int>& matNames,
		vector<Material*>& materials,
		vector<GlTexture*>& textures,
		string mtlname)
{
	cout << "Parsing " << mtlname << endl;

	ifstream in(mtlname.c_str());
	if(in.good())
	{
		char buffer[1024];
		Material* m = 0;
		int matIndex = 0;
		while(in.good())
		{
			in.getline(buffer, 1024);

			// Skip comments
			if(buffer[0] == '#') continue;

			stringstream ss(buffer);
			string keyword;
			ss >> keyword;

			if(keyword == "newmtl")
			{
				string matName;
				ss >> matName;
				map<string, int>::iterator it = matNames.find(matName);
				if(it == matNames.end())
				{
					m = new Material;
					m->r = 128;
					m->g = 128;
					m->b = 128;
					m->texture_index = -1;
					materials.push_back(m);
					matNames[matName] = matIndex;
					matIndex++;

				}
				else
				{
					//m = materials[matNames[matName]];
					cout << "ObjIO::parseMtlFile(): Warning: Duplicate material: " << matName << endl;
				}
			}
			else if(keyword == "Ka")
			{
				float r, g, b;
				ss >> r >> g >> b;
				m->r = (uchar)r * 255;
				m->g = (uchar)g * 255;
				m->b = (uchar)b * 255;
			}
			else if(keyword == "map_Kd")
			{
				string texname;
				ss >> texname;
				GlTexture* texture = TextureFactory::instance().getTexture(texname);
				textures.push_back(texture);
				m->texture_index = textures.size() - 1;
			}
			else
			{
				continue;
			}
		}
	}
	else
	{
		cout << "ObjIO::parseMtlFile(): Error opening '" << mtlname << "'." << endl;
	}
}

ModelPtr ObjIO::read(string filename)
{
	ifstream in(filename.c_str());

	vector<float> 		vertices;
	vector<float> 		normals;
	vector<float> 		texcoords;
	vector<uint>		faceMaterials;
	vector<uint>  		faces;
	vector<Material*> 	materials;
	vector<GlTexture*>	textures;

	map<string, int> matNames;

	int currentMat = 0;

	if(in.good())
	{
		char buffer[1024];
		while(in.good())
		{
			in.getline(buffer, 1024);

			// Skip comments
			if(buffer[0] == '#') continue;

			stringstream ss(buffer);
			string keyword;
			ss >> keyword;
			float x, y, z;
			if(keyword == "v")
			{
				ss >> x >> y >> z;
				vertices.push_back(x);
				vertices.push_back(y);
				vertices.push_back(z);
			}
			else if(keyword == "vt")
			{
				ss >> x >> y >> z;
				texcoords.push_back(x);
				texcoords.push_back(1 - y);
				texcoords.push_back(z);
			}
			else if(keyword == "vn")
			{
				ss >> x >> y >> z;
				normals.push_back(x);
				normals.push_back(y);
				normals.push_back(z);
			}
			else if(keyword == "f")
			{
				vector<string> tokens;
				tokenize(buffer, tokens);

				if(tokens.size() < 4)
					continue;

				vector<string> tokens2;
				tokenize(tokens.at(1),tokens2,"/");
				int a = atoi(tokens2.at(0).c_str());
				tokens2.clear();

				tokenize(tokens.at(2),tokens2,"/");
				int b = atoi(tokens2.at(0).c_str());
				tokens2.clear();

				tokenize(tokens.at(3),tokens2,"/");
				int c = atoi(tokens2.at(0).c_str());
				tokens2.clear();

				faces.push_back(a - 1);
				faces.push_back(b - 1);
				faces.push_back(c - 1);

				// Use current material
				faceMaterials.push_back(currentMat);
			}
			else if(keyword == "usemtl")
			{
				string mtlname;
				ss >> mtlname;
				// Find name and set current material
				map<string, int>::iterator it = matNames.find(mtlname);
				if(it == matNames.end())
				{
					cout << "ObjIO:read(): Warning material '" << mtlname << "' is undefined." << endl;
				}
				else
				{
					currentMat = it->second;
				}
			}
			else if(keyword == "mtllib")
			{
				string mtlfile;
				ss >> mtlfile;
				parseMtlFile(matNames, materials, textures, mtlfile);
			}
		}

	}
	else
	{
		cout << timestamp << "ObjIO::read(): Unable to open file'" << filename << "'." << endl;
	}

/*	cout << "OBJ INFO:" << endl;
	cout << vertices.size() / 3<< endl;
	cout << normals.size() / 3<< endl;
	cout << faces.size() / 3 << endl;
	cout << faceMaterials.size() << endl;

	cout << "Mat info: " << endl;
	cout << textures.size() << endl;
	cout << materials.size() << endl;
*/

/*	for(int i = 0; i < materials.size(); i++)
	{
		if(materials[i]->texture_index != -1) cout << materials[i]->texture_index << endl;
	}*/

	MeshBufferPtr mesh = MeshBufferPtr(new MeshBuffer);

	if(materials.size())
	{
		mesh->setMaterialArray(materials);
	}

	if(faceMaterials.size() == faces.size() / 3)
	{
		mesh->setFaceMaterialIndexArray(faceMaterials);
	}
	else
	{
		cout << "ObjIO::read(): Warning: Face material index buffer does not match face number." << endl;
	}

	if(textures.size())
	{
		mesh->setTextureArray(textures);
	}

	mesh->setVertexTextureCoordinateArray(texcoords);
	mesh->setVertexArray(vertices);
	mesh->setVertexNormalArray(normals);
	mesh->setFaceArray(faces);

/*	for(int i = 0; i < vertices.size() / 3; i++)
	{
		cout << vertices[3 * i + 0] << " " << vertices[3 * i + 1] << " " << vertices[3 * i + 2] << endl;
		cout << normals[3 * i + 0] << " " << normals[3 * i + 1] << " " << normals[3 * i + 2] << endl;
		cout << texcoords[3 * i + 0] << " " << texcoords[3 * i + 1] << " " << texcoords[3 * i + 2] << endl;
		cout << endl;
	}

	for(int i = 0; i < faces.size() / 3; i++)
	{
		cout << faces[3 * i + 0] << " " << faces[3 * i + 1] << " " << faces[3 * i + 2] << endl;
		cout << endl;
	}
*/

	ModelPtr m(new Model(mesh));
	m_model = m;
	return m;
}

void ObjIO::save( string filename )
{
	typedef Vertex<uchar> ObjColor;

	size_t lenVertices;
	size_t lenNormals;
	size_t lenColors;
	size_t lenFaces;
	size_t lenTextureCoordinates;
	size_t lenFaceIndices;
	size_t lenFaceMaterials;
	size_t lenFaceMaterialIndices;
	coord3fArr vertices           = m_model->m_mesh->getIndexedVertexArray( lenVertices );
	coord3fArr normals            = m_model->m_mesh->getIndexedVertexNormalArray( lenNormals );
	coord3fArr textureCoordinates = m_model->m_mesh->getIndexedVertexTextureCoordinateArray( lenTextureCoordinates );
	uintArr    faceIndices        = m_model->m_mesh->getFaceArray( lenFaces );
	materialArr materials		  = m_model->m_mesh->getMaterialArray(lenFaceMaterials);
	uintArr	faceMaterialIndices   = m_model->m_mesh->getFaceMaterialIndexArray(lenFaceMaterialIndices);

	std::map<ObjColor, unsigned int> colorMap;


	std::set<unsigned int> materialIndexSet;
	std::set<unsigned int> colorIndexSet;

	ofstream out(filename.c_str());
	ofstream mtlFile("textures.mtl");

	if(out.good())
	{
		out<<"mtllib textures.mtl"<<endl;

		if ( !vertices )
		{
			cerr << "Received no vertices to store. Aborting save operation." << endl;
			return;
		}
		out << endl << endl << "##  Beginning of vertex definitions.\n";

		for( size_t i=0; i < lenVertices; ++i )
		{
			out << "v " << vertices[i][0] << " "
					<< vertices[i][1] << " "
					<< vertices[i][2] << endl;
		}

		out<<endl;

		out << endl << endl << "##  Beginning of vertex normals.\n";
		for( size_t i=0; i < lenNormals; ++i )
		{
			out << "vn " << normals[i][0] << " "
					<< normals[i][1] << " "
					<< normals[i][2] << endl;
		}

		out << endl << endl << "##  Beginning of vertexTextureCoordinates.\n";

		for( size_t i=0; i < lenTextureCoordinates; ++i )
		{
			out << "vt " << textureCoordinates[i][0] << " "
					<< textureCoordinates[i][1] << " "
					<< textureCoordinates[i][2] << endl;
		}


		out << endl << endl << "##  Beginning of faces.\n";
		// format of a face: f v/vt/vn
		for( size_t i = 0; i < lenFaces; ++i )
		{

			Material* m = materials[faceMaterialIndices[i]];
			if(m->texture_index >= 0)
			{
				out << "usemtl texture_" << m->texture_index << endl;
			}
			else
			{
				out << "usemtl color_" << faceMaterialIndices[i] << endl;
			}

			//unsigned int* faceTextureIndices
			//float**       textureCoordinates
			//usemtl....
			// +1 after every index since in obj the 0-th vertex has index 1.
			out << "f "
					<< faceIndices[i * 3 + 0] + 1 << "/"
					<< faceIndices[i * 3 + 0] + 1 << "/"
					<< faceIndices[i * 3 + 0] + 1 << " "
					<< faceIndices[i * 3 + 1] + 1 << "/"
					<< faceIndices[i * 3 + 1] + 1 << "/"
					<< faceIndices[i * 3 + 1] + 1 << " "
					<< faceIndices[i * 3 + 2] + 1 << "/"
					<< faceIndices[i * 3 + 2] + 1 << "/"
					<< faceIndices[i * 3 + 2] + 1 << endl;
		}

		out<<endl;
		out.close();
	}
	else
	{
		cerr << "no good. file! \n";
	}



	if( mtlFile.good() )
	{
		for(int i = 0; i < lenFaceMaterials; i++)
		{
			Material* m = materials[i];
			if(m->texture_index == -1)
			{
				mtlFile << "newmtl color_" << i << endl;
				mtlFile << "Ka "
						<< m->r / 255.0f << " "
						<< m->g / 255.0f << " "
						<< m->b / 255.0f << endl;
				mtlFile << "Kd "
						<< m->r / 255.0f << " "
						<< m->g / 255.0f << " "
						<< m->b / 255.0f << endl;
			}
			else
			{
				mtlFile << "newmtl texture_"      << m->texture_index << endl;
				mtlFile << "Ka 1.000 1.000 1.000" << endl;
				mtlFile << "Kd 1.000 1.000 1.000" << endl;
				mtlFile << "map_Kd texture_"      << m->texture_index << ".ppm" << endl << endl;
			}
		}
	}
	mtlFile.close();
}

} // Namespace lssr
