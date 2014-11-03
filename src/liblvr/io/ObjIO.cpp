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

#include "io/ObjIO.hpp"
#include "objLoader.h"

#include <climits>
#include <iostream>
#include <fstream>
#include <string.h>
#include <locale.h>
#include <sstream>

#include <boost/filesystem.hpp>
#include "boost/tuple/tuple.hpp"

#include "io/PLYIO.hpp"
#include "io/Timestamp.hpp"
#include "geometry/Vertex.hpp"
#include "display/GlTexture.hpp"
#include "display/TextureFactory.hpp"


namespace lvr
{
using namespace std; // Bitte vergebt mir....
// Meinst du wirklich, dass ich dir so etwas durchgehen lassen kann?


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

	// Get path object
	boost::filesystem::path p(mtlname);
	p = p.remove_filename();

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
				m->r = (unsigned char)(r * 255);
				m->g = (unsigned char)(g * 255);
				m->b = (unsigned char)(b * 255);
			}
			else if(keyword == "map_Kd")
			{
				string texname;
				ss >> texname;

				// Add full path to texture file name
				boost::filesystem::path tex_file = p / texname;

				GlTexture* texture = TextureFactory::instance().getTexture(tex_file.string());
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
    // Get path from filename
    boost::filesystem::path p(filename);

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
			    // Get current path
			    p = p.remove_filename();

			    // Append .mtl file name
				string mtlfile;
				ss >> mtlfile;
				p = p / mtlfile;

				// Get path as string and parse mtl
				string mtl_path = p.string();
				parseMtlFile(matNames, materials, textures, mtl_path);
			}
		}

	}
	else
	{
		cout << timestamp << "ObjIO::read(): Unable to open file'" << filename << "'." << endl;
	}

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

	ModelPtr m(new Model(mesh));
	m_model = m;
	return m;
}

void ObjIO::save( string filename )
{
	typedef Vertex<unsigned char> ObjColor;

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

} // Namespace lvr
