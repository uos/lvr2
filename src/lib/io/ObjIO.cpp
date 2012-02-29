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
#include <string.h>

namespace lssr
{
  using namespace std; // Bitte vergebt mir....
  // Meinst du wirklich, dass ich dir so etwas durchgehen lassen kann?

  ModelPtr ObjIO::read( string filename ) // TODO: Format correctly
  {
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
    cout << timestamp << "Number of vertices: "            << numVertices      << endl;
    cout << timestamp << "Number of vertex normals: "      << numVertexNormals << endl;
    cout << timestamp << "Number of vertex colors: "       << numVertexColors  << endl;
    cout << timestamp << "Number of faces: "               << numFaces         << endl;
    cout << timestamp << "Number of texture coordinates: " << numTextures      << endl;
    cout << timestamp << "Number of spheres: "             << numSpheres       << endl;
    cout << timestamp << "Number of planes: "              << numPlanes        << endl;
    cout << timestamp << "Number of point lights: "        << numPointLights   << endl;
    cout << timestamp << "Number of disc lights: "         << numDiscLights    << endl;
    cout << timestamp << "Number of quad lights: "         << numQuadLights    << endl;
    cout << timestamp << "Number of materials: "           << numMaterials     << endl;
    if(objData->camera != NULL)
      {
	cout << timestamp << "Found a camera" << endl;
      }
    cout << endl;

    // Buffers
    floatArr vertices;
    floatArr vertexNormals;
    ucharArr vertexColors;
    uintArr faceIndices;
    uintArr textureIndexBuffer;
    floatArr textureCoordBuffer;
    ucharArr faceColorBuffer;

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
	textureIndexBuffer = uintArr( new unsigned int[ numFaces * 3 ] );
      }
    if( numTextures )
      {
	textureCoordBuffer = floatArr( new float[ numTextures * 3 ] );
      }
    if( numMaterials )
      {
	faceColorBuffer = ucharArr( new uchar[ numMaterials * 3 ] );
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
	
	textureIndexBuffer[ i * 3 ]     = o->material_index;
	textureIndexBuffer[ i * 3 + 1 ] = o->material_index;
	textureIndexBuffer[ i * 3 + 2 ] = o->material_index;
      }

    // texture coordinates
    for(int i = 0; i < numTextures; ++i)
      {
	obj_vector *o = objData->textureList[i];
	textureCoordBuffer[ i * 3 ]     = o->e[ 0 ];
	textureCoordBuffer[ i * 3 + 1 ] = o->e[ 1 ];
	textureCoordBuffer[ i * 3 + 2 ] = o->e[ 2 ];
      }

    // face colors
    for(int i = 0; i < numMaterials; ++i)
      {
	obj_material *o = objData->materialList[i];
	faceColorBuffer[ i * 3 ]     = o->amb[0] * 255;
	faceColorBuffer[ i * 3 + 1]  = o->amb[1] * 255;
	faceColorBuffer[ i * 3 + 2 ] = o->amb[2] * 255;
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
	mesh->setFaceTextureIndexArray(        textureIndexBuffer, numFaces );
	mesh->setVertexTextureCoordinateArray( textureCoordBuffer, numTextures );
	mesh->setFaceColorArray(               faceColorBuffer,    numMaterials );
      }

    ModelPtr m( new Model( mesh ) );
    m_model = m;
    return m;
  };

  void ObjIO::save( string filename )
  {
    typedef Vertex<uchar> ObjColor;

    size_t lenVertices;
    size_t lenNormals;
    size_t lenColors;
    size_t lenFaces;
    size_t lenTextureCoordinates;
    size_t lenFaceIndices;
    size_t lenFaceColors;
    coord3fArr vertices           = m_model->m_mesh->getIndexedVertexArray( lenVertices );
    coord3fArr normals            = m_model->m_mesh->getIndexedVertexNormalArray( lenNormals );
    coord3fArr textureCoordinates = m_model->m_mesh->getIndexedVertexTextureCoordinateArray( lenTextureCoordinates );
    uintArr    faceIndices        = m_model->m_mesh->getFaceArray( lenFaces );
    uintArr    faceTextureIndices = m_model->m_mesh->getFaceTextureIndexArray( lenFaceIndices );
    ucharArr   faceColors         = m_model->m_mesh->getFaceColorArray( lenFaceColors );

    std::map<ObjColor, unsigned int> colorMap;


    std::set<unsigned int> textureIndexSet;
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

        unsigned int prevTextureIndex = UINT_MAX;
        out << endl << endl << "##  Beginning of faces.\n";
        // format of a face: f v/vt/vn
        for( size_t i = 0; i < lenFaces; ++i )
	  {
            if( lenFaceIndices > 0 && faceTextureIndices[ i * 3 ] != UINT_MAX )
	      {
                if( faceTextureIndices[ i * 3 ] != prevTextureIndex )
		  {
                    out << "usemtl texture_"<<faceTextureIndices[ i * 3 ]<<endl;
                    prevTextureIndex = faceTextureIndices[ i * 3 ];
                    textureIndexSet.insert(faceTextureIndices[ i * 3 ]);
		  }
	      }
            else if( lenFaceColors > 0 )
	      {
                ObjColor color(
			       faceColors[ i * 3 ],
			       faceColors[ i * 3 + 1 ],
			       faceColors[ i * 3 + 2 ] );
                pair<std::map<Vertex<uchar>, unsigned int>::iterator, bool> pommes 
		  = colorMap.insert( make_pair<ObjColor, unsigned int>( color, i ) );
                if( pommes.second == false )
		  {
                    out << "usemtl color_" << colorMap[color] << endl;
		  }
                else
		  {
                    out << "usemtl color_" << i << endl;
		  }

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
        if( textureIndexSet.size() > 0 )
	  {
            std::set<unsigned int>::iterator index;
            for ( index = textureIndexSet.begin(); index != textureIndexSet.end(); index++ )
	      {
                mtlFile << "newmtl texture_"      << *index << endl;
                mtlFile << "Ka 1.000 1.000 1.000" << endl;
                mtlFile << "Kd 1.000 1.000 1.000" << endl;
                mtlFile << "map_Kd texture_"      << *index << ".ppm" << endl << endl;
	      }
	  }
        if( colorMap.size() > 0 )
	  {
            std::map<ObjColor, unsigned int>::iterator mapIter = colorMap.begin();
            for( ; mapIter != colorMap.end(); mapIter++ )
	      {
                mtlFile << "newmtl color_" << mapIter->second << endl;
                mtlFile << "Ka " 
			<< mapIter->first[0] / 255.0f << " " 
			<< mapIter->first[1] / 255.0f << " "
			<< mapIter->first[2] / 255.0f << endl;
                mtlFile << "Kd " 
			<< mapIter->first[0] / 255.0f << " "
			<< mapIter->first[1] / 255.0f << " "
			<< mapIter->first[2] / 255.0f << endl;
	      }
	  }
      }
    mtlFile.close();
  }

} // Namespace lssr
