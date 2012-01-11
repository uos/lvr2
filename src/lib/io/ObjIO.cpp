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
 */

#include "ObjIO.hpp"
#include <climits>
#include "boost/tuple/tuple.hpp"
#include "../geometry/Vertex.hpp"

namespace lssr
{
using namespace std; // Bitte vergebt mir....
// Meinst du wirklich, dass ich dir so etwas durchgehen lassen kann?


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
