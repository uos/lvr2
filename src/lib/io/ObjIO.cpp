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
 */

#include "ObjIO.hpp"

namespace lssr
{

void ObjIO::save(Model model, string filename)
{

	m_model = model;

    ofstream out(filename.c_str());
    ofstream mtlFile("textures.mtl");

    if(out.good())
    {
    	if(m_textureIndices != 0)
    		out<<"mtllib textures.mtl"<<endl<<endl;

        for(size_t i = 0; i < m_vertexCount; i++)
        {
            unsigned int index = i * 3;
            out << "v " << m_vertices[index] << " "
                        << m_vertices[index + 1] << " "
                        << m_vertices[index + 2] << " " << endl;
        }

        out<<endl;
        for(size_t i = 0; i < m_vertexCount; i++)
        {
        	unsigned int index = i * 3;
            out << "vn " << m_normals[index] << " "
                         << m_normals[index + 1] << " "
                         << m_normals[index + 2] << " " << endl;
        }

        out<<endl;
        if(m_textureIndices != 0)
        {
        	for(size_t i = 0; i < m_vertexCount; i++)
        	{
        		unsigned int index = i * 3;
        		out << "vt " << m_textureCoords[index] << " "
        				<< m_textureCoords[index + 1] << " "
        				<< m_textureCoords[index + 2] << " " << endl;
        	}
        }

//        // write material file
//        if(m_textureIndices != 0)
//        {
//        	if(mtlFile.good())
//        	{
//        		mtlFile<<"newmtl texture_"<< UINT_MAX <<endl;
//        		mtlFile<<"Kd 0.000 1.000 1.000"<<endl;
//        		mtlFile<<"Ka 0.000 1.000 0.000"<<endl<<endl;
//
//        		for (size_t i = 0; i<this->m_textureCount; i++)
//        		{
//        			mtlFile<<"newmtl texture_"<<m_textures[i]<<endl;
//        			mtlFile<<"Ka 1.000 1.000 1.000"<<endl;
//        			mtlFile<<"Kd 1.000 1.000 1.000"<<endl;
//        			mtlFile<<"map_Kd texture_"<<m_textures[i]<<".ppm"<<endl<<endl;
//        		}
//        		mtlFile.close();
//
//        	}
//        }
//        int oldTextureIndex=-1, facesUsed=0, j=0, pommes=0;
//        //unsigned int counter = UINT_MAX-1;
//        for(size_t i = 0; i < m_faceCount; i++)
//        {
//        	unsigned int index = 3 * i;
//        	// Calculate the buffer positions for the three
//        	// triangle vertices (each vertex has 3 coordinates)
//        	unsigned int v1 = m_indices[index + 0];
//        	unsigned int v2 = m_indices[index + 1];
//        	unsigned int v3 = m_indices[index + 2];
//
//        	if(m_textureIndices != 0)
//        	{
//
//        		if(/*j >= m_nRegions &&*/ oldTextureIndex != m_textureIndices[index] && m_textureIndices[index] != UINT_MAX)
//        		{
//        			out << endl << "usemtl texture_" << m_textureIndices[index] << endl;
//        		}
//
//        		if(j < m_nRegions && facesUsed == pommes && m_textureIndices[index] == UINT_MAX)
//        		{
//        			mtlFile.open("textures.mtl", ios::app);
//        			mtlFile << "newmtl color_" << j <<endl;
//
//        			mtlFile << "Kd " << (m_colors[v1*3+0])/255.0f << " " << (m_colors[v1*3+1])/255.0f << " "
//        					<< (m_colors[v1*3+2])/255.0f << endl;
//
//        			mtlFile << "Ka " << (m_colors[v1*3+0])/255.0f << " " << (m_colors[v1*3+1])/255.0f << " "
//        					<< (m_colors[v1*3+2])/255.0f << endl << endl;
//
//        			out << endl;
//        			out << endl << "usemtl color_" << j << endl;
//        			pommes += m_regionSizeBuffer[j];
//        			j++;
//        			mtlFile.close();
//        			oldTextureIndex = -1;
//        		}
//
//        		oldTextureIndex = m_textureIndices[index];
//        		out << "f " << v1 + 1 << "/" << v1 + 1 << "/" << v1 + 1 << " "
//        				<< v2 + 1 << "/" << v2 + 1 << "/" << v2 + 1 << " "
//        				<< v3 + 1 << "/" << v3 + 1 << "/" << v3 + 1 << endl;
//
//
//        	} else /* If no texture coordinates given write dummy info.*/
//        	{
//        		out << "f " << v1 + 1 << "//" << v1 + 1 << " "
//        				<< v2 + 1 << "//" << v2 + 1 << " "
//        				<< v3 + 1 << "//" << v3 + 1 << endl;
//        	}
//        	facesUsed++;
//        }

        out.close();

    }
    else
    {
    	cerr << "no good. file! \n";
    }
}

}
