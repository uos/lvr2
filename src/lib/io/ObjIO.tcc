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
 *  Created on: 10.03.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType, typename IndexType>
ObjIO<CoordType, IndexType>::ObjIO()
{
    m_normals     			= 0;
    m_vertices    			= 0;
    m_indices      			= 0;
    m_vertexCount  			= 0;
    m_faceCount    			= 0;
    m_textureCoords			= 0;
	m_textureIndices		= 0;
	m_textures				= 0;
	m_textureCount			= 0;
	m_textureIndicesCount	= 0;
	m_textureCoordsCount    = 0;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setVertexArray(CoordType* vertices, size_t c)
{
    m_vertices = vertices;
    m_vertexCount = c;
}


template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setNormalArray(CoordType* normals, size_t c)
{
    m_normals = normals;
    m_vertexCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setIndexArray(IndexType* indices, size_t c)
{
    m_indices = indices;
    m_faceCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setTextureCoords(CoordType* coords, size_t c)
{
    m_textureCoords = coords;
    m_textureCoordsCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setColors(uchar* coords, size_t c)
{
    m_colors = coords;
    m_colorCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setTextureIndices(IndexType* indices, size_t c)
{
    m_textureIndices = indices;
    m_textureIndicesCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::setTextures(IndexType* textures, size_t c)
{
    m_textures = textures;
    m_textureCount = c;
}

template<typename CoordType, typename IndexType>
void ObjIO<CoordType, IndexType>::write(string filename)
{
    ofstream out(filename.c_str());
    ofstream mtlFile("textures.mtl");

    if(out.good())
    {
    	if(m_textureIndices != 0)
    		out<<"mtllib textures.mtl"<<endl<<endl;

        for(size_t i = 0; i < m_vertexCount; i++)
        {
            IndexType index = i * 3;
            out << "v " << m_vertices[index] << " "
                        << m_vertices[index + 1] << " "
                        << m_vertices[index + 2] << " " << endl;
        }

        out<<endl;
        for(size_t i = 0; i < m_vertexCount; i++)
        {
            IndexType index = i * 3;
            out << "vn " << m_normals[index] << " "
                         << m_normals[index + 1] << " "
                         << m_normals[index + 2] << " " << endl;
        }

        out<<endl;
        if(m_textureIndices != 0)
        {
        	for(size_t i = 0; i < m_vertexCount; i++)
        	{
        		IndexType index = i * 3;
        		out << "vt " << m_textureCoords[index] << " "
        				<< m_textureCoords[index + 1] << " "
        				<< m_textureCoords[index + 2] << " " << endl;
        	}
        }

        // write material file
        if(m_textureIndices != 0)
        {
        	if(mtlFile.good())
        	{
        		mtlFile<<"newmtl texture_"<< UINT_MAX <<endl;
        		mtlFile<<"Kd 0.000 1.000 1.000"<<endl;
        		mtlFile<<"Ka 0.000 1.000 0.000"<<endl<<endl;

        		for (size_t i = 0; i<this->m_textureCount; i++)
        		{
        			mtlFile<<"newmtl texture_"<<m_textures[i]<<endl;
        			mtlFile<<"Ka 1.000 1.000 1.000"<<endl;
        			mtlFile<<"Kd 1.000 1.000 1.000"<<endl;
        			mtlFile<<"map_Kd texture_"<<m_textures[i]<<".ppm"<<endl<<endl;
        		}
        		mtlFile.close();

        	}
        }
        //unsigned int counter = UINT_MAX-1;
        for(size_t i = 0; i < m_faceCount; i++)
        {
            IndexType index = 3 * i;
            // Calculate the buffer positions for the three
            // triangle vertices (each vertex has 3 coordinates)
            IndexType v1 = m_indices[index + 0];
            IndexType v2 = m_indices[index + 1];
            IndexType v3 = m_indices[index + 2];

            if(m_textureIndices != 0)
            {
            	int oldTextureIndex = -1;
            	if(oldTextureIndex != m_textureIndices[index])
            	{
            		out <<endl<< "usemtl texture_" <<m_textureIndices[index]<< endl;
            	}
 /*           	if(m_textureIndices[index] == UINT_MAX)
            	{
            		mtlFile.open("textures.mtl", ios::app);
            		mtlFile <<"newmtl color_"<< counter <<endl;

            		mtlFile << "Kd " << m_colors[v1*3+0] << " " << m_colors[v1*3+1] << " "
            				<< m_colors[v1*3+2] <<endl;

            		mtlFile << "Ka " << m_colors[v1*3+0] << " " << m_colors[v1*3+1] << " "
            				<< m_colors[v1*3+2] <<endl;

            		out <<endl<< "usemtl color_" <<counter<< endl;
            		out << "f " << v1 + 1 << "//" << v1 + 1 << " "
            				    << v2 + 1 << "//" << v2 + 1 << " "
            				    << v3 + 1 << "//" << v3 + 1 << endl << endl;
            		oldTextureIndex = counter;
            		counter--;
            		mtlFile.close();
            	} else { */
            		oldTextureIndex = m_textureIndices[index];
            		out << "f " << v1 + 1 << "/" << v1 + 1 << "/" << v1 + 1 << " "
            				<< v2 + 1 << "/" << v2 + 1 << "/" << v2 + 1 << " "
            				<< v3 + 1 << "/" << v3 + 1 << "/" << v3 + 1 << endl;
            /*	} */
            }
            else /* If no texture coordinates given write dummy info.*/
            {
            	out << "f " << v1 + 1 << "//" << v1 + 1 << " "
            			<< v2 + 1 << "//" << v2 + 1 << " "
            			<< v3 + 1 << "//" << v3 + 1 << endl;
            }
        }

        out.close();

    }
    else
    {
		cerr << "no good. file! \n";
    }
}

}
