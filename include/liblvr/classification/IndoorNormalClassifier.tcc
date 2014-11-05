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
 * IndoorNormalClassifier.tcc
 *
 *  Created on: 12.04.2012
 *      Author: Thomas Wiemann
 */

namespace lvr
{

template<typename VertexT, typename NormalT>
uchar* IndoorNormalClassifier<VertexT, NormalT>::getColor(int index)
{
	float fc[3];
	uchar* c = new uchar[3];

	RegionLabel label = classifyRegion(index);
	switch(label)
	{
	case Ceiling:
		Colors::getColor(fc, BLUE);
		break;
	case Floor:
		Colors::getColor(fc, RED);
		break;
	case Wall:
		Colors::getColor(fc, GREEN);
		break;
	default:
		Colors::getColor(fc, LIGHTGREY);
		break;
	}

	for(int i = 0; i < 3; i++)
	{
		c[i] = (uchar)(fc[i] * 255);
	}

	return c;
}

template<typename VertexT, typename NormalT>
uchar IndoorNormalClassifier<VertexT, NormalT>::r(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[0];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
uchar IndoorNormalClassifier<VertexT, NormalT>::g(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[1];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
uchar IndoorNormalClassifier<VertexT, NormalT>::b(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[2];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
RegionLabel IndoorNormalClassifier<VertexT, NormalT>::classifyRegion(int index)
{

	if((unsigned int) index < this->m_regions->size())
	{
		NormalT n_ceil(0.0, 1.0, 0.0);
		NormalT n_floor(0.0, -1.0, 0.0);

		// Get region and normal
		Region<VertexT, NormalT>* region = this->m_regions->at(index);
		NormalT normal = region->m_normal;

		if(region->size() >= this->m_minSize)
		{
			// Check if ceiling or floor
			if(n_ceil 	* normal > 0.98) return Ceiling;
			if(n_floor 	* normal > 0.98) return Floor;

			// Check for walls
			float radius = sqrt(normal.x * normal.x + normal.z * normal.z);
			if(radius > 0.95) return Wall;
		}
	}

	return Unknown;
}

template<typename VertexT, typename NormalT>
void IndoorNormalClassifier<VertexT, NormalT>::createRegionBuffer(
				int region_id,
				map<VertexT, int> &vertex_map,
				vector<int> &indices,
				vector<float> &vertices,
				vector<float> &normals,
				vector<uint> &colors
				)
{
	//int index_counter = 0;
	size_t	vertex_position = 0;

	Region<VertexT, NormalT>* region = this->m_regions->at(region_id);

	// get the color
	uchar* color = getColor(region_id);
	uchar red   = color[0];
	uchar green = color[1];
	uchar blue  = color[2];

	// Check if region is a planar cluster
	VertexT current;
	NormalT normal;
	for(unsigned int a = 0; a < region->m_faces.size(); a++)
	{
		for(int d = 0; d < 3; d++)
		{
			HalfEdgeFace<VertexT, NormalT>* f = region->m_faces[a];

			current = (*f)(d)->m_position;
			normal =  (*f)(d)->m_normal;

			if(vertex_map.find(current) != vertex_map.end())
			{
				// Use already present vertex
				vertex_position = vertex_map[current];
			}
			else
			{
				// Create new index
				vertex_position = vertices.size() / 3;

				// Insert new vertex to vertex map and save relevant information
				vertex_map.insert(pair<VertexT, int>(current, vertex_position));

				vertices.push_back(current.x);
				vertices.push_back(current.y);
				vertices.push_back(current.z);

				normals.push_back(normal.x);
				normals.push_back(normal.y);
				normals.push_back(normal.z);

				colors.push_back(red);
				colors.push_back(green);
				colors.push_back(blue);
			}

			indices.push_back(vertex_position);
		}
	}
}

template<typename VertexT, typename NormalT>
void IndoorNormalClassifier<VertexT, NormalT>::writeBuffers(
		ofstream &out,
		RegionLabel label,
		vector<int> &indices,
		vector<float> &vertices,
		vector<float> &normals,
		vector<uint> &colors)
{
	static int c;
	string str_label = "Unclassified";
	switch(label)
	{
	case Ceiling 	: str_label = "Ceiling"; 	break;
	case Floor		: str_label = "Floor";		break;
	case Wall		: str_label = "Wall";		break;
	case Unknown	: return;
	}

	out << str_label << c << std::endl;
	c++;
	out << indices.size() / 3 << " " << vertices.size() / 3 << std::endl;
	for(size_t c = 0; c < indices.size() / 3; c++)
	{
		size_t buff_pos = 3 * c;
		out << indices[buff_pos] << " " << indices[buff_pos + 1] << " " << indices[buff_pos + 2] << std::endl;
	}

	for(size_t c = 0; c < vertices.size() / 3; c++)
	{
		size_t buff_pos = 3 * c;
		out << vertices[buff_pos] << " " << vertices[buff_pos + 1] << " " << vertices[buff_pos + 2] << " ";
		out << normals [buff_pos] << " " << normals [buff_pos + 1] << " " <<  normals[buff_pos + 2] << " ";
		out << colors  [buff_pos] << " " << colors  [buff_pos + 1] << " " <<   colors[buff_pos + 2] << std::endl;
	}
}

template<typename VertexT, typename NormalT>
void IndoorNormalClassifier<VertexT, NormalT>::writeMetaInfo()
{
	ofstream out("clusters.clu");

	if(!out.good())
	{
		std::cout << "Unable to open cluster file." << std::endl;
		return;
	}

	// Save all small clusters in a seperate mesh for
	// performance reasons
	vector<int> 	uc_indices;
	vector<float> 	uc_vertices;
	vector<uint> 	uc_colors;
	vector<float> 	uc_normals;
	map<VertexT, int> uc_vertex_map;

	for(unsigned int i = 0; i < this->m_regions->size(); i++)
	{
		// Get current region and label
		Region<VertexT, NormalT>* region = this->m_regions->at(i);

		if(region->m_faces.size() > 20)
		{

			RegionLabel label = classifyRegion(i);

			// Buffer vectors
			vector<int> 	indices;
			vector<float> 	vertices;
			vector<float> 	normals;
			vector<uint>	colors;

			map<VertexT, int> vertex_map;

			createRegionBuffer(i, vertex_map, indices, vertices, normals, colors);
			writeBuffers(out, label, indices, vertices, normals, colors);

		}
		else
		{
			createRegionBuffer(i, uc_vertex_map, uc_indices, uc_vertices, uc_normals, uc_colors);
		}

	}

	// Write all unclassified clusters
	writeBuffers(out, Unknown, uc_indices, uc_vertices, uc_normals, uc_colors);
	out.close();
}

} /* namespace lvr */
