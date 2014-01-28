/**
 * Copyright (C) 2011 Uni Osnabrück
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

/**
 * NormalClassifier.tcc
 *
 *  Created on: 01.11.2013
 *      Author: Simon Herkenhoff
 */

namespace lvr
{

template<typename VertexT, typename NormalT>
bool NormalClassifier<VertexT, NormalT>::generatesLabel()
{
	return true;
}

template<typename VertexT, typename NormalT>
string NormalClassifier<VertexT, NormalT>::getLabel(int index)
{
	Region<VertexT, NormalT>* region = this->m_regions->at(index);

	string label_str;

	if (region->hasLabel())
	{
		label_str = region->getLabel();
	}
	else
	{

		NormalLabel label_type = classifyRegion(index);

		switch(label_type)
		{
			case VerticalFace:
				label_str = "vertical";
				break;
			case HorizontalupperFace:
				label_str = "horizontalupper";
				break;
			case HorizontallowerFace:
				label_str = "horizontallower";
				break;
			default:
				label_str = "unknown";
				break;
		}

		region->setLabel(label_str);
	}

	return label_str;
}

template<typename VertexT, typename NormalT>
uchar* NormalClassifier<VertexT, NormalT>::getColor(int index)
{
	float fc[3];
	uchar* c = new uchar[3];

	NormalLabel label_type = classifyRegion(index);
	switch(label_type)
	{
		case VerticalFace:
			Colors::getColor(fc, BLUE);
			break;
		case HorizontalupperFace:
			Colors::getColor(fc, RED);
			break;
		case HorizontallowerFace:
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
uchar NormalClassifier<VertexT, NormalT>::r(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[0];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
uchar NormalClassifier<VertexT, NormalT>::g(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[1];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
uchar NormalClassifier<VertexT, NormalT>::b(int i)
{
	uchar* c = getColor(i);
	uchar ret = c[2];
	delete c;
	return ret;
}

template<typename VertexT, typename NormalT>
NormalLabel NormalClassifier<VertexT, NormalT>::classifyRegion(int index)
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
			if(n_ceil 	* normal > 0.98) return HorizontalupperFace;
			if(n_floor 	* normal > 0.98) return HorizontallowerFace;

			// Check for walls
			float radius = sqrt(normal.x * normal.x + normal.z * normal.z);
			if(radius > 0.95) return VerticalFace;
		}
	}

	return UnknownFace;
}

template<typename VertexT, typename NormalT>
void NormalClassifier<VertexT, NormalT>::createRegionBuffer(
				int region_id,
				map<VertexT, int> &vertex_map,
				vector<int> &indices,
				vector<float> &vertices,
				vector<float> &normals,
				vector<uint> &colors
				)
{
	//int index_counter = 0;
	int	vertex_position = 0;

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
void NormalClassifier<VertexT, NormalT>::writeMetaInfo()
{
	std::cout << timestamp << "METHOD NormalClassifier::writeMetaInfo() NOT YET IMPLEMENTED" << std::endl;
	return;
}


} /* namespace lvr */
