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

namespace lssr
{


template<typename VertexT, typename NormalT>
uchar* IndoorNormalClassifier<VertexT, NormalT>::getColor(int i)
{
	float* fc;

	RegionLabel label = classifyRegion(i);
	switch(label)
	{
	case Ceiling:
		Colors::getColor(fc, GREEN);
		break;
	case Floor:
		Colors::getColor(fc, RED);
		break;
	case Wall:
		Colors::getColor(fc, BLUE);
		break;
	default:
		Colors::getColor(fc, LIGHTGREY);
	}

	uchar* c = new uchar[3];
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

	if(index < this->m_regions->size())
	{
		NormalT n_ceil(0.0, 1.0, 0.0);
		NormalT n_floor(0.0, -1.0, 0.0);

		// Get region and normal
		Region<VertexT, NormalT>* region = this->m_regions->at(index);
		NormalT normal = region->m_normal;

		// Only classify regions with a minimum of 10 faces
		if(region->size() > 10)
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



} /* namespace lssr */
