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
 * ColorGradientPlaneClassifier.cpp
 *
 *  Created on: 11.04.2012
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename VertexT, typename NormalT>
ColorGradientPlaneClassifier<VertexT, NormalT>::ColorGradientPlaneClassifier(vector<Region<VertexT, NormalT>* >* region, GradientType t)
	: RegionClassifier<VertexT, NormalT>(region)
{
	m_colorMap = new ColorMap(256);
	m_gradientType = t;
}

template<typename VertexT, typename NormalT>
uchar ColorGradientPlaneClassifier<VertexT, NormalT>::r(int i)
{
	uchar* c = getColor(i);
	uchar tmp = c[0];
	delete[] c;
	return tmp;
}

template<typename VertexT, typename NormalT>
uchar ColorGradientPlaneClassifier<VertexT, NormalT>::g(int i)
{
	uchar* c = getColor(i);
	uchar tmp = c[1];
	delete[] c;
	return tmp;
}

template<typename VertexT, typename NormalT>
uchar ColorGradientPlaneClassifier<VertexT, NormalT>::b(int i)
{
	uchar* c = getColor(i);
	uchar tmp = c[2];
	delete[] c;
	return tmp;
}



template<typename VertexT, typename NormalT>
uchar* ColorGradientPlaneClassifier<VertexT, NormalT>::getColor(int i)
{
	uchar* c = new uchar[3];
	c[0] = 0;
	c[1] = 200;
	c[2] = 0;

	Region<VertexT, NormalT>* r = 0;
	if(i < this->m_regions->size())
	{
		r = this->m_regions->at(i);
	}

	if(r)
	{
		float fc[3];
		m_colorMap->getColor(fc, i, m_gradientType);
		for(int i = 0; i < 3; i++)
		{
			c[i] = (uchar)(fc[i] * 255);
		}
	}

	return c;
}

} // namespace lssr
