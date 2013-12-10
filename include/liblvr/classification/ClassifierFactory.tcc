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
 * ClassifierFactory.cpp
 *
 *  Created on: 11.04.2012
 *      Author: Thomas Wiemann
 */



namespace lvr
{

template<typename VertexT, typename NormalT>
RegionClassifier<VertexT, NormalT>*  ClassifierFactory<VertexT, NormalT>::get(string name, HMesh* mesh)
{
	if(name == "PlaneJet")
	{
		return new ColorGradientPlaneClassifier<VertexT, NormalT>(&mesh->m_regions, JET);
	}
	else if(name == "PlaneHot")
	{
		return new ColorGradientPlaneClassifier<VertexT, NormalT>(&mesh->m_regions, HOT);
	}
	else if(name == "PlaneHSV")
	{
		return new ColorGradientPlaneClassifier<VertexT, NormalT>(&mesh->m_regions, HSV);
	}
	else if(name == "PlaneGrey")
	{
		return new ColorGradientPlaneClassifier<VertexT, NormalT>(&mesh->m_regions, GREY);
	}
	else if(name == "PlaneSimpsons")
	{
		return new ColorGradientPlaneClassifier<VertexT, NormalT>(&mesh->m_regions, SIMPSONS);
	}
	else if(name == "NormalClassifier")
	{
		return new NormalClassifier<VertexT, NormalT>(&mesh->m_regions);
	}
	else if(name == "IndoorNormals")
	{
		return new IndoorNormalClassifier<VertexT, NormalT>(&mesh->m_regions);
	}

	return 0;
}

} /* namespace lvr */
