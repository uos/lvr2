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
 * RegionClassifier.h
 *
 *  Created on: 11.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef REGIONCLASSIFIER_H_
#define REGIONCLASSIFIER_H

#include <vector>
#include "geometry/Region.hpp"
using std::vector;


namespace lvr
{

/**
 * @brief	Base class for cluster classification.
 */
template<typename VertexT, typename NormalT>
class RegionClassifier
{
public:

	/**
	 * @brief Constructs a classifier for the given set of clusters
	 * @param region	A vector containing the region to classify
	 */
	RegionClassifier(vector<Region<VertexT, NormalT>* >* region) : m_regions(region) {};

	/**
	 * @brief Dtor.
	 */
	virtual ~RegionClassifier() {};

	/**
	 * @brief Accesses the given region by index
	 *
	 * @param region	The region to access
	 * @return			The region or null of index out of range
	 */
	Region<VertexT, NormalT>* operator[](int i)
	{
		if(i <= m_regions->size())
			{
				return m_regions->at(i);
			}
			else
			{
				return 0;
			}
	}

	/**
	 * @brief Returns the r component for the given region
	 */
	virtual uchar r(int region) { return 0;  }

	/**
	 * @brief Returns the g component for the given region
	 */
	virtual uchar g(int region) { return 255;}

	/**
	 * @brief Returns the b component for the given region
	 */
	virtual uchar b(int region) { return 0;  }


	/**
	 * @brief  Overwrite this class to write information about the clusters
	 * 		   to a file
	 */
	virtual void writeMetaInfo() {};



protected:

	/// A pointer to a vector containing regions
	vector<Region<VertexT, NormalT>* >*  m_regions;
};

} /* namespace lvr */

#include "RegionClassifier.tcc"

#endif /* REGIONCLASSIFIER_H_ */
