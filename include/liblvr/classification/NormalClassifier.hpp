/**
 * Copyright (C) 2013 Uni Osnabrück
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
 * NormalClassifier.hpp
 *
 *  Created on: 01.10.2013
 *      Author: Simon Herkenhoff
 */

#ifndef NORMALCLASSIFIER_HPP_
#define NORMALCLASSIFIER_HPP_

#include "display/Color.hpp"

namespace lvr
{

enum NormalLabel
{
	VerticalFace=40,
	HorizontallowerFace,
	HorizontalupperFace,
	UnknownFace
};

/**
 * @brief Classifier for normal and area based cluster interpretation as
 * presented in @ref KI2011
 */
template<typename VertexT, typename NormalT>
class NormalClassifier : public RegionClassifier<VertexT, NormalT>
{
public:

	/**
	 * @brief Ctor
	 * @param region A vector of planar clusters
	 */
	NormalClassifier(vector<Region<VertexT, NormalT>* >* region)
		: RegionClassifier<VertexT, NormalT>(region) {};

	/**
	 * @brief Dtor.
	 */
	virtual ~NormalClassifier() {};

	/**
	 * @brief Returns the r component for the given region
	 */
	virtual uchar r(int region);
	/**
	 * @brief Returns the g component for the given region
	 */
	virtual uchar g(int region);
	/**
	 * @brief Returns the b component for the given region
	 */
	virtual uchar b(int region);

	/**
	 * @brief Returns the label as a string
	 */
	virtual string regiontostr(int region);

	virtual void writeMetaInfo();

	virtual void createBuffer();

private:

	NormalLabel classifyRegion(int region);
	uchar* getColor(int region);
	string getLabel(NormalLabel label);

	void createRegionBuffer(
		int region_id,
		map<VertexT, int> &map,
		vector<int> &indices,
		vector<float> &vertices,
		vector<float> &normals,
		vector<uint> &colors);

};

} /* namespace lvr */

#include "NormalClassifier.tcc"

#endif /* NORMALCLASSIFIER_HPP_ */
