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
 * IndoorNormalClassifier.hpp
 *
 *  Created on: 12.04.2012
 *      Author: Thomas Wiemann
 */

#ifndef INDOORNORMALCLASSIFIER_HPP_
#define INDOORNORMALCLASSIFIER_HPP_

//#include "RegionClassifier.hpp"
#include "display/Color.hpp"

namespace lvr
{

enum RegionLabel
{
	vertical,
	horizontallower,
	horizontalupper,
	unknown
};

/**
 * @brief	Classifier for normal and area based cluster interpretation as
 * 			presented in @ref KI2011
 */
template<typename VertexT, typename NormalT>
class IndoorNormalClassifier : public RegionClassifier<VertexT, NormalT>
{
public:

	/**
	 * @brief Ctor
	 * @param region		A vector of planar clusters
	 */
	IndoorNormalClassifier(vector<Region<VertexT, NormalT>* >* region)
		: RegionClassifier<VertexT, NormalT>(region) {};

	/**
	 * @brief Dtor.
	 */
	virtual ~IndoorNormalClassifier() {};

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
	virtual string rl(int region);

	virtual void writeMetaInfo();

private:

	RegionLabel classifyRegion(int region);
	uchar* getColor(int region);
	string getLabel(RegionLabel label);

	void createRegionBuffer(
					int region_id,
					map<VertexT, int> &map,
					vector<int> &indices,
					vector<float> &vertices,
					vector<float> &normals,
					vector<uint> &colors,
					vector<string> &labels);

	void writeBuffers(
			ofstream &out,
			RegionLabel label,
			vector<int> &indices,
			vector<float> &vertices,
			vector<float> &normals,
			vector<uint> &colors,
			vector<string> &labels);


};

} /* namespace lvr */

#include "IndoorNormalClassifier.tcc"

#endif /* INDOORNORMALCLASSIFIER_HPP_ */
