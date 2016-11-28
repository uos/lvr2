/*
 * FurnitureFeatureClassifier.tcc
 *
 *  Created on: Nov 25, 2016
 *      Author: robot
 */

#include "FurnitureFeatureClassifier.hpp"

namespace lvr {

template<typename VertexT, typename NormalT>
FurnitureFeatureClassifier<VertexT, NormalT>::FurnitureFeatureClassifier(vector<Region<VertexT, NormalT>* >* region)
	: RegionClassifier<VertexT, NormalT>(region)
{

}

template<typename VertexT, typename NormalT>
FurnitureFeatureClassifier<VertexT, NormalT>::~FurnitureFeatureClassifier()
{

}

template<typename VertexT, typename NormalT>
void FurnitureFeatureClassifier<VertexT, NormalT>::classifyRegion(int Region)
{
	Region* region = m_regions[i];
}

} /* namespace lvr */
