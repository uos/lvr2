/*
 * FurnitureFeatureClassifier.tcc
 *
 *  Created on: Nov 25, 2016
 *      Author: robot
 */

#include <lvr/classification/FurnitureFeatureClassifier.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/geometry/BoundingBox.hpp>

#include <iostream>
using std::cout;
using std::endl;

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
void FurnitureFeatureClassifier<VertexT, NormalT>::classifyRegion(int r)
{
	PlanarClusterFeature pf;

	if(r < m_regions.size())
	{
		Region* region = m_regions[r];
		if(region)
		{
			region->calcArea();
			pf.index = r;
			pf.area = region->getArea();
			pf.nx = region->m_normal.x;
			pf.ny = region->m_normal.y;
			pf.nz = region->m_normal.z;

			BoundingBox bb = region->getBoundingBox();
			pf.bbx = bb.getCentroid().x;
			pf.bby = bb.getCentroid().y;
			pf.bbz = bb.getCentroid().z;
			pf.bbw = bb.getXSize();
			pf.bbh = bb.getYSize();
			pf.bbd = bb.getZSize();

			NormalT n_ceil(0.0, 1.0, 0.0);
			NormalT n_floor(0.0, -1.0, 0.0);
			float radius = sqrt(pf.nx * pf.nx + pf.nz * pf.nz);

			pf.orientation = 0;
		    if(n_ceil * normal > 0.98 || n_floor * normal > 0.98)
		    {
		    	pf.orientation = 1;
		    }
		    else if(radius > 0.95)
		    {
		    	pf.orientation = 2;
		    }

		    m_features.push_back(pf);
		}
	}
	else
	{
		cout << timestamp << " Furniture Classifier: Region number out of bounds." << endl;
	}
}

} /* namespace lvr */
