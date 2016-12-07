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

	if(r < this->m_regions->size())
	{
		Region<VertexT, NormalT>* region = this->m_regions->at(r);
		if(region)
		{
			region->calcArea();
			pf.index = r;
			pf.area = region->getArea();
			pf.nx = region->m_normal.x;
			pf.ny = region->m_normal.y;
			pf.nz = region->m_normal.z;

			BoundingBox<VertexT> bb = region->getBoundingBox();
			VertexT centroid = bb.getCentroid();
			pf.cx = centroid.x;
			pf.cy = centroid.y;
			pf.cz = centroid.z;
			pf.w = bb.getXSize();
			pf.h = bb.getYSize();
			pf.d = bb.getZSize();

			NormalT n_ceil(0.0, 1.0, 0.0);
			NormalT n_floor(0.0, -1.0, 0.0);
			float radius = sqrt(pf.nx * pf.nx + pf.nz * pf.nz);

			pf.orientation = UNKNOWN;
		    if(n_ceil * region->m_normal > 0.98 || n_floor * region->m_normal > 0.98)
		    {
		    	pf.orientation = HORIZONTAL;
		    }
		    else if(radius > 0.95)
		    {
		    	pf.orientation = VERTICAL;
		    }

		    this->m_features.push_back(pf);

//		    std::cout << "Index: " << pf.index << std::endl;
//		    std::cout << "Centroid: " << pf.cx << " " << pf.cy << " " << pf.cz << std::endl;
//		    std::cout << "BBOX: " << pf.w << " " << pf.h << " " << pf.d << std::endl;
//		    std::cout << "Normal: " << pf.nx << " " << pf.ny << " " << pf.nz << std::endl;
//		    std::cout << "Area: " << pf.area << std::endl;
//		    std::cout << "Orientation: " << pf.orientation << std::endl;

		}
	}
	else
	{
		cout << timestamp << " Furniture Classifier: Region number out of bounds." << endl;
	}
}

} /* namespace lvr */
