/*
 * PolygonMesh.tcc
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

#include "PolygonMesh.hpp"

namespace lvr {

template<typename VertexT, typename NormalT>
PolygonMesh<VertexT, NormalT>::PolygonMesh() {
	// TODO Auto-generated constructor stub

}

template<typename VertexT, typename NormalT>
PolygonMesh<VertexT, NormalT>::~PolygonMesh() {
	// TODO Auto-generated destructor stub
}

template<typename VertexT, typename NormalT>
void PolygonMesh<VertexT, NormalT>::addPolyRegion(PolyRegion polyregion) {
	m_regions.push_back(polyregion);
}

template<typename VertexT, typename NormalT>
void PolygonMesh<VertexT, NormalT>::addPolyRegions(std::vector<PolyRegion> polyregions) {
	// das gleiche wie addPolyRegion nur mit mehreren Regions zur gleichen Zeit - erstmal unwichtig
	std::vector<PolyRegion>::iterator it;
	for(it = polyregions.begin() ; it != polyregions.end() ; ++it)
	{
		m_regions.push_back((*it));
	}
}


template<typename VertexT, typename NormalT>
std::vector<PolyRegion> PolygonMesh<VertexT, NormalT>::getPolyRegions()
{
	return m_regions;
}

} /* namespace lvr */
