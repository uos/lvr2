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

}

template<typename VertexT, typename NormalT>
void PolygonMesh<VertexT, NormalT>::addPolyRegions(std::vector<PolyRegion> polyregions) {
	// das gleiche wie addPolyRegion nur mit mehreren Regions zur gleichen Zeit - erstmal unwichtig
}

} /* namespace lvr */
