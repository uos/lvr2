/*
 * PolygonMesh.hpp
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

#ifndef POLYGONMESH_HPP_
#define POLYGONMESH_HPP_

#include "PolygonRegion.hpp"
#include <vector>

namespace lvr {

template<typename VertexT, typename NormalT>
class PolygonMesh {
public:
	typedef PolygonRegion<VertexT, NormalT> PolyRegion;
	PolygonMesh();

	PolygonMesh(std::vector<PolyRegion> first_regions);

	virtual ~PolygonMesh();

	void addPolyRegion(PolyRegion polyregion);

	void addPolyRegions(std::vector<PolyRegion> polyregions);


private:
	std::vector<PolyRegion> m_meshes;
};

} /* namespace lvr */
#endif /* POLYGONMESH_HPP_ */
