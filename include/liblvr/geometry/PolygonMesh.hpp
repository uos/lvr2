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

	/**
	 * @brief standard constructor
	 */
	PolygonMesh();

	/**
	 * @brief constructor
	 *
	 * @param first_regions the "first" Polygonregions of this Mesh, first means if you do not want to start with an empty Mesh
	 */
	PolygonMesh(std::vector<PolyRegion> first_regions);


	/**
	 * @brief destructor
	 */
	virtual ~PolygonMesh();


	/**
	 * @brief Add one new PolygonRegion, it will be pushed into the container (std::vector)
	 */
	void addPolyRegion(PolyRegion polyregion);


	/**
	 * @brief Add few new PolygonRegion, it will be pushed into the container (std::vector)
	 */
	void addPolyRegions(std::vector<PolyRegion> polyregions);


private:
	// container for all PolygonRegions in this Mesh
	std::vector<PolyRegion> m_meshes;
};

} /* namespace lvr */
#endif /* POLYGONMESH_HPP_ */
