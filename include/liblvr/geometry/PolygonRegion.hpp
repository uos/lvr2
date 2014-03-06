/*
 * Polygonregion.hpp
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

#ifndef POLYGONREGION_HPP_
#define POLYGONREGION_HPP_

#include "Vertex.hpp"
#include "Normal.hpp"
#include "Polygon.hpp"
#include <vector>

namespace lvr {

template<typename VertexT, typename NormalT>
class PolygonRegion {
public:
	typedef Polygon<VertexT, NormalT> Poly;

	/**
	 * @brief standard constructor
	 */
	PolygonRegion();

	/**
	 * @brief constructor
	 *
	 * @param new_polygons
	 */
	PolygonRegion(std::vector<Poly> new_polygons, std::string label, NormalT normal);

	~PolygonRegion();

	void setPolygons(std::vector<Poly> new_polygons);

	std::vector<Poly> getPolygons();


private:
	// List of all Polygons, the first one is the outer Polygon of this Region
	std::vector<Poly> m_polygons;

	// label of this Region
	std::string m_label;

	// Normal of this Region
	NormalT m_normal;

};

} /* namespace lvr */

#include "PolygonRegion.tcc"
#endif /* POLYGONREGION_HPP_ */
