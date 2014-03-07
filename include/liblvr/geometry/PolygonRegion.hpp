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
	//typedef Polygon<VertexT, NormalT> Poly;

	/**
	 * @brief standard constructor
	 */
	PolygonRegion();


	/**
	 * @brief constructor
	 *
	 * @param new_polygons all Polygons of this Region (first in vector is the outer polygon)
	 * @param label        the label of this Region
	 * @param normal       the normal of this Region
	 */
	PolygonRegion(std::vector<Polygon<VertexT, NormalT>> new_polygons, std::string label, NormalT normal);


	/**
	 * @brief destructor
	 */
	~PolygonRegion();


	/**
	 * @brief If you have a Polygon, but its data is wrong or empty, use this and fill it
	 *
	 * @param new_polygons all Polygons of this new Region (first in vector is the outer polygon)
	 * @param label        the label of this new Region
	 * @param normal       the normal of this new Region
	 */
	void setPolygons(std::vector<Polygon<VertexT, NormalT>> new_polygons, std::string label, std::string normal);


	/**
	 * @brief get all the Polygons
	 *
	 * @return all Polygons of this Region
	 */
	std::vector<Polygon<VertexT, NormalT>> getPolygons();


private:
	// List of all Polygons, the first one is the outer Polygon of this Region
	std::vector<Polygon<VertexT, NormalT>> m_polygons;

	// label of this Region
	std::string m_label;

	// Normal of this Region
	NormalT m_normal;

};

} /* namespace lvr */

#include "PolygonRegion.tcc"
#endif /* POLYGONREGION_HPP_ */
