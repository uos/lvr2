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
	void setPolygons(std::vector<Polygon<VertexT, NormalT> > new_polygons, std::string label, NormalT normal);

	/**
	 * @brief get all the Polygons
	 *
	 * @return all Polygons of this Region
	 */
	std::vector<Polygon<VertexT, NormalT> > getPolygons();

	/**
	 * @brief Returns the normal of this region
	 *
	 * @return the normal of this region
	 */
	NormalT getNormal();

	/**
	 * @brief Returns the number of points in this region
	 *
	 * @return the umber of points in this region
	 */
	size_t getSize();

	/**
	 * @brief Return the label of this region
	 *
	 * @return the label of this region
	 */
	std::string getLabel();

	/**
	 * @brief Return the lower left point of its BoundingBox
	 *
	 * @return lower left point of its BoundingBox
	 */
	VertexT getBoundMin();

	/**
	 * @brief Return the upper right point of its BoundingBox
	 *
	 * @return lower upper right of its BoundingBox
	 */
	VertexT getBoundMax();

	/**
	 * @brief Setter for the label
	 *
	 * @param new_label the new label of this region
	 */
	void setLabel(std::string new_label);


	/**
	 * @brief Setter for the normal
	 *
	 * @param new normal for this region
	 */
	void setNormal(NormalT new_normal);

	/**
	 * @brief Returns one polygon of this region (the first polygon)
	 */
	Polygon<VertexT, NormalT> getPolygon();


private:
	/**
	 * @brief Calculates the axis aligned BoundingBox for this Polygonregion
	 */
	void calcBoundingBox();

	// List of all Polygons, the first one is the outer Polygon of this Region
	std::vector<Polygon<VertexT, NormalT>> m_polygons;

	// label of this Region
	std::string m_label;

	// Normal of this Region
	NormalT m_normal;

	// Edges of the BoundingBox
	VertexT m_bound_min;
	VertexT m_bound_max;

};

} /* namespace lvr */

#include "PolygonRegion.tcc"
#endif /* POLYGONREGION_HPP_ */
