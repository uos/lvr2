/*
 * Polygon.hpp
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

#ifndef POLYGON_HPP_
#define POLYGON_HPP_

#include "Vertex.hpp"
#include "Normal.hpp"
#include <vector>


namespace lvr {

template<typename VertexT, typename NormalT>
class Polygon {
public:
	/**
	 * @brief standard constructor
	 */
	Polygon();

	/**
	 * constructor
	 */
	Polygon(std::vector<VertexT> new_vertices);

	/**
	 * @brief destructor
	 */
	virtual ~Polygon();

	/**
	 * @brief Returns all vertices of this polygon
	 * @return all vertices of this polygon
	 */
	std::vector<VertexT> getVertices();

	/**
	 * @brief Sets the vertices of this polygon
	 */
	void setVertices(std::vector<VertexT>);

	/**
	 * @brief Returns the number of vertices in this polygon
	 * @return the number of vertices in this polygon
	 */
	size_t getSize();

private:
	// list of all vertices
	std::vector<VertexT> m_vertices;
};

} /* namespace lvr */

#include "Polygon.tcc"

#endif /* POLYGON_HPP_ */
