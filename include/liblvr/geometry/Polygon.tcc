/*
 * Polygon.tcc
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

namespace lvr {

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::Polygon() {
	// TODO Auto-generated constructor stub

}

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::Polygon(std::vector<VertexT> new_vertices) {
	m_vertices = new_vertices;
}

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::~Polygon() {
	m_vertices.clear();
}

template<typename VertexT, typename NormalT>
std::vector<VertexT> Polygon<VertexT, NormalT>::getVertices() {
	return m_vertices;
}

template<typename VertexT, typename NormalT>
size_t Polygon<VertexT, NormalT>::getSize() {
	return m_vertices.size();
}

} /* namespace lvr */
