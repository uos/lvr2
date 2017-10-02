/*
 * Polygon.tcc
 *
 *  Created on: 06.03.2014
 *      Author: Dominik Feldschnieders (dofeldsc@uos.de)
 */

namespace lvr {

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::Polygon() {
	// TODO Auto-generated constructor stub

}

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::Polygon(std::vector<VertexT> new_vertices) {
	this->m_vertices = new_vertices;
}

template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT>::~Polygon() {
	this->m_vertices.clear();
}

template<typename VertexT, typename NormalT>
std::vector<VertexT> Polygon<VertexT, NormalT>::getVertices() {
	return this->m_vertices;
}

template<typename VertexT, typename NormalT>
void Polygon<VertexT, NormalT>::setVertices(std::vector<VertexT> vertices) {
	this->m_vertices = vertices;
}

template<typename VertexT, typename NormalT>
size_t Polygon<VertexT, NormalT>::getSize() {
	return this->m_vertices.size();
}

} /* namespace lvr */
