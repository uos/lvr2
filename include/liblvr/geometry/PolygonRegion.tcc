/*
 * Polygonregion.tcc
 *
 *  Created on: 06.03.2014
 *      Author: dofeldsc
 */

namespace lvr {

template<typename VertexT, typename NormalT>
PolygonRegion<VertexT, NormalT>::PolygonRegion() {
	m_label = "unknown";
	NormalT normal(1.0, 0.0, 0.0);
	m_normal = normal;
}

template<typename VertexT, typename NormalT>
PolygonRegion<VertexT, NormalT>::PolygonRegion(std::vector<Polygon<VertexT, NormalT>> new_polygons, std::string label, NormalT normal) {
	m_polygons = new_polygons;
	m_normal   = normal;
	m_label    = label;
}


template<typename VertexT, typename NormalT>
PolygonRegion<VertexT, NormalT>::~PolygonRegion() {
	m_polygons.clear();
}

template<typename VertexT, typename NormalT>
std::vector<Polygon<VertexT, NormalT>> PolygonRegion<VertexT, NormalT>::getPolygons() {
	return m_polygons;
}

template<typename VertexT, typename NormalT>
void PolygonRegion<VertexT, NormalT>::setPolygons(std::vector<Polygon<VertexT, NormalT>> new_polygons, std::string label, std::string normal) {
	m_polygons = new_polygons;
	m_normal   = normal;
	m_label    = label;
}


template<typename VertexT, typename NormalT>
NormalT PolygonRegion<VertexT, NormalT>::getNormal(){
	return m_normal;
}


template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT> PolygonRegion<VertexT, NormalT>::getPolygon(){
	if(!m_polygons.empty())
	{
		return m_polygons[0];
	}
	else
	{
		Polygon<VertexT, NormalT> tmp;
		return tmp;
	}
}


} /* namespace lvr */
