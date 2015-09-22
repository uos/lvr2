/*
 * Polygonregion.tcc
 *
 *  Created on: 06.03.2014
 *      Author: Dominik Feldschnieders (dofeldsc@uos.de)
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
	calcBoundingBox();
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
void PolygonRegion<VertexT, NormalT>::setPolygons(std::vector<Polygon<VertexT, NormalT>> new_polygons, std::string label, NormalT normal) {
	m_polygons = new_polygons;
	m_normal   = normal;
	m_label    = label;
	calcBoundingBox();
}

template<typename VertexT, typename NormalT>
VertexT PolygonRegion<VertexT, NormalT>::getBoundMin() {
	return m_bound_min;
}

template<typename VertexT, typename NormalT>
VertexT PolygonRegion<VertexT, NormalT>::getBoundMax() {
	return m_bound_max;
}

template<typename VertexT, typename NormalT>
NormalT PolygonRegion<VertexT, NormalT>::getNormal(){
	return m_normal;
}

template<typename VertexT, typename NormalT>
size_t PolygonRegion<VertexT, NormalT>::getSize(){
	return m_polygons.at(0).getSize();
}

template<typename VertexT, typename NormalT>
std::string PolygonRegion<VertexT, NormalT>::getLabel(){
	return m_label;
}

template<typename VertexT, typename NormalT>
void PolygonRegion<VertexT, NormalT>::setLabel(std::string new_label)
{
	m_label = new_label;
}

template<typename VertexT, typename NormalT>
void PolygonRegion<VertexT, NormalT>::setNormal(NormalT new_normal)
{
	m_normal = new_normal;
}


template<typename VertexT, typename NormalT>
Polygon<VertexT, NormalT> PolygonRegion<VertexT, NormalT>::getPolygon(){
	// return the outer Polygon(-shell)
	if(!m_polygons.empty())
	{
		return m_polygons[0];
	}
	// or an empty Polygon
	else
	{
		Polygon<VertexT, NormalT> tmp;
		return tmp;
	}
}

template<typename VertexT, typename NormalT>
void PolygonRegion<VertexT, NormalT>::calcBoundingBox()
{
	// get the right min and max values
	float minx = numeric_limits<float>::max();
	float miny = numeric_limits<float>::max();
	float minz = numeric_limits<float>::max();
	float maxx = numeric_limits<float>::min();
	float maxy = numeric_limits<float>::min();
	float maxz = numeric_limits<float>::min();

	// calc the axis aligned Bounding-Box
	typename vector<Polygon<VertexT, NormalT> >::iterator it;
	for(it = m_polygons.begin() ; it != m_polygons.end() ; ++it)
	{
		typename std::vector<VertexT>::iterator it_p;
		std::vector<VertexT> points = it->getVertices();
		for(it_p = points.begin() ; it_p != points.end() ; ++it_p)
		{
			if      ((*it_p).x < minx) minx = (*it_p).x;
			else if ((*it_p).x > maxx) maxx = (*it_p).x;

			if      ((*it_p).y < miny) miny = (*it_p).y;
			else if ((*it_p).y > maxy) maxy = (*it_p).y;

			if      ((*it_p).z < minz) minz = (*it_p).z;
			else if ((*it_p).z > maxz) maxz = (*it_p).z;
		}
	}
	m_bound_min = VertexT(minx, miny, minz);
	m_bound_max = VertexT(maxx, maxy, maxz);
}


} /* namespace lvr */
