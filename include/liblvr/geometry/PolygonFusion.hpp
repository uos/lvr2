/*
 * PolygonFusion.hpp
 *
 *  Created on: 05.03.2014
 *      Author: dofeldsc
 */

#ifndef POLYGONFUSION_HPP_
#define POLYGONFUSION_HPP_

// Boost includes for the Fusion
#include <boost148/geometry.hpp>
#include <boost148/geometry/geometries/point_xy.hpp>
#include <boost148/geometry/geometries/polygon.hpp>
#include <boost148/geometry/domains/gis/io/wkt/wkt.hpp>

#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "geometry/PolygonRegion.hpp"
#include <vector>
#include <map>

// lvr includes
#include "io/Timestamp.hpp"

using namespace lvr;

template<typename VertexT, typename NormalT>
class PolygonFusion {
public:
	typedef PolygonRegion<VertexT, NormalT> Polyregion;
	typedef std::map<std::string, std::vector<Polyregion> > polyRegionMap;

	PolygonFusion();
	virtual ~PolygonFusion();

	bool addFusionMesh(PolygonMesh<VertexT, NormalT> mesh);

	bool doFusion();



private:
	bool isPlanar(Polyregion a, Polyregion b);

	// Vector for all data (Polygonmesh)
	polyRegionMap           m_polyregionmap;



	Timestamp ts;
};

#endif /* POLYGONFUSION_HPP_ */
