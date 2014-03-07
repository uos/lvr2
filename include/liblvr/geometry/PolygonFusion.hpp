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
#include "geometry/PolygonMesh.hpp"
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

	/**
	 * @brief standard constructor
	 */
	PolygonFusion();


	/**
	 * destructor
	 */
	~PolygonFusion();


	/**
	 * @brief add a new PolygonMesh to the Fusion (store it in the container)
	 */
	void addFusionMesh(PolygonMesh<VertexT, NormalT> mesh);


	/**
	 * @brief Fuse all the Meshes (Polygons) in the container
	 *
	 * 		At first, only the Polygons with the same label
	 *
	 * @return returns false, if something went wrong
	 */
	bool doFusion();


private:
	/**
	 * @brief This function tests if these two Polygons are planar
	 *
	 * @return true, if these Polygons are planar
	 */
	bool isPlanar(Polyregion a, Polyregion b);

	// Vector for all data (Polygonmesh)
	polyRegionMap           m_polyregionmap;



	Timestamp ts;
};

#include "PolygonFusion.tcc"
#endif /* POLYGONFUSION_HPP_ */
