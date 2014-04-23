/*
 * PolygonFusion.hpp
 *
 *  Created on: 05.03.2014
 *      Author: Dominik Feldschnieders (dofeldsc@uos.de)
 *      Author: Simon Herkenhoff       (sherkenh@uos.de)
 */

#ifndef POLYGONFUSION_HPP_
#define POLYGONFUSION_HPP_

// Boost includes for the Fusion
#include <boost148/geometry.hpp>
#include <boost148/geometry/geometries/point_xy.hpp>
#include <boost148/geometry/geometries/polygon.hpp>
#include <boost148/geometry/domains/gis/io/wkt/wkt.hpp>
#include <boost/foreach.hpp>

#include <Eigen/Core>
//lvr include
#include "geometry/Vertex.hpp"
#include "geometry/Normal.hpp"
#include "geometry/PolygonRegion.hpp"
#include "geometry/PolygonMesh.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"
// std includes
#include <vector>
#include <map>
#include <sstream>
#include <string>

// lvr includes
#include "io/Timestamp.hpp"

namespace lvr
{

/**
 * @brief This class contains all the functionality for a 3D Polygonfusion.
 * 		The data is represented by the PolygonMesh class and some others.
 *		This class is still in progress and there are some important TODO`s
 *
 * The procedure is as follows:
 *	0.5) Wait and store all given meshes till the method "doFusion" is called
 *	 1) put polyregions into bins according to labels
 *	 2) in these bins, find "co-planar" polyregions -> same plane (Δ)
 *	 3) transform these polygons into 2D space (see spuetz fusion)
 *	 4) apply boost::geometry::union_ for these polygons
 *	 5) transform resulting 2D polygon back into 3d space (inverse of step 3)
 *	 6) place resulting 3D polygon in response mesh
 *	 7) insert all left overs into response.mesh
 *
 */

template<typename VertexT, typename NormalT>
class PolygonFusion {
public:
	// some useful typedefs
	typedef PolygonRegion<VertexT, NormalT>               PolyRegion;
	typedef std::map<std::string, std::vector<PolyRegion> > PolyRegionMap;
	typedef std::vector<PolygonMesh<VertexT, NormalT> >   PolyMesh;
	typedef AdaptiveKSearchSurface<VertexT, NormalT>      akSurface;

	typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float> > BoostPolygon;

	/**
	 * @brief standard constructor
	 */
	PolygonFusion();


	/**
	 * @brief destructor
	 */
	~PolygonFusion();


	/**
	 * @brief Add a new PolygonMesh to the Fusion (store it in the container)
	 *
	 * @param mesh the new PolygonMesh
	 */
	void addFusionMesh(PolygonMesh<VertexT, NormalT> mesh);


	/**
	 * @brief Fuse all the Meshes (Polygons) in the container
	 *
	 * 		At first, only the Polygons with the same label
	 *
	 * 	 1) put polyregions into bins according to labels
	 *	 2) in these bins, find "co-planar" polyregions -> same plane (Δ)
	 *	 3) transform these polygons into 2D space
	 *	 4) apply boost::geometry::union_ for these polygons
	 *	 5) transform resulting 2D polygon back into 3d space (inverse of step 3)
	 *	 6) place resulting 3D polygon in response mesh
	 *	 7) insert all left overs into response.mesh
	 *
	 * @param output The resulting PolygonMesh
	 *
	 * @return returns false, if something went wrong
	 */
	bool doFusion(std::vector<PolyRegion> &output);


	/**
	 * @brief reset the Fusion ( Clear all the data and wait for new instructions )
	 */
	void reset();

	/**
	 * @brief setter for m_useRansac
	 *
	 * @param use_ransac true or false
	 */
	void setRansac(bool use_ransac);


private:
	/**
	 * @brief This function tests if these two Polygons are planar
	 *
	 * @param a the current PolygonRegion
	 * @param b the PolygonRegion, which should be checked
	 *
	 * @return true, if these Polygons are planar
	 */
	bool isPlanar(PolyRegion a, PolyRegion b);

	/**
	 * @brief This method calculates a transformation matrix from the xyz-Plane (3D) to the xy-Plane (2D).
	 * 		The three points span a plane, which is needed for the calculation.
	 *
	 * @param a  a point of this plane
	 * @param b  a second point of this plane
	 * @üaram c  a third point of this plane
	 *
	 * @return the transformation matrix
	 */
	Eigen::Matrix4f calcTransform(VertexT a, VertexT b, VertexT c);

	/**
	 * @brief This method transforms two PolygonRegions in 2D, fuses them and transforms it back
	 *
	 * @param coplanar_polys all coplanar polygons, which should get fused
	 * @param result the new fused PolygonRegion
	 *
	 * @return true, if everything works
	 */
	bool fuse(std::vector<PolyRegion> coplanar_polys, std::vector<PolygonRegion<VertexT, NormalT>> &result);


	/**
	 * @brief Transforms from 3D in 2D and transformation from lvr::PolygonRegion in Boost_Polygo
	 *
	 * @param a PolygonRegion which will be transformed
	 * @param trans transformation as 4x4 matrix
	 *
	 * @return the resulting BoostPolygon
	 */
	BoostPolygon transformto2DBoost(PolyRegion a, Eigen::Matrix4f trans);


	/**
	 * @brief Transforms from 2D in 3D and transformation from Boost_Polygon in lvr::PolygonRegion
	 *
	 * @param a BoostPolygon which will be transformed
	 * @param trans transformation as 4x4 matrix
	 *
	 * @return the resulting lvr PolygonRegion
	 */
	PolygonRegion<VertexT, NormalT> transformto3Dlvr(BoostPolygon a, Eigen::Matrix4f trans);

	// Vector for all data (Polygonmesh)
	PolyRegionMap	m_polyregionmap;

	// Vector for all Polymeshes
	PolyMesh 		m_meshes;

	// tresholt for coplanar check with best fit plane
	double			m_distance_threshold;

	// simplify distance for boost function simplify
	double 			m_simplify_dist;

	// thresholt for overlapping bounding box check
	double 			m_distance_threshold_bounding;

	// part of a TODO
	int dirty_fix;

	// true, if the best fit plane for the transformation should be calculated with Ransac
	// false, the best fit plane will be calculated by the interpolated normal and the centroid of all points
	bool m_useRansac;

	Timestamp ts;
};

} // End of namespace lvr
#include "PolygonFusion.tcc"
#endif /* POLYGONFUSION_HPP_ */
