/*
 * PolygonFusion.tcc
 *
 *  Created on: 05.03.2014
 *      Author: Dominik Feldschnieders (dofeldsc@uos.de)
 *      Author: Simon Herkenhoff       (sherkenh@uos.de)
 */

// TODO Le Kack mit der Transformation in die xy-Ebene funktioniert ja noch nicht richtig...dirty fix wieder rausnehmen
// TODO Die Label aus den Messages müssen später auch wieder in diese Übertragen werden
// TODO Logger einbauen
// TODO in fuse die Gewichtung der normale nochmal überdenken
// TODO transformto2DBoost überarbeiten, boost::geometry::correct einbauen


#include "PolygonFusion.hpp"

namespace lvr
{
template <typename Point>
class round_coordinates
{
private :
    std::vector<Point>* vec;

public :
    round_coordinates(std::vector<Point>* v)
        : vec(v)
    {}

    inline void operator()(Point& p)
    {
        using boost::geometry::get;
        vec->push_back(p);
    }
};


template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::PolygonFusion() {
	// TODO noch in Options auslagern
	m_distance_threshold = 0.05;
	m_simplify_dist = 0.5;
	m_distance_threshold_bounding = 0.05;
	m_useRansac = false;
}


template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::~PolygonFusion() {
	reset();
}


template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::addFusionMesh(PolygonMesh<VertexT, NormalT> mesh) {
	m_meshes.push_back(mesh);
}

template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::setRansac(bool use_ransac)
{
	m_useRansac = use_ransac;
}

template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::doFusion(std::vector<PolyRegion> &output)
{
	std::cout << "Starting PolygonFusion with " << m_meshes.size() << " PolygonMeshes!!" << std::endl;

	// 0.5) prepare map and other vectors
	// 1) put polyregions into bins according to labels
	// 2) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// 3) transform these polygons into 2D space
	// 4) apply boost::geometry::union_ for these polygons
	// 5) transform resulting 2D polygon back into 3d space (inverse of step 3)
	// 6) place resulting 3D polygon in response.mesh
	// 7) insert all left overs into response.mesh

	// step 1) put polyregions into bins according to labels
	// TODO unknown regions werden jetzt betrachtet, entgültige Entscheidung treffen?
	typename PolyMesh::iterator polymesh_iter;
	//std::vector<lvr_tools::PolygonMesh>::iterator polymesh_iter;
	for( polymesh_iter = m_meshes.begin(); polymesh_iter != m_meshes.end(); ++polymesh_iter )
	{

		std::vector<PolyRegion> regions;
		regions = (*polymesh_iter).getPolyRegions();
		typename std::vector<PolyRegion>::iterator polyregion_iter;
		for( polyregion_iter = regions.begin(); polyregion_iter != regions.end(); ++polyregion_iter )
		{
			// only try to fuse, if this region has a label TODO was soll jetzt genau gefust werden
			if ( (*polyregion_iter).getLabel() != "unknown" )
			{
				// if prelabel already exists in map, just push back PolyGroup, else create a new one
				typename PolyRegionMap::iterator it;
				it = m_polyregionmap.find((*polyregion_iter).getLabel());

				if (it != m_polyregionmap.end())
				{
					it->second.insert(it->second.end(), (*polyregion_iter));
				}
				else
				{
					std::vector<PolyRegion> tmp_regions;
					tmp_regions.push_back((*polyregion_iter));
					m_polyregionmap.insert(std::pair<std::string, std::vector<PolyRegion> >((*polyregion_iter).getLabel(), tmp_regions));
				}
			}
			// store the unknown regions
			else
			{
				// if you only want the labeled Regions in the output-vector, removed the following command
				output.push_back((*polyregion_iter));
			}
		}
	}
	std::cout << "Aufteilen der Regionen nach ihren labeln abgeschlossen" << std::endl;
	std::cout << "Es gibt insgesamt ('unknown' wird nicht betrachtet) "<< m_polyregionmap.size()  << " verschiedene Label." << std::endl;


	// step 2-5) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// TODO fix coplanar detection (gewichtet nach Anzahl Punkten)
	// TODO benchmark coplanar threshold and fusion / detection order (not only first one)
	typename PolyRegionMap::iterator map_iter;
	for( map_iter = m_polyregionmap.begin(); map_iter != m_polyregionmap.end(); ++map_iter )
	{
		// init the container
		std::vector<PolyRegion> polyregions = (*map_iter).second;
		std::vector<PolyRegion> coplanar_regions;
		std::vector<PolyRegion> nonplanar_regions;
		std::vector<PolyRegion> fused_regions;


		typename std::vector<PolyRegion>::iterator region_iter;
		for( region_iter = polyregions.begin(); region_iter != polyregions.end(); )
		{
			// assume there exists at least least one coplanar region
			coplanar_regions.push_back((*region_iter));

			// check all polygons with the same label, if they are coplanar to the actual polygon
			typename std::vector<PolyRegion>::iterator coplanar_iter;
			for( coplanar_iter = polyregions.begin(); coplanar_iter != polyregions.end(); )
			{
				// do not compare a polygon to itself
				if ( region_iter != coplanar_iter )
				{
					// if they are coplanar,
					if ( isPlanar((*region_iter), (*coplanar_iter)) )
					{
						coplanar_regions.push_back((*coplanar_iter));
						// remove element from vector
						coplanar_iter = polyregions.erase(coplanar_iter);
					}
					else
					{
						++coplanar_iter;
					}
				}
				else
				{
					++coplanar_iter;
				}
			} // end for coplanar


			// assumption was wrong, no coplanar region for this PolygonRegion
			if ( coplanar_regions.size() == 1 )
			{
				// save this Region in the nonplaner Regions
				nonplanar_regions.push_back((*region_iter));
				coplanar_regions.clear();
			}
			// assumption was correct, need to do fusion with a least two PolygonRegions
			else
			{
				// Try to fuse all coplanar regions at once
				fuse(coplanar_regions, fused_regions);
			}

			// increment region iterator
			region_iter = polyregions.erase(region_iter);

			// store the fused polygonregions in the output vector
			typename std::vector<PolyRegion>::iterator out_it;
			for(out_it = fused_regions.begin() ; out_it != fused_regions.end() ; ++out_it)
			{
				output.push_back((*out_it));
			}

			// clear the container
			fused_regions.clear();
			coplanar_regions.clear();
		} // end for Polyregions with same label

		// store the polygonregions without interest in the output vector
		typename std::vector<PolyRegion>::iterator out_it;
		for(out_it = nonplanar_regions.begin() ; out_it != nonplanar_regions.end() ; ++out_it)
		{
			output.push_back((*out_it));
		}
		nonplanar_regions.clear();
	} // end for map

	// done!
	return true;
}




template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::isPlanar(PolyRegion a, PolyRegion b)
{
	bool coplanar = true;

	// at first, make boundingbox-check
	VertexT min_a = a.getBoundMin();
	VertexT min_b = b.getBoundMin();
	VertexT max_a = a.getBoundMax();
	VertexT max_b = b.getBoundMax();

	// include the distance_threshold (it looks good but there were no big analysis...)
	VertexT dist_thres(m_distance_threshold_bounding, m_distance_threshold_bounding, m_distance_threshold_bounding);
	min_a -= dist_thres;
	max_a += dist_thres;
	min_b -= dist_thres;
	max_b += dist_thres;

	// check BoundingBoxes, if they overlap
	if ( max_a.x < min_b.x || min_a.x > max_b.x ) coplanar = false;
	if ( max_a.y < min_b.y || min_a.y > max_b.y ) coplanar = false;
	if ( max_a.z < min_b.z || min_a.z > max_b.z ) coplanar = false;

	// if the BoundingBox-check failed, return false
	if(!coplanar)
	{
		return coplanar;
	}

	// Now, check real coplanarity
	NormalT norm_a;
	VertexT point_a;

	// get the first vertex of the first polygon of this region and the normal of this region
	std::vector<VertexT> tmp_vec = a.getPolygon().getVertices();
	if(tmp_vec.size() != 0)
	{
		point_a = tmp_vec[0];
		norm_a = a.getNormal();
	}
	else
	{
		std::cout << timestamp << "Polygon a in isPlanar war leer, daher wird false zurueckgegeben! (!!!Dieser Fall sollte nicht auftreten!!!)" << std::endl;
		return false;
	}

	// span the plane (Hesse normal form)
	float n_x = norm_a.x;
	float n_y = norm_a.y;
	float n_z = norm_a.z;

	float p1_x = point_a.x;
	float p1_y = point_a.y;
	float p1_z = point_a.z;

	float d = - ((n_x * p1_x) + (n_y * p1_y) + (n_z * p1_z) );
	float distance = 0.0;

	std::vector<Polygon<VertexT, NormalT>> polygons_b;
	polygons_b = b.getPolygons();

	// Check for all points (outer Polygon(-shell)) from polyregion b, the point to plane (polyregion a) distance
	std::vector<VertexT> check_me = polygons_b.at(0).getVertices();
	typename std::vector<VertexT>::iterator point_iter;
	for( point_iter = check_me.begin(); coplanar != false, point_iter != check_me.end(); ++point_iter )
	{
		distance = abs( ( ( n_x * (*point_iter).x ) + ( n_y  *  (*point_iter).y ) + ( n_z  *  (*point_iter).z ) + d ) ) / sqrt( n_x * n_x + n_y * n_y + n_z * n_z );
		if ( distance > m_distance_threshold )
		{
			coplanar = false;
		}
	}

	return coplanar;
}

template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::fuse(std::vector<PolyRegion> coplanar_polys, std::vector<PolygonRegion<VertexT, NormalT>> &result)
{
	// we need all points from the polygons, so we can calculate a best fit plane
	int c_count = 0;
	std::vector<VertexT> ransac_points;
	VertexT centroid(0.0, 0.0, 0.0);
	NormalT c_normal(0.0, 0.0, 0.0);

	typename std::vector<PolyRegion>::iterator region_iter;
	for(region_iter = coplanar_polys.begin(); region_iter != coplanar_polys.end(); ++region_iter)
	{
		// estimate interpolate normal
		c_normal += (region_iter->getNormal() * region_iter->getSize());
		c_count += region_iter->getSize();
		std::vector<Polygon<VertexT, NormalT> > polygons = region_iter->getPolygons();
		typename std::vector<Polygon<VertexT, NormalT> >::iterator poly_iter;
		for(poly_iter = polygons.begin(); poly_iter != polygons.end(); ++poly_iter)
		{
			std::vector<VertexT> points = poly_iter->getVertices();
			typename std::vector<VertexT>::iterator point_iter;
			for(point_iter = points.begin(); point_iter != points.end(); ++point_iter)
			{
				// estimate interpolate centroid
				centroid += (*point_iter);
				ransac_points.push_back(*point_iter);
			}
		}
	}

	// normalize normal and interpolate centroid
	c_normal /= c_count;
	c_normal.normalize();
	centroid /= ransac_points.size();

	// calculate best fit plane with ransac or with interpolated normal and centroid
	Plane<VertexT, NormalT> plane;
	if(m_useRansac)
	{
		// see typedef in header
		akSurface akss;

		// calc best fit plane with ransac
		bool ransac_success = true;
		plane = akss.calcPlaneRANSACfromPoints(centroid, ransac_points.size(), ransac_points, c_normal, ransac_success);

		if (!ransac_success)
		{
			cout << timestamp << "UNABLE TO USE RANSAC FOR PLANE CREATION" << endl;
			return false;
		}
	}
	else
	{
		plane.a = 0;
		plane.b = 0;
		plane.c = 0;
		plane.n = c_normal;
		plane.p = centroid;
	}

	//float d = (plane.p.x * plane.n.x) + (plane.p.y * plane.n.y) + (plane.p.z * plane.n.z);

	// calc 2 points on this best fit plane, we need it for the transformation in 2D
	// take a random vector and cross it with the normal...you get a vector that can be used
	// to span the plane
	VertexT vec1(1, 2, 3);
	VertexT vec2(3, 2, 1);
	VertexT check_vec(0.0, 0.0, 0.0);

	vec1.crossTo(plane.n);
	vec2.crossTo(plane.n);

	//check if vec1 or vec 2 was paralell or equal to the normal of the plane
	if(check_vec == vec1)
	{
		VertexT tmp(2, 2, 3);
		vec1 = tmp.cross(plane.n);
	}
	else if(check_vec == vec2)
	{
		VertexT tmp(2, 2, 3);
		vec2 = tmp.cross(plane.n);
	}

	vec1 += plane.p;
	vec2 += plane.p;

	// calc transform
	Eigen::Matrix4f trans_mat;
	Eigen::Matrix4f trans_mat_inv;

	trans_mat = calcTransform(plane.p, vec1, vec2);

	// need transform from 3D to 2D and back from 2D to 3D, so we need the inverse matrix too
	trans_mat_inv = trans_mat.inverse();

	std::vector<BoostPolygon> input;
	std::vector<BoostPolygon> output;

	// transform all the PolygonRegions
	typename std::vector<PolyRegion>::iterator poly_iter;
	for(poly_iter = coplanar_polys.begin(); poly_iter != coplanar_polys.end(); ++poly_iter)
	{
		// transform to 2D boost polyon
		input.push_back(transformto2DBoost((*poly_iter), trans_mat));
	}

	// the rest of the method is a little bit dirty, because boost (at the latest with ROS compatible Version) can´t handle
	// Polygons with intersections but there are a bunch of methods, that let us handle it but only in a dirty style.
	// (so, i hope these problems are fixed if someone new tries...)
	bool first_it = true;
	std::vector<BoostPolygon> intersectors;


	// for all PolygonRegions aka BoostPolygons
	typename std::vector<BoostPolygon>::iterator input_iter;
	for(input_iter = input.begin(); input_iter != input.end(); ++input_iter)
	{
		BoostPolygon simplified_a, simplified_b;
		// in the first round, store the PolygonRegions in a container, which is used for fusion
		if(first_it)
		{
			// if this Polygonregions has intersections, try to simplify it and store it
			if(boost::geometry::intersects((*input_iter)))
			{
				boost::geometry::simplify((*input_iter), simplified_a, m_simplify_dist);
				output.push_back(simplified_a);
				first_it = false;
			}
			else
			{
				first_it = false;
				output.push_back((*input_iter));
			}
		}
		// if output is empty, we need a "first" element in it
		else if(output.size() == 0) first_it = true;
		else
		{
			// if this Polygonregions has intersections, try to simplify it and try to fuse it
			if(boost::geometry::intersects((*input_iter)))
			{
				boost::geometry::simplify((*input_iter), simplified_b, m_simplify_dist);

				BoostPolygon tmp;
				// get the first polygon and erase it, if not there are double polys in the output-vec
				if (output.size() == 1)
				{
					tmp = output[0];
					output.clear();
				}
				else
				{
					// TODO nicht immer nur den ersten nehmen, am besten groessten???
					tmp = output[0];
					output.erase(output.begin());

					// if this Polygonregions has intersections, try to simplify it for the fusion
					if(boost::geometry::intersects(tmp))
					{
						boost::geometry::simplify(output[0], tmp, m_simplify_dist);
					}
				}

				// try to catch exception from boost union, problem with intersections
				try
				{
					boost::geometry::union_(tmp, simplified_b , output);
				}
				// TODO hier können noch viele unbehandelte Sonderfälle auftreten
				catch(...)
				{
					// if the fusion failed and the container is empty, check for intersections and store them
					// in the output, if the still have intersections, store them in the intersectors container
					if(output.size() == 0)
					{
						if (boost::geometry::intersects(simplified_b))
						{
							intersectors.push_back(simplified_b);
						}
						else output.push_back(simplified_b);

						if (boost::geometry::intersects(tmp))
						{
							intersectors.push_back(tmp);
							first_it = true;
						}
						else output.push_back(tmp);
					}
					else
					{
						// TODO hier muss noch weiter probiert werden und nicht alle weg
						intersectors.push_back(tmp);
						intersectors.push_back(simplified_b);
					}
				}
			}
			// current poly has no intersections
			else
			{
				BoostPolygon fuse_poly;
				// get the first polygon and erase it, if not there are double polys in the output-vec
				if (output.size() == 1)
				{
					fuse_poly = output[0];
					output.clear();
				}
				else
				{
					// TODO nicht immer nur den ersten nehmen, am besten groessten???
					fuse_poly = output[0];
					output.erase(output.begin());
				}
				// if this Polygonregions has intersections, try to simplify it for the fusion
				if(boost::geometry::intersects(fuse_poly))
				{
					BoostPolygon tmp = fuse_poly;
					boost::geometry::simplify(tmp, fuse_poly, m_simplify_dist);
				}

				// try to fuse both Polygons
				try
				{
					boost::geometry::union_(fuse_poly, (*input_iter) , output);
				}
				catch(...)
				{
					// if it doesn´t work, store the Polygons in the corresponding container
					output.push_back((*input_iter));
					if(boost::geometry::intersects(fuse_poly)) intersectors.push_back(fuse_poly);
					else output.push_back(fuse_poly);

				}
			}
		} // end if else first_it

	} // end for Boostpolygone


	// store the unfused polygonregions in the output vec and transform them into 3D
	for(input_iter = intersectors.begin(); input_iter != intersectors.end(); ++input_iter)
	{
		result.push_back(transformto3Dlvr((*input_iter),trans_mat_inv));
	}

	// transform the fused polygonregion in 3D and store then in the output vec
	typename std::vector<BoostPolygon>::iterator output_iter;
	for(output_iter = output.begin(); output_iter != output.end(); ++output_iter)
	{
		result.push_back(transformto3Dlvr((*output_iter), trans_mat_inv));
	}

	return true;
}


template<typename VertexT, typename NormalT>
Eigen::Matrix4f PolygonFusion<VertexT, NormalT>::calcTransform(VertexT a, VertexT b, VertexT c)
{

     // calculate the plane-vectors
     VertexT vec_AB = b - a;
     VertexT vec_AC = c - a;

     // calculate the required angles for the rotations
     double alpha   = atan2(vec_AB.z, vec_AB.x);
     double beta    = -atan2(vec_AB.y, cos(alpha)*vec_AB.x + sin(alpha)*vec_AB.z);
     double gamma   = -atan2(-sin(alpha)*vec_AC.x+cos(alpha)*vec_AC.z,
                    sin(beta)*(cos(alpha)*vec_AC.x+sin(alpha)*vec_AC.z) + cos(beta)*vec_AC.y);

     Eigen::Matrix4f trans;
     trans << 1.0, 0.0, 0.0, -a.x,
    		  0.0, 1.0, 0.0, -a.y,
    		  0.0, 0.0, 1.0, -a.z,
    		  0.0, 0.0, 0.0, 1.0;

     Eigen::Matrix4f roty;
     roty << cos(alpha), 0.0, sin(alpha), 0.0,
    		 0.0, 1.0, 0.0, 0.0,
    		 -sin(alpha), 0.0, cos(alpha), 0.0,
    		 0.0, 0.0, 0.0, 1.0;

     Eigen::Matrix4f rotz;
     rotz << cos(beta), -sin(beta), 0.0, 0.0,
    		 sin(beta), cos(beta), 0.0, 0.0,
    		 0.0, 0.0, 1.0, 0.0,
    		 0.0, 0.0, 0.0, 1.0;

     Eigen::Matrix4f rotx;
     rotx << 1.0, 0.0, 0.0, 0.0,
    		 0.0, cos(gamma), -sin(gamma), 0.0,
    		 0.0, sin(gamma), cos(gamma), 0.0,
    		 0.0, 0.0, 0.0, 1.0;

     /*
      * transformation to the xy-plane:
      * first translate till the point a is in the origin of ordinates,
      * then rotate around the the y-axis till the z-value of the point b is zero,
      * then rotate around the z-axis till the y-value of the point b is zero,
      * then rotate around the x-axis till all z-values are zero
      */

     return rotx * rotz * roty * trans;
}


template<typename VertexT, typename NormalT>
boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float> > PolygonFusion<VertexT, NormalT>::transformto2DBoost(PolyRegion a, Eigen::Matrix4f trans)
{
	// TODO boost::geometry::correct - Einbauen, spart die staendigen Abfragen und umdrehen von Polygonen bzw. Loechern

	BoostPolygon result, tmp_poly;
	bool first_it;
	bool first_poly = true;
	int dirty_x = 0;
	int dirty_y = 0;
	int dirty_z = 0;
	// we need some stringstreams to build the BoostPolygons
	std::stringstream res_poly_ss;
	res_poly_ss << "POLYGON(";
	std::stringstream first_poly_ss;
	std::stringstream poly_ss;

	// get all the polygons from this region
	std::vector<Polygon<VertexT, NormalT> > polygons = a.getPolygons();
	typename std::vector<Polygon<VertexT, NormalT> >::iterator poly_iter;
	// for all polygons in this region
	for(poly_iter = polygons.begin() ; poly_iter != polygons.end() ; ++poly_iter )
	{
		first_it = true;

		// get all vertices from this polygon
		std::vector<VertexT> points = poly_iter->getVertices();
		typename std::vector<VertexT>::iterator point_iter;
		// for all vertices: transform them into the xy-plane
		for(point_iter = points.begin(); point_iter != points.end(); ++point_iter)
		{
			Eigen::Matrix<double, 4, 1> pt(point_iter->x, point_iter->y, point_iter->z, 1);

			float x = 	trans(0,0) * pt.coeffRef(0) +
						trans(0,1) * pt.coeffRef(1) +
						trans(0,2) * pt.coeffRef(2) +
						trans(0,3) * pt.coeffRef(3);

			float y = 	trans(1,0) * pt.coeffRef(0) +
						trans(1,1) * pt.coeffRef(1) +
						trans(1,2) * pt.coeffRef(2) +
						trans(1,3) * pt.coeffRef(3);

// Debug stuff, test if trans goes right
			float z = 	trans(2,0) * pt.coeffRef(0) +
						trans(2,1) * pt.coeffRef(1) +
						trans(2,2) * pt.coeffRef(2) +
						trans(2,3) * pt.coeffRef(3);

			if(abs(x) <= abs(z) && abs(x) <= abs(y) )
			{
				dirty_x++;
				x = y;
				y = z;
			}
			else if(abs(y) <= abs(z) && abs(y) <= abs(x))
			{
				dirty_y++;
				y = z;
			}
			// z is minimal
			else if (abs(z) <= abs(y) && abs(z) <= abs(x))
			{
				dirty_z++;
			}
			else
			{
				std::cout << "x,y und z lassen sich in keiner Reihenfolge birngen" << std::endl;
			}


			// transform in BoostPolygon - a BoostPolygon looks like "POLYGON( 1 1, 1 2, 2 2, 2 1, 1 1)"
			if (first_it)
			{
				// save the first one, for closing the polygon
				first_poly_ss << std::to_string(x) << " " << std::to_string(y);
				first_it = false;

				poly_ss << "(" << std::to_string(x) << " " << std::to_string(y) << ", ";
			}
			else
			{
				poly_ss << std::to_string(x) << " " << std::to_string(y) << ", ";
			}
		}
		poly_ss << first_poly_ss.str() << ")";

		// flush the stringstream
		first_poly_ss.str("");
		first_poly_ss.clear();
		// check every single polygon, if it conform to the boost-polygon-style
		std::string test_poly_str = "POLYGON(";
		test_poly_str.append(poly_ss.str());
		test_poly_str.append(")");

		// read_wkt creates a BoostPolygon from the string
		boost::geometry::read_wkt(test_poly_str, tmp_poly);


		if(first_poly)
		{
			first_poly = false;
			// if the outer Polygon(-shell) has a negativ area, do the polygon it the other direction (boost polygon style)
			if(boost::geometry::area(tmp_poly) <= 0)
			{
				// clear / flush the stringstreams
				first_poly_ss.str("");
				first_poly_ss.clear();
				poly_ss.str("");
				poly_ss.clear();
				first_it = true;

				// get all vertices from this polygon
				std::vector<VertexT> points = poly_iter->getVertices();
				typename std::vector<VertexT>::iterator point_iter;
				for(point_iter = points.end() - 1; point_iter != points.begin() - 1; --point_iter)
				{
					Eigen::Matrix<double, 4, 1> pt(point_iter->x, point_iter->y, point_iter->z, 1);

					float x = 	trans(0,0) * pt.coeffRef(0) +
								trans(0,1) * pt.coeffRef(1) +
								trans(0,2) * pt.coeffRef(2) +
								trans(0,3) * pt.coeffRef(3);

					float y = 	trans(1,0) * pt.coeffRef(0) +
								trans(1,1) * pt.coeffRef(1) +
								trans(1,2) * pt.coeffRef(2) +
								trans(1,3) * pt.coeffRef(3);

					float z = 	trans(2,0) * pt.coeffRef(0) +
								trans(2,1) * pt.coeffRef(1) +
								trans(2,2) * pt.coeffRef(2) +
								trans(2,3) * pt.coeffRef(3);

					if(abs(x) <= abs(z) && abs(x) <= abs(y) )
					{
						dirty_x++;;
						x = y;
						y = z;
					}
					else if(abs(y) <= abs(z) && abs(y) <= abs(x))
					{
						dirty_y++;;
						y = z;
					}
					// z is minimal
					else if (abs(z) <= abs(y) && abs(z) <= abs(x))
					{
						dirty_z++;
					}
					else
					{
						std::cout << "x,y und z lassen sich in keiner Reihenfolge birngen" << std::endl;
					}
					//std::cout << "In transformto2DBoost (in if(poly_first)) ist der transformierte Vektor bzw. Matrix: " << std::endl;
					//std::cout << tmp_mat << std::endl;

					// transform in BoostPolygon
					if (first_it)
					{
						// save the first one, for closing the polygon
						first_poly_ss << std::to_string(x) << " " << std::to_string(y);
						first_it = false;

						poly_ss << "(" << std::to_string(x) << " " << std::to_string(y) << ", ";
					}
					else
					{
						poly_ss << std::to_string(x) << " " << std::to_string(y) << ", ";
					}
				}
				poly_ss << first_poly_ss.str() << ")";
			}
		}
		// if one of the inner Polygons (holes) has a positive area, do the polygon it the other direction (boost polygon style)
		else if(boost::geometry::area(tmp_poly) >= 0 )
		{
			// clear / flush the stringstreams
			first_poly_ss.str("");
			first_poly_ss.clear();
			poly_ss.str("");
			poly_ss.clear();

//			std::cout << "andere Richtung bei inneren" << std::endl;
			//boost::reverse(tmp_poly);
			first_it = true;

			// get all vertices from this polygon
			std::vector<VertexT> points = poly_iter->getVertices();
			typename std::vector<VertexT>::iterator point_iter;
			for(point_iter = points.end() - 1; point_iter != points.begin() - 1 ;  --point_iter)
			{
				Eigen::Matrix<double, 4, 1> pt(point_iter->x, point_iter->y, point_iter->z, 1);

				float x = 	trans(0,0) * pt.coeffRef(0) +
							trans(0,1) * pt.coeffRef(1) +
							trans(0,2) * pt.coeffRef(2) +
							trans(0,3) * pt.coeffRef(3);

				float y = 	trans(1,0) * pt.coeffRef(0) +
							trans(1,1) * pt.coeffRef(1) +
							trans(1,2) * pt.coeffRef(2) +
							trans(1,3) * pt.coeffRef(3);

				float z = 	trans(2,0) * pt.coeffRef(0) +
							trans(2,1) * pt.coeffRef(1) +
							trans(2,2) * pt.coeffRef(2) +
							trans(2,3) * pt.coeffRef(3);

				if(abs(x) <= abs(z) && abs(x) <= abs(y) )
				{
					dirty_x++;
					x = y;
					y = z;
				}
				else if(abs(y) <= abs(z) && abs(y) <= abs(x))
				{
					dirty_y++;
					y = z;
				}
				// z is minimal
				else if (abs(z) <= abs(y) && abs(z) <= abs(x))
				{
					dirty_z++;
				}
				else
				{
					std::cout << "x,y und z lassen sich in keiner Reihenfolge birngen" << std::endl;
				}
				// transform in BoostPolygon
				if (first_it)
				{
					// save the first one, for closing the polygon
					first_poly_ss << std::to_string(x) << " " << std::to_string(y);
					first_it = false;

					poly_ss << "(" << std::to_string(x) << " " << std::to_string(y) << ", ";
				}
				else
				{
					poly_ss << std::to_string(x) << " " << std::to_string(y) << ", ";
				}
			}
			poly_ss << first_poly_ss.str() << ")";
		}
		// now, this "part" of the polygon can be added to the complete polygon
		res_poly_ss << poly_ss.str();
		first_poly_ss.str("");
		first_poly_ss.clear();
	}
	res_poly_ss << ")";

	boost::geometry::read_wkt(res_poly_ss.str(), result);

	if(dirty_x >= dirty_y && dirty_x >= dirty_z) dirty_fix = 1;
	else if(dirty_y >= dirty_x && dirty_y >= dirty_z) dirty_fix = 2;
	else if(dirty_z >= dirty_x && dirty_z >= dirty_y) dirty_fix = 3;

    std::cout << "Dirty-fix ist: " << dirty_fix << std::endl;
    std::cout << "Dirty-fix ist x: " << dirty_x << std::endl;
    std::cout << "Dirty-fix ist y: " << dirty_y << std::endl;
    std::cout << "Dirty-fix ist z: " << dirty_z << std::endl;

	return result;
}

template<typename VertexT, typename NormalT>
PolygonRegion<VertexT, NormalT> PolygonFusion<VertexT, NormalT>::transformto3Dlvr(BoostPolygon poly, Eigen::Matrix4f trans)
{
    typedef boost::geometry::model::d2::point_xy<float> point;
    using boost::geometry::get;

    // store all the points in vec
    std::vector<point> vec;
    boost::geometry::for_each_point(poly, round_coordinates<point>(&vec));

    // some different modi
    double f_x, f_y;
    bool first_p = true;
    double tmp_x, tmp_y, tmp_z;

    std::vector<Polygon<VertexT, NormalT>> poly_vec;
    std::vector<VertexT> point_vec;
    std::vector<point>::iterator point_iter;
    for(point_iter = vec.begin() ; point_iter != vec.end() ; ++point_iter)
    {
    	// to determine every single polygon (contour, hole etc.)
    	if (first_p)
    	{

    		if(dirty_fix == 1)
    		{
    			tmp_x = 0;
    			tmp_y = get<0>((*point_iter));
    			tmp_z = get<1>((*point_iter));
    		}
    		else if (dirty_fix == 2)
    		{
    			tmp_x = get<0>((*point_iter));
    			tmp_y = 0;
    			tmp_z = get<1>((*point_iter));
    		}
    		else if (dirty_fix == 3)
    		{
    			tmp_x = get<0>((*point_iter));
    			tmp_y = get<1>((*point_iter));
    			tmp_z = 0;
    		}
			Eigen::Matrix<double, 4, 1> pt(tmp_x, tmp_y, tmp_z, 1);

			float x = 	trans(0,0) * pt.coeffRef(0) +
						trans(0,1) * pt.coeffRef(1) +
						trans(0,2) * pt.coeffRef(2) +
						trans(0,3) * pt.coeffRef(3);

			float y = 	trans(1,0) * pt.coeffRef(0) +
						trans(1,1) * pt.coeffRef(1) +
						trans(1,2) * pt.coeffRef(2) +
						trans(1,3) * pt.coeffRef(3);

			float z = 	trans(2,0) * pt.coeffRef(0) +
						trans(2,1) * pt.coeffRef(1) +
						trans(2,2) * pt.coeffRef(2) +
						trans(2,3) * pt.coeffRef(3);

    		f_x = get<0>((*point_iter));
    		f_y = get<1>((*point_iter));
    		first_p = false;

			// store the point
			VertexT tmp(x, y, z);
			point_vec.push_back(tmp);
    	}
    	else
    	{
    		double x,y;
    		x = get<0>((*point_iter));
    		y = get<1>((*point_iter));

    		if(x == f_x && y == f_y)
    		{
//    			std::cout << "Ein Polygon ist vollstaendig, da letzter und ersten Punkt gleich sind" << std::endl;
    			Polygon<VertexT, NormalT> bla(point_vec);
    			poly_vec.push_back(bla);
    			point_vec.clear();
    			first_p = true;
    		}
    		else
    		{
        		if(dirty_fix == 1)
        		{
        			tmp_x = 0;
        			tmp_y = get<0>((*point_iter));
        			tmp_z = get<1>((*point_iter));
        		}
        		else if (dirty_fix == 2)
        		{
        			tmp_x = get<0>((*point_iter));
        			tmp_y = 0;
        			tmp_z = get<1>((*point_iter));
        		}
        		else if (dirty_fix == 3)
        		{
        			tmp_x = get<0>((*point_iter));
        			tmp_y = get<1>((*point_iter));
        			tmp_z = 0;
        		}
    			Eigen::Matrix<double, 4, 1> pt(tmp_x, tmp_y, tmp_z, 1);

    			float x = 	trans(0,0) * pt.coeffRef(0) +
    					trans(0,1) * pt.coeffRef(1) +
    					trans(0,2) * pt.coeffRef(2) +
    					trans(0,3) * pt.coeffRef(3);

    			float y = 	trans(1,0) * pt.coeffRef(0) +
    					trans(1,1) * pt.coeffRef(1) +
    					trans(1,2) * pt.coeffRef(2) +
    					trans(1,3) * pt.coeffRef(3);

    			float z = 	trans(2,0) * pt.coeffRef(0) +
    					trans(2,1) * pt.coeffRef(1) +
    					trans(2,2) * pt.coeffRef(2) +
    					trans(2,3) * pt.coeffRef(3);

    			// store the point
    			VertexT tmp(x, y, z);
    			point_vec.push_back(tmp);
    		}
    	} // end else
    } // end for

    //TODO hier muessen noch das Label und die normale zur Verfügung stehen
    std::string label = "noch_keins_da";
    NormalT normal;
    PolyRegion result(poly_vec, label, normal);
    std::cout << "Dirty-fix ist: " << dirty_fix << std::endl;
    return result;
}


template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::reset()
{
	m_polyregionmap.clear();
	m_meshes.clear();
}

} // Ende of namespace lvr

