/*
 * PolygonFusion.tcc
 *
 *  Created on: 05.03.2014
 *      Author: dofeldsc
 */

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
	std::cout << "Beim Auslesen: x = " << get<0>(p) << " y = " << get<1>(p) << std::endl;
	vec->push_back(p);
    }
};
/*
template <typename Point>
void list_coordinates(Point const& p)
{
    using boost::geometry::get;

    std::cout << "x = " << get<0>(p) << " y = " << get<1>(p) << std::endl;

}
*/

template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::PolygonFusion() {
	// TODO Auto-generated constructor stub
	// noch in Options auslagern
	m_distance_threshold = 0.05;
}


template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::~PolygonFusion() {
	// TODO Auto-generated destructor stub
}


template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::addFusionMesh(PolygonMesh<VertexT, NormalT> mesh) {
	// Hier stellt sich mir die Frage, ob wir jetzt gleich die Regionen in die Map eintragen
	// oder erstmal die gesamten Meshes speichern und dann später bei der Fusion erst die Map benutzen?
	m_meshes.push_back(mesh);
}


template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::doFusion(std::vector<PolyRegion> &output)
{
	// TODO Umbauen auf neue Struktur
	std::cout << "Starting PolygonFusion with " << m_meshes.size() << " Polygons!!" << std::endl;

	// 0.5) prepare map and other vectors
	// 1) put polyregions into bins according to labels
	// 2) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// 3) transform these polygons into 2D space (see spuetz fusion)
	// 4) apply boost::geometry::union_ for these polygons
	// 5) transform resulting 2D polygon back into 3d space (inverse of step 3)
	// 6) place resulting 3D polygon in response.mesh
	// 7) insert all left overs into response.mesh

	// step 0.5)
	//typedef std::map<std::string, std::vector<lvr_tools::PolygonRegion> > polyRegionMap;
	//polyRegionMap polygonsByRegion;

	// step 1) put polyregions into bins according to labels
	// TODO unknown regions!!
	typename PolyMesh::iterator polymesh_iter;
	//std::vector<lvr_tools::PolygonMesh>::iterator polymesh_iter;
	for( polymesh_iter = m_meshes.begin(); polymesh_iter != m_meshes.end(); ++polymesh_iter )
	{
		//std::vector<lvr_tools::PolygonRegion>::iterator polyregion_iter;
		std::vector<PolyRegion> regions;
		regions = (*polymesh_iter).getPolyRegions();
		typename std::vector<PolyRegion>::iterator polyregion_iter;
		for( polyregion_iter = regions.begin(); polyregion_iter != regions.end(); ++polyregion_iter )
		{
			if ( (*polyregion_iter).getLabel() != "unknown" )
			{
				// if prelabel already exists in map, just push back PolyGroup, else create
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
		}
	}
	std::cout << "Aufteilen der Regionen nach ihren labeln abgeschlossen" << std::endl;
/*
// debug stuff
	// Anzeigen von allen Polygonen mit gleichem Label
	ROS_WARN("Vor der Schleife: Size der Map (Label) %d", polygonsByRegion.size());

	polyRegionMap::iterator map_iter2;
	for( map_iter2 = polygonsByRegion.begin(); map_iter2 != polygonsByRegion.end(); ++map_iter2 )
	{

		lvr_tools::PolygonMesh polymesh;

		std::vector<lvr_tools::PolygonRegion> polyregions = (*map_iter2).second;
		std::vector<lvr_tools::PolygonRegion>::iterator reg_it;
		for(reg_it = polyregions.begin() ; reg_it != polyregions.end() ; ++reg_it)
		{
			polymesh.polyregions.push_back((*reg_it));
		}
		ROS_ERROR("So jetzt zeig mal die regions an: &s", (*reg_it).label.c_str());
		polymesh_pub.publish(polymesh);
		sleep(10);
	}
// debug end
*/
	// step 2-5) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// TODO fix coplanar detection (gewichtet nach Anzahl Punkten)
	// TODO benchmark coplanar threshold and fusion / detection order (not only first one)
	typename PolyRegionMap::iterator map_iter;
	for( map_iter = m_polyregionmap.begin(); map_iter != m_polyregionmap.end(); ++map_iter )
	{
		std::cout << "trying to fuse " << (*map_iter).second.size() << " polygons with regionlabel: " <<  (*map_iter).first << std::endl;

		std::vector<PolyRegion> polyregions = (*map_iter).second;

		std::vector<PolyRegion> coplanar_regions;
		std::vector<PolyRegion> nonplanar_regions;
		std::vector<PolyRegion> fused_regions;

		size_t count = 0;
		typename std::vector<PolyRegion>::iterator region_iter;
		for( region_iter = polyregions.begin(); region_iter != polyregions.end(); )
		{
			std::cout << "still need to process " << (polyregions.size() - (count++)) <<
					" / " << polyregions.size() << " outer PolygonRegions" << std::endl;

			// assume there exists at least least one coplanar region
			coplanar_regions.push_back((*region_iter));

//			if ( polyfusion_first_publish )
//			{
//				lvr_tools::PolygonMesh pm;
//				pm.header.frame_id = (*region_iter).header.frame_id;
//				pm.header.stamp = (*region_iter).header.stamp;
//				pm.polyregions.push_back()
//			}

			typename std::vector<PolyRegion>::iterator coplanar_iter;
			for( coplanar_iter = polyregions.begin(); coplanar_iter != polyregions.end(); )
			{
				// do not compare a polygon to itself
				if ( region_iter != coplanar_iter )
				{
					// do stuff
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
			}

			std::cout << "Es wurden Coplanare polygone gefunden und werden gefused: " << coplanar_regions.size() << std::endl;
			// assumption was wrong, no coplanar region
			if ( coplanar_regions.size() == 1 )
			{
				nonplanar_regions.push_back((*region_iter));
				coplanar_regions.clear();
			}
			// assumption was correct, need to do fusion
			else
			{
				// hier haben wir in coplanar_regions mehr als eine polyregion, die zusammen passend

				// gemittelte normale berechnen
				size_t nPoints = 0;
				VertexT normal;
				typename vector<PolyRegion>::iterator region_iter;
				for ( region_iter = coplanar_regions.begin(); region_iter != coplanar_regions.end(); ++region_iter )
				{
					normal += (region_iter->getNormal() * region_iter->getSize());
					nPoints += region_iter->getSize();
				}
				normal /= nPoints;


				fuse(coplanar_regions, output);
				std::cout << "YEAH YEAH YEAH, die Fusion ist durchgelaufen" << std::endl;
				// transform
				// do fusion
				// transform back
				// push_back
				// erase
			}

			// increment region iterator
			region_iter = polyregions.erase(region_iter);
		}
	}

	// done!

	return false;
}




template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::isPlanar(PolyRegion a, PolyRegion b)
{
	std::cout << "IsPlanar wurde aufgerufen" << std::endl;
	// To-Do Umbauen für die neuen Typen (Polygon statt msg:Polygon)
	bool coplanar = true;

	NormalT norm_a;
	VertexT point_a;
	norm_a = a.getNormal();
	// get the first vertex of the first polygon of this region
	point_a = a.getPolygon().getVertices()[0];

	// normale * p = d
	// a*x + b*y + c*z + d = 0
	float n_x = norm_a.x;
	float n_y = norm_a.y;
	float n_z = norm_a.z;

	float p1_x = point_a.x;
	float p1_y = point_a.y;
	float p1_z = point_a.z;

	float d = -(n_x * p1_x + n_y * p1_y + n_z * p1_z);// / sqrt( n_x * n_x + n_y * n_y + n_z * n_z );
	float distance = 0.0;

	std::cout << "Punkt ( " << p1_x << " ," <<
			p1_y << ", " << p1_z << ") mit d = " << d << std::endl;

	std::vector<Polygon<VertexT, NormalT>> polygons_b;
	polygons_b = b.getPolygons();


// Frage: Wir betrachten hier nur das äußere Polygon, reicht das? Also ich glaub schon, da die ja eh auf einer Ebene liegen sollten
	std::vector<VertexT> check_me = polygons_b.at(0).getVertices();
	typename std::vector<VertexT>::iterator point_iter;
	for( point_iter = check_me.begin(); coplanar != false, point_iter != check_me.end(); ++point_iter )
	{
		distance = abs( ( ( n_x * (*point_iter).x ) + ( n_y  *  (*point_iter).y ) + ( n_z  *  (*point_iter).z ) + d ) ) / sqrt( n_x * n_x + n_y * n_y + n_z * n_z );
		if ( distance > m_distance_threshold )
		{
			coplanar = false;
			std::cout << "punkt ( " << (*point_iter).x << " ," <<
					(*point_iter).y << ", " << (*point_iter).z << ") nicht Coplanar mit distance: " << distance << std::endl;
		}
		else
		{
			std::cout << "******** COPLANAR!! Distance is " << distance << std::endl;
			/*
			// TODO remove after testing
			if ( polyfusion_first_publish )
			{
				lvr_tools::PolygonMesh pm;
				pm.header.frame_id = a.header.frame_id;
				pm.header.stamp = a.header.stamp;
				pm.polyregions.push_back(a);
				poly_debug1_pub.publish(pm);
				pm.header.frame_id = b.header.frame_id;
				pm.header.stamp = b.header.stamp;
				pm.polyregions.clear();
				pm.polyregions.push_back(b);
				poly_debug2_pub.publish(pm);
				polyfusion_first_publish = false;
				ROS_WARN("published two polygons with distance %f", distance);
				// punkte ausgeben
				vector<lvr_tools::Polygon>::iterator poly_iter;
				for(poly_iter = a.polygons.begin(); poly_iter != a.polygons.end(); ++poly_iter)
				{
					ROS_WARN("Anzahl Punkte in Polygon: %d", poly_iter->points.size());
//					vector<geometry_msgs::Point32>::iterator point_iter;
//					for(point_iter = poly_iter->points.begin(); point_iter != poly_iter->points.end() ; ++point_iter)
//					{
//						//ROS_WARN("Punkt: %f  %f  %f  ", (*point_iter).x, (*point_iter).y, (*point_iter).z);
//					}
				}
			} */
		}
	}

	if(coplanar) std::cout << "IsPlanar liefert true zurueck" << std::endl;
	else         std::cout << "IsPlanar liefert false zurueck mit distance: " << distance << std::endl;
	return coplanar;
}

template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::fuse(std::vector<PolyRegion> coplanar_polys, std::vector<PolygonRegion<VertexT, NormalT>> &result){
	// TODO Boost Fusion hier implementieren und
std::cout << "fuse wurde aufgerufen" << std::endl;
	// we need all points from the polygons, so we can calculate a best fit plane
	std::vector<VertexT> ransac_points;
	VertexT centroid;
	typename std::vector<PolyRegion>::iterator region_iter;
	for(region_iter = coplanar_polys.begin(); region_iter != coplanar_polys.end(); ++region_iter)
	{
		std::vector<Polygon<VertexT, NormalT> > polygons = region_iter->getPolygons();
		typename std::vector<Polygon<VertexT, NormalT> >::iterator poly_iter;
		for(poly_iter = polygons.begin(); poly_iter != polygons.end(); ++poly_iter)
		{
			std::vector<VertexT> points = poly_iter->getVertices();
			typename std::vector<VertexT>::iterator point_iter;
			for(point_iter = points.begin(); point_iter != points.end(); ++point_iter)
			{
				centroid += (*point_iter);
				ransac_points.push_back(*point_iter);
			}
		}
	}

	// normalize centroid
	centroid /= ransac_points.size();

	// calc best fit plane with ransac
	akSurface akss;

	Plane<VertexT, NormalT> plane;
	bool ransac_success = true;
	plane = akss.calcPlaneRANSACfromPoints(ransac_points.at(0), ransac_points.size(), ransac_points, ransac_success);

	if (!ransac_success)
	{
		cout << timestamp << "UNABLE TO USE RANSAC FOR PLANE CREATION" << endl;
		return false;
	}

	float d = (plane.p.x * plane.n.x) + (plane.p.y * plane.n.y) + (plane.p.z * plane.n.z);

	// calc 2 points on this best fit plane, we need it for the transformation in 2D
	// TODO Hier aufpassen, wenn vec1 oder vec2 genau die Normalen ist, bzw parallel
	VertexT vec1(1, 2, 3);
	VertexT vec2(3, 2, 1);

	vec1 = vec1.cross(plane.n);
	vec2 = vec2.cross(plane.n);

	vec1 += plane.p;
	vec2 += plane.p;

	// calc transform
	Eigen::Matrix4f trans_mat;
	Eigen::Matrix4f trans_mat_inv;

std::cout << "***********Input von calcTransform: " << std::endl;
std::cout << plane.p;
std::cout << vec1;
std::cout << vec2;
	trans_mat = calcTransform(plane.p, vec1, vec2);

	// need transform from 3D to 2D and back from 2D to 3D
	trans_mat_inv = trans_mat.inverse();

	std::vector<BoostPolygon> input;
	std::vector<BoostPolygon> output;

std::cout << "**********PolygonRegion vor der Transformation*********" << std::endl;
	typename std::vector<PolyRegion>::iterator poly_iter;
	for(poly_iter = coplanar_polys.begin(); poly_iter != coplanar_polys.end(); ++poly_iter)
	{
		// transform and fuse
// debug output
		std::vector<Polygon<VertexT, NormalT>> points=  poly_iter->getPolygons();
		typename std::vector<Polygon<VertexT, NormalT>>::iterator its;
		cout << "Region hat " << points.size() << " Polygone" << std::endl;
		for(its = points.begin() ; its != points.end() ; ++its)
		{
			std::vector<cVertex> vert = its->getVertices();
			std::vector<cVertex>::iterator it2;
			cout << "Polygon hat " << vert.size() << " Punkte" << std::endl;
			for(it2 = vert.begin() ; it2 != vert.end() ; ++it2)
			{
				cout << "x: " << (*it2).x << "  y: " << (*it2).y << "  z: " << (*it2).z << endl;
			}
		}
//end debug
		input.push_back(transformto2DBoost((*poly_iter), trans_mat));
	}

// debug
	std::cout << "????????? BoostPolygone vor der union" << std::endl;
	int l = 0;
	BOOST_FOREACH(BoostPolygon const& p, input)
	{
	    std::cout << boost::geometry::wkt(p) << std::endl;
	    //std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
	    std::cout << "not union-polygon " << l++ << " has area " << boost::geometry::area(p) << std::endl;
	}
// debug ende

	// TODO Sonderfall abfangen, wenn Union "Fehlschlaegt", da man dann nicht weiß mit welchem
	// Polygon die Vereinigung gemacht werden soll
	bool first_it = true;
	typename std::vector<BoostPolygon>::iterator input_iter;
	for(input_iter = input.begin(); input_iter != input.end(); ++input_iter)
	{
		// transform and fuse
		if(first_it)
		{
			first_it = false;
			output.push_back((*input_iter));
		}
		else
		{
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
			}
			cout << "Boost should union them all" << endl;
			boost::geometry::union_(tmp, (*input_iter) , output);
		}

	}


// debug
	int i = 0;
    BOOST_FOREACH(BoostPolygon const& p, output)
    {
        std::cout << boost::geometry::wkt(p) << std::endl;
        //std::cout << i++ << ": " << boost::geometry::area(p) << std::endl;
        std::cout << "union-polygon " << i++ << " has area " << boost::geometry::area(p) << std::endl;
    }
// debug ende


	typename std::vector<BoostPolygon>::iterator output_iter;
	for(output_iter = output.begin(); output_iter != output.end(); ++output_iter)
	{
		std::cout << "Also im OutputVektor von fuse sollte was drinstehen" << std::endl;
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
	// TODO Transformation von 3D in 2D und Umwandlung von lvr::PolygonRegion Boost_Polygon
	// 		 Boost Polygon als Rückgabewert
	// TODO remove boost example code
std::cout << "transformto2DBoost aufgerufen" << std::endl;
	BoostPolygon result, tmp_poly;
	std::string poly_str;
	std::string res_poly_str;
	std::string tmp_str;
	std::string first_poly_str;
	bool first_it;
	bool first_poly = true;
	res_poly_str = "POLYGON(";

	// get all the polygons from this region
	std::vector<Polygon<VertexT, NormalT> > polygons = a.getPolygons();
	typename std::vector<Polygon<VertexT, NormalT> >::iterator poly_iter;
	// for all polygons in this region
	for(poly_iter = polygons.begin(); poly_iter != polygons.end(); ++poly_iter)
	{
		first_it = true;

		// get all vertices from this polygon
		std::vector<VertexT> points = poly_iter->getVertices();
		typename std::vector<VertexT>::iterator point_iter;
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

			float z = 	trans(2,0) * pt.coeffRef(0) +
						trans(2,1) * pt.coeffRef(1) +
						trans(2,2) * pt.coeffRef(2) +
						trans(2,3) * pt.coeffRef(3);
			std::cout << "******************* z-Wert: " << z << std::endl;

//			// Transformation
//			Eigen::Matrix4f tmp_mat;
//			for(int i = 0 ; i < 4 ; i++)
//			{
//				for(int j = 0 ; j < 4 ; j++)
//				{
//					tmp_mat(i, j) = 0;
//				}
//			}
//			tmp_mat(0,0) = (*point_iter).x;
//			tmp_mat(1,1) = (*point_iter).y;
//			tmp_mat(2,2) = (*point_iter).z;
//			tmp_mat(0,3) = 1;
//			tmp_mat(1,3) = 1;
//			tmp_mat(2,3) = 1;
//			tmp_mat(3,3) = 1;
//
//			//transform point in 2D
//			tmp_mat = tmp_mat * trans;
//
//std::cout << "In transformto2DBoost ist der transformierte Vektor bzw. Matrix: " << std::endl;
//std::cout << tmp_mat << std::endl;
//
//			float x,y;
//			x = tmp_mat(0,0);
//			y = tmp_mat(1,1);

			// transform in BoostPolygon
			if (first_it)
			{
				// save the first one, for closing the polygon
				first_poly_str.append(std::to_string(x));
				first_poly_str.append(" ");
				first_poly_str.append(std::to_string(y));
				first_it = false;

				poly_str.append("(");
				poly_str.append(std::to_string(x));
				poly_str.append(" ");
				poly_str.append(std::to_string(y));
				poly_str.append(", ");
			}
			else
			{
				poly_str.append(std::to_string(x));
				poly_str.append(" ");
				poly_str.append(std::to_string(y));
				poly_str.append(", ");
			}
		}
		poly_str.append(first_poly_str);
		poly_str.append(")");

		// check every single polygon, if it conform to the boost-polygon-style
		std::string test_poly_str = "POLYGON(";
		test_poly_str.append(poly_str);
		test_poly_str.append(")");
std::cout << "TestPolygon: " << test_poly_str << std::endl;
		boost::geometry::read_wkt(test_poly_str, tmp_poly);

		// if it is positive, do the polygon it the other direction (boost polygon style)
		if(first_poly)
		{
			first_poly = false;
			if(boost::geometry::area(tmp_poly) <= 0)
			{
				first_poly_str.clear(),
				poly_str.clear();
				std::cout << "falsche Richtung bei ersten Polygon, da area:" << boost::geometry::area(tmp_poly) << std::endl;
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

					//std::cout << "In transformto2DBoost (in if(poly_first)) ist der transformierte Vektor bzw. Matrix: " << std::endl;
					//std::cout << tmp_mat << std::endl;

					// transform in BoostPolygon
					if (first_it)
					{
						// save the first one, for closing the polygon
						first_poly_str.append(std::to_string(x));
						first_poly_str.append(" ");
						first_poly_str.append(std::to_string(y));
						first_it = false;

						poly_str.append("(");
						poly_str.append(std::to_string(x));
						poly_str.append(" ");
						poly_str.append(std::to_string(y));
						poly_str.append(", ");
					}
					else
					{
						poly_str.append(std::to_string(x));
						poly_str.append(" ");
						poly_str.append(std::to_string(y));
						poly_str.append(", ");
					}
				}
				poly_str.append(first_poly_str);
				poly_str.append(")");
			}
		}
		else if(boost::geometry::area(tmp_poly) >= 0 )
		{
			first_poly_str.clear(),
			poly_str.clear();
			std::cout << "andere Richtung bei inneren" << std::endl;
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

				// transform in BoostPolygon
				if (first_it)
				{
					// save the first one, for closing the polygon
					first_poly_str.append(std::to_string(x));
					first_poly_str.append(" ");
					first_poly_str.append(std::to_string(y));
					first_it = false;

					poly_str.append("(");
					poly_str.append(std::to_string(x));
					poly_str.append(" ");
					poly_str.append(std::to_string(y));
					poly_str.append(", ");
				}
				else
				{
					poly_str.append(std::to_string(x));
					poly_str.append(" ");
					poly_str.append(std::to_string(y));
					poly_str.append(", ");
				}
			}
			poly_str.append(first_poly_str);
			poly_str.append(")");
		}
		// now, this "part" of the polygon can be added to the complete polygon
		res_poly_str.append(poly_str);
	}

	res_poly_str.append(")");


	std::cout << "Genau vorm Erstellen des Polygons mit string: " << res_poly_str << std::endl;
	boost::geometry::read_wkt(res_poly_str, result);
	std::cout << "danach" << std::endl;

	std::cout << "transformto2DBoost durchgelaufen" << std::endl;
	return result;
}

template<typename VertexT, typename NormalT>
PolygonRegion<VertexT, NormalT> PolygonFusion<VertexT, NormalT>::transformto3Dlvr(BoostPolygon poly, Eigen::Matrix4f trans){
	// TODO Transformation von 2D in 3D und Umwandlung von Boost_Polygon in lvr::PolygonRegion
	std::cout << "transformto3Dlvr aufgerufgen" << std::endl;
    typedef boost::geometry::model::d2::point_xy<float> point;
    using boost::geometry::get;

    // store all the points in vec
    std::vector<point> vec;
    boost::geometry::for_each_point(poly, round_coordinates<point>(&vec));

    std::vector<Polygon<VertexT, NormalT>> poly_vec;
    std::vector<VertexT> point_vec;
    double f_x, f_y;
    bool first_p = true;
    std::vector<point>::iterator point_iter;
    for(point_iter = vec.begin() ; point_iter != vec.end() ; ++point_iter)
    {
    	// to determine every single polygon (contour, hole etc.)
    	if (first_p)
    	{

			Eigen::Matrix<double, 4, 1> pt(get<0>((*point_iter)), get<1>((*point_iter)), 0, 1);

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
//
//			// Transformation
//			Eigen::Matrix4f tmp_mat;
//			for(int i = 0 ; i < 4 ; i++)
//			{
//				for(int j = 0 ; j < 4 ; j++)
//				{
//					tmp_mat(i, j) = 0;
//				}
//			}
//			//TODO z wird also bei der Transformation auf Null projiziert?!
//			tmp_mat(0,0) = f_x;
//			tmp_mat(1,1) = f_y;
//			tmp_mat(2,2) = 0;
//			tmp_mat(0,3) = 1;
//			tmp_mat(1,3) = 1;
//			tmp_mat(2,3) = 1;
//			tmp_mat(3,3) = 1;
//
//			tmp_mat = tmp_mat * trans;
//
//			// store the point
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
    			std::cout << "Ein Polygon ist vollstaendig, da letzter und ersten Punkt gleich sind" << std::endl;
    			Polygon<VertexT, NormalT> bla(point_vec);
    			poly_vec.push_back(bla);
    			point_vec.clear();
    			first_p = true;
    		}
    		else
    		{
    			Eigen::Matrix<double, 4, 1> pt(get<0>((*point_iter)), get<1>((*point_iter)), 0, 1);

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
    	}
    }

	std::cout << "transformto3Dlvr durchgelaufen mit Rueckgabe von Vec mit size: " << poly_vec.size() << std::endl;
    //TODO hier muessen noch das Label und die normale zur Verfügung stehen
    std::string label = "noch_keins_da";
    NormalT normal;
    PolyRegion result(poly_vec, label, normal);
    return result;
}


template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::reset()
{
	m_polyregionmap.clear();
	m_meshes.clear();
}

} // Ende of namespace lvr

