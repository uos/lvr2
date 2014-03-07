/*
 * PolygonFusion.tcc
 *
 *  Created on: 05.03.2014
 *      Author: dofeldsc
 */

#include "PolygonFusion.hpp"

namespace lvr
{

template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::PolygonFusion() {
	// TODO Auto-generated constructor stub
	m_distance_threshold = 0.05;
}


template<typename VertexT, typename NormalT>
PolygonFusion<VertexT, NormalT>::~PolygonFusion() {
	// TODO Auto-generated destructor stub
}


template<typename VertexT, typename NormalT>
void PolygonFusion<VertexT, NormalT>::addFusionMesh(PolygonMesh<VertexT, NormalT> mesh) {

}


template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::doFusion()
{
	/* To-Do Umbauen auf neue Struktur
	ROS_INFO("polygonFusion!!");

	// 0.5) prepare map and other vectors
	// 1) put polyregions into bins according to labels
	// 2) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// 3) transform these polygons into 2D space (see spuetz fusion)
	// 4) apply boost::geometry::union_ for these polygons
	// 5) transform resulting 2D polygon back into 3d space (inverse of step 3)
	// 6) place resulting 3D polygon in response.mesh
	// 7) insert all left overs into response.mesh

	// step 0.5)
	typedef std::map<std::string, std::vector<lvr_tools::PolygonRegion> > polyRegionMap;
	polyRegionMap polygonsByRegion;

	// step 1) put polyregions into bins according to labels
	// TODO unknown regions!!
	std::vector<lvr_tools::PolygonMesh>::iterator polymesh_iter;
	for( polymesh_iter = meshes.begin(); polymesh_iter != meshes.end(); ++polymesh_iter )
	{
		std::vector<lvr_tools::PolygonRegion>::iterator polyregion_iter;
		for( polyregion_iter = (*polymesh_iter).polyregions.begin(); polyregion_iter != (*polymesh_iter).polyregions.end(); ++polyregion_iter )
		{
			if ( (*polyregion_iter).label != "unknown" )
			{
				// if prelabel already exists in map, just push back PolyGroup, else create
				polyRegionMap::iterator it = polygonsByRegion.find((*polyregion_iter).label);

				if (it != polygonsByRegion.end())
				{
					it->second.insert(it->second.end(), (*polyregion_iter));
				}
				else
				{
					std::vector<lvr_tools::PolygonRegion> regions;
					regions.push_back((*polyregion_iter));
					polygonsByRegion.insert(std::pair<std::string, std::vector<lvr_tools::PolygonRegion> >((*polyregion_iter).label, regions));
				}
			}
		}
	}

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

	// step 2-5) in these bins, find "co-planar" polyregions -> same plane (Δ)
	// TODO fix coplanar detection
	// TODO benchmark coplanar threshold and fusion / detection order (not only first one)
	polyRegionMap::iterator map_iter;
	for( map_iter = polygonsByRegion.begin(); map_iter != polygonsByRegion.end(); ++map_iter )
	{
		ROS_WARN("trying to fuse polygons with regionlabel %s", (*map_iter).first.c_str());

		std::vector<lvr_tools::PolygonRegion> polyregions = (*map_iter).second;

		std::vector<lvr_tools::PolygonRegion> coplanar_regions;
		std::vector<lvr_tools::PolygonRegion> nonplanar_regions;
		std::vector<lvr_tools::PolygonRegion> fused_regions;

		std::vector<lvr_tools::PolygonRegion>::iterator region_iter;
		for( region_iter = polyregions.begin(); region_iter != polyregions.end(); )
		{
			ROS_ERROR("still need to process %d outer PolygonRegions", polyregions.size());
			// assume there exists at least least one coplanar region
			coplanar_regions.push_back((*region_iter));

//			if ( polyfusion_first_publish )
//			{
//				lvr_tools::PolygonMesh pm;
//				pm.header.frame_id = (*region_iter).header.frame_id;
//				pm.header.stamp = (*region_iter).header.stamp;
//				pm.polyregions.push_back()
//			}

			std::vector<lvr_tools::PolygonRegion>::iterator coplanar_iter;
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

			// assumption was wrong, no coplanar region
			if ( coplanar_regions.size() == 1 )
			{
				nonplanar_regions.push_back((*region_iter));
				coplanar_regions.clear();
			}
			// assumption was correct, need to do fusion
			else
			{
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
	ROS_ERROR("NOT YET IMPLEMENTED");
	*/
	return false;
}




template<typename VertexT, typename NormalT>
bool PolygonFusion<VertexT, NormalT>::isPlanar(Polyregion a, Polyregion b)
{
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

	float d = (n_x * p1_x + n_y * p1_y + n_z * p1_z) / sqrt( n_x * n_x + n_y * n_y + n_z * n_z );
	float distance = 0.0;

	std::vector<Polygon<VertexT, NormalT>> polygons_b;
	polygons_b = b.getPolygons();


// Frage: Wir betrachten hier nur das äußere Polygon, reicht das? Also ich glaub schon, da die ja eh auf einer Ebene liegen sollten
	typename std::vector<Polygon<VertexT, NormalT>>::iterator point_iter;
	for( point_iter = polygons_b.begin(); coplanar != false, point_iter != polygons_b.end(); ++point_iter )
	{
		distance = abs( ( ( n_x * (*point_iter).x ) + ( n_y  *  (*point_iter).y ) + ( n_z  *  (*point_iter).z ) + d ) ) / sqrt( n_x * n_x + n_y * n_y + n_z * n_z );
		if ( distance > m_distance_threshold )
		{
			coplanar = false;
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

	return coplanar;
}

} // Ende of namespace lvr

