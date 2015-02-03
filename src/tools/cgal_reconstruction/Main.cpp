/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

#include "io/Timestamp.hpp"
#include "io/ModelFactory.hpp"

using namespace lvr;

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/property_map.h>

#include <utility> // defines std::pair
#include <list>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;

typedef std::pair<Point, Vector> PointVectorPair;

int main(int argc, char** argv)
{

	// Read given file
	ModelPtr model = ModelFactory::readModel(argv[1]);

	std::list<PointVectorPair> points;

	std::cout << timestamp <<  "Creating model.." << std::endl;

	// Get point buffer and convert to
	size_t n;
	floatArr b = model->m_pointCloud->getPointArray(n);
	for(size_t i = 0; i < n; i++)
	{
		points.push_back(PointVectorPair(Point(b[3 * i], b[3 * i + 1], b[3 * i + 2]), Vector(0.0, 0.0, 0.0)));
	}

	std::cout << timestamp << "Estimating normals..." << std::endl;

	// Estimates normals direction.
	// Note: pca_estimate_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.


	const int nb_neighbors = atoi(argv[2]); // K-nearest neighbors = 3 rings
	CGAL::pca_estimate_normals(points.begin(), points.end(),
			CGAL::First_of_pair_property_map<PointVectorPair>(),
			CGAL::Second_of_pair_property_map<PointVectorPair>(),
			nb_neighbors);

	std::cout << timestamp << "Orientating normals..." << std::endl;

	// Orients normals.
	// Note: mst_orient_normals() requires an iterator over points
	// as well as property maps to access each point's position and normal.
	std::list<PointVectorPair>::iterator unoriented_points_begin =
			CGAL::mst_orient_normals(points.begin(), points.end(),
					CGAL::First_of_pair_property_map<PointVectorPair>(),
					CGAL::Second_of_pair_property_map<PointVectorPair>(),
					nb_neighbors);

	// Optional: delete points with an unoriented normal
	// if you plan to call a reconstruction algorithm that expects oriented normals.
	points.erase(unoriented_points_begin, points.end());

	// Optional: after erase(), use Scott Meyer's "swap trick" to trim excess capacity
	std::list<PointVectorPair>(points).swap(points);

	std::cout << timestamp << "Creating output buffer" << std::endl;

	PointBufferPtr buffer = PointBufferPtr(new PointBuffer);
	std::list<PointVectorPair>::iterator it;
	floatArr pts(new float[3 * points.size()]);
	floatArr normals(new float[3 * points.size()]);
	int c = 0;
	for(it = points.begin(); it != points.end(); it++)
	{
		Point p = it->first;
		Vector n = it->second;
		pts[3 * c    ] = p[0];
		pts[3 * c + 1] = p[1];
		pts[3 * c + 2] = p[2];

		normals[3 * c    ] = n[0];
		normals[3 * c + 1] = n[1];
		normals[3 * c + 2] = n[2];
		//std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
		c++;
	}
	//std::cout << c << " & " << points.size() << std::endl;

	std::cout << timestamp << "Creating model" << std::endl;
	buffer->setPointArray(pts, points.size());
	buffer->setPointNormalArray(normals, points.size());

	ModelPtr outModel(new Model(buffer));


	std::cout << timestamp << "Saving result" << std::endl;
	ModelFactory::saveModel(outModel, "normals.ply");

}
