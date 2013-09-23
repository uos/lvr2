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

/*
 *  MeshSlicer.tcc
 *
 *  @date 21.08.2013
 *  @author Henning Deeken (hdeeken@uos.de)
 *  @author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

namespace lvr
{

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Vector_3 Vector;
typedef K::Plane_3 Plane;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef K::Segment_3 Segment; 
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef Tree::Object_and_primitive_id Object_and_primitive_id;
typedef Tree::Primitive_id Primitive_id;

///
/// Mesh Construction Methods
///

MeshSlicer::MeshSlicer()
{
   clear();
}

void MeshSlicer::addMesh(MeshBufferPtr mesh)
{
	clear();
	
	size_t num_verts;
	size_t num_faces;
	
	floatArr vert_tmp = mesh->getVertexArray(num_verts);
	uintArr face_tmp = mesh->getFaceArray(num_faces);
	
	for(size_t i = 0; i < num_verts; i++)
	{
		vertices.push_back(vert_tmp[3 * i]);
		vertices.push_back(vert_tmp[3 * i + 1]); 
		vertices.push_back(vert_tmp[3 * i + 2]);
	}
	
	for(size_t i = 0; i < num_faces; i++)
	{
		faces.push_back(face_tmp[3 * i]);
		faces.push_back(face_tmp[3 * i + 1]); 
		faces.push_back(face_tmp[3 * i + 2]);
	}
}

vector<float> MeshSlicer::addMeshAndCompute2dSlice(MeshBufferPtr mesh)
{
	addMesh(mesh);
	return compute2dSlice();
}

vector<float> MeshSlicer::addMeshAndCompute2dProjection(MeshBufferPtr mesh)
{
	addMesh(mesh);
	return compute2dProjection();
}

///
/// Integration Methods
///

void MeshSlicer::clear()
{
	vertices.clear();
	faces.clear();
	output.clear();
}

vector<float> MeshSlicer::compute2dSlice()
{
	buildTree(); 
	computeIntersections(segments);
	return output;
}

vector<float> MeshSlicer::compute2dProjection()
{
	buildTree(); 
	computeProjections(segments);
	return output;
}

/// AABB Tree Operations

void MeshSlicer::buildTree()
{	
	std::list<Triangle> triangles;	
	
	if(faces.size() > 0)
	{
		for(size_t i = 0; i < faces.size(); i+=3)
		{
			int vertex_ind_1 = faces.at(i);
			int vertex_ind_2 = faces.at(i+1);
			int vertex_ind_3 = faces.at(i+2);
			
			Point a(vertices.at(vertex_ind_1*3), vertices.at(vertex_ind_1*3+1), vertices.at(vertex_ind_1*3+2));
			Point b(vertices.at(vertex_ind_2*3), vertices.at(vertex_ind_2*3+1), vertices.at(vertex_ind_2*3+2));
			Point c(vertices.at(vertex_ind_3*3), vertices.at(vertex_ind_3*3+1), vertices.at(vertex_ind_3*3+2));
			triangles.push_back(Triangle(a,b,c));
		}

		tree.clear();
		tree.insert(triangles.begin(), triangles.end());
	}
}

Plane MeshSlicer::getQueryPlane(string dimension, double value)
{
	Point  p;
	
		if(dimension.compare("x") == 0)
		{
			coord_x = 1.0;
			coord_y = 0.0;
			coord_z = 0.0;
			p = Point(value,0.0, 0.0);
		}
		else if(dimension.compare("y") == 0)
		{
			coord_x = 0;
			coord_y = 1;
			coord_z = 0;
			p = Point(0.0, value,0.0);
		}
		else if(dimension.compare("z") == 0)
		{
			coord_x = 0;
			coord_y = 0;
			coord_z = 1;
			p = Point(0.0, 0.0, value);
		}
		else
		{
			cout << "ERROR: Could not set dimension." << endl;
		}
	
	Vector v(coord_x, coord_y, coord_z);
	
	// introduces a query plane that passes through point p and that is orthogonal to v. 	
    Plane  plane_query(p, v);
    
    //TODO make output optional
    cout << endl << "Query Plane :" << endl; 
	cout << "Dimension   : " << dimension.c_str() << endl;
	cout << "Value       : " << value << endl;
	cout << "Plane Point : "  << p <<  endl;
	cout << "Plane Normal: "  << " (" << coord_x << ", " << coord_y << ", " << coord_z << ")" << endl <<endl;
	
	return plane_query;
}

void MeshSlicer::computeIntersections(vector<Segment>& segments)
{	
	//cout << timestamp << "Start Computing Intersections... " << endl <<endl;

	Plane  plane_query = getQueryPlane(dimension, value);

	try
	{
		cout << "Found " << tree.number_of_intersected_primitives(plane_query) << "intersections(s) with plane: " << plane_query << endl;
	}
	catch(CGAL::Precondition_exception e)
	{
		cout << "ERROR"<< endl;
		cout << e.what() << endl;
	}

	// computes all intersections with segment query (as pairs object - primitive_id)
	std::list<Object_and_primitive_id> intersections;

	cout << "Calculating intersection segments...";
	tree.all_intersections(plane_query, std::back_inserter(intersections));
	//cout << "There are " << intersections.size() << "intersections" << endl;
 	for (std::list<Object_and_primitive_id>::iterator it = intersections.begin(); it != intersections.end(); it++)
	{
		Object_and_primitive_id op = *it;
		CGAL::Object object = op.first;
		Segment segment;

		if(CGAL::assign(segment,object))
		{
			segments.push_back(segment);		
			
			output.push_back((float) segment.source().x());
			output.push_back((float) segment.source().y());
			output.push_back((float) segment.source().z());
			
			output.push_back((float) segment.target().x());
			output.push_back((float) segment.target().y());
			output.push_back((float) segment.target().z());
		
			//cout << "Adding Segment from: " << ((float) segment.source().x()) << ", " << ((float) segment.source().y()) << ", " << ((float) segment.source().z()) << " to " << ((float) segment.target().x()) << ", " <<  ((float) segment.target().y() )<< ", " << ((float) segment.target().z()) << end;
 		}
		else std::cout << "ERROR: intersection object is unknown" << std::endl; 
	}

	cout << timestamp << "Finished Computing " << output.size()/3 << " Intersections..." << endl;	
}

void MeshSlicer::computeProjections(vector<Segment>& segments)
{
	
	//cout << timestamp << "Start Computing Intersections... " << endl <<endl;
	
	double current_value = value;
	double increment = offset / number_of_slices;
	int i = 0; 
	while(current_value < (value + offset))
	{
		Plane  plane_query = getQueryPlane(dimension, current_value);

		try
		{
			cout << "Found " << tree.number_of_intersected_primitives(plane_query) << "intersections(s) with " << i  << "/" <<number_of_slices <<"th plane: " << plane_query << endl;
		}
		catch(CGAL::Precondition_exception e)
		{
			cout << "ERROR"<< endl;
			cout << e.what() << endl;
		}

		// computes all intersections with segment query (as pairs object - primitive_id)
		std::list<Object_and_primitive_id> intersections;

		// BETTER LIST ALL SLICED TRIANGLES AND PROJECT THEM AFTER WARDS
		tree.all_intersections(plane_query, std::back_inserter(intersections));
		for (std::list<Object_and_primitive_id>::iterator it = intersections.begin(); it != intersections.end(); it++)
		{
			Object_and_primitive_id op = *it;
			CGAL::Object object = op.first;
			Segment segment;

			if(CGAL::assign(segment,object))
			{
				segments.push_back(segment);
			
				output.push_back((float) segment.source().x());
				output.push_back((float) segment.source().y());
				output.push_back((float) segment.source().z());
			
				output.push_back((float) segment.target().x());
				output.push_back((float) segment.target().y());
				output.push_back((float) segment.target().z());
			}
			else std::cout << "ERROR: intersection object is unknown" << std::endl; 
		}
		
		current_value += increment;
		i++;
	}
	
	cout << timestamp << "Finished Computing " << output.size()/3 << " Intersections in total." << endl;
	
}

} // namespace lvr
