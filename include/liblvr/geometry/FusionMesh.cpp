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
 * FusionMesh.tcc
 *
 *  @date 13.11.2008
 *	@author Henning Deeken (hdeeken@uos.de)
 *	@author Ann-Katrin Häuser (ahaeuser@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */

namespace lvr
{

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K,Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
	
///
/// Mesh Construction Methods
///

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh()
{
   verbose = false;
   clearLocalBuffer();
   clearGlobalBuffer();
}

template<typename VertexT, typename NormalT> FusionMesh<VertexT, NormalT>::FusionMesh(MeshBufferPtr mesh)
{
   verbose = false;
   clearLocalBuffer();
   clearGlobalBuffer();
   addMesh(mesh);
   integrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addVertex(VertexT v)
{
    // Create new FusionMeshVertex and increase vertex counter
    m_local_vertices.push_back(new FusionVertex<VertexT, NormalT>(v));    
    m_local_index++;
    //cout << "Adding Vertex - " << v << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addNormal(NormalT n)
{
    // Is a vertex exists at globalIndex, save normal
    assert(m_local_index == m_local_vertices.size());
    m_local_vertices[m_local_index - 1]->m_normal = n;
    
    //cout << "Adding Normal - " << n << endl;  
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c, FFace* &face)
{		
    // Create a new face
    face = new FFace;

	face->m_index[0] = a;
	face->m_index[1] = b;
	face->m_index[2] = c;
	
	FVertex* v0 = m_local_vertices[a];
	FVertex* v1 = m_local_vertices[b];
	FVertex* v2 = m_local_vertices[c];
	
	face->vertices[0] = v0;
	face->vertices[1] = v1;
	face->vertices[2] = v2;
   
    m_local_faces.push_back(face);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c)
{
	  //cout << "Adding Triangle..." << a << " " << b << " " << c << endl;
      FFace* face;
      addTriangle(a, b, c, face);
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMesh(MeshBufferPtr mesh)
{
    size_t num_verts, num_norms, num_faces;
    floatArr vertices = mesh->getVertexArray(num_verts);
    floatArr normals = mesh->getVertexNormalArray(num_norms);
    
    //if(num_norms != num_verts)
	//	cout << "Unequal number of vertices and normals" << endl;
    
    // Add all vertices
    for(size_t i = 0; i < num_verts; i++)
    {
        addVertex(VertexT(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]));
    }
    
    // Add all faces
    uintArr faces = mesh->getFaceArray(num_faces);
    for(size_t i = 0; i < num_faces; i++)
    {
        addTriangle(faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
    }
    
      // Add all vertex normals, in case we need that.
    /*
    
    for(size_t i = 0; i < num_norms; i++)
    {
         addNormal(NormalT(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]));
		
    } 
    */ 
}

///
/// Integration Methods
///

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearLocalBuffer()
{
	m_local_index = 0;
	m_local_vertices.clear();
	m_local_faces.clear();

}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::clearGlobalBuffer()
{
	m_global_index = 0;
	m_global_vertices.clear();
	m_global_faces.clear();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printLocalBufferStatus()
{
	cout << timestamp << "Local Buffer" << endl;
	cout << timestamp << "#Index         :" << m_local_index <<  endl;
	cout << timestamp << "#Vertices      :" << m_local_vertices.size() <<  endl;
	cout << timestamp << "#Faces         :" << m_local_faces.size() << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printGlobalBufferStatus()
{
	cout << timestamp << "Global Buffer" << endl;
	cout << timestamp << "#Index         :" << m_global_index <<  endl;
	cout << timestamp << "#Vertices      :" << m_global_vertices.size() <<  endl;
	cout << timestamp << "#Faces         :" << m_global_faces.size() << endl; 
}

// sloppy variant with possibly redundant vertices, to add properly make set checkup
template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalVertex(FVertex *v)
{
    // Create new FusionMeshVertex and increase vertex counter
    //v->m_self_index = m_global_index;
    //cout << "at insertion " << v->m_self_index << "-" << v->m_position << endl;
    
    m_global_vertices.push_back(v);
    
   // HIER wird auch der m_self_index des bereits vorhandenen Vertex im Global Buffer ersetzt
    m_global_vertices[m_global_index]->m_self_index = m_global_index;
   // cout << "after insertion " << v->m_self_index << "-" << v->m_position << endl;
  
   // cout << "in globale" <<  m_global_vertices[m_global_index]->m_self_index << "-" <<  m_global_vertices[m_global_index]->m_position << endl << endl;
  
    
    m_global_index++; // = m_global_vertices.size() - 1;   

   // cout << "Adding Global Vertex at global buffer position " << m_global_index <<  endl;
   // cout << "m_Self_index " << v->m_self_index <<  endl;

}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addGlobalTriangle(FFace *face, int increment)
{
	face->m_index[0] = face->m_index[0] + increment;
	face->m_index[1] = face->m_index[1] + increment;
	face->m_index[2] = face->m_index[2] + increment;

	m_global_faces.push_back(face);   
   // cout << "Adding Global Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::lazyIntegrate()
{
	if(verbose)
	{
		cout <<endl << timestamp << "Start Lazy Integration..." << endl << endl;
		//printLocalBufferStatus();
		//printGlobalBufferStatus();
	
	}
	
	size_t num_current_local_vertices  = m_local_vertices.size();
	size_t num_current_local_faces  = m_local_faces.size();
		
	size_t num_current_global_vertices = m_global_vertices.size();
	size_t num_current_global_faces = m_global_faces.size();

    // Add all vertices
    for(size_t i = 0; i < num_current_local_vertices; i++)
    {
        addGlobalVertex(m_local_vertices[i]);
    }
    
    // Add all faces
    for(size_t i = 0; i < num_current_local_faces; i++)
    {
		addGlobalTriangle(m_local_faces[i], num_current_global_vertices);
    }
    
    clearLocalBuffer();
    
    if(verbose)
    {
		cout << endl << timestamp << "Finished Lazy Integration..." << endl << endl;
	
		//printLocalBufferStatus();
		//printGlobalBufferStatus();
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::remoteIntegrate(vector<FFace*>& faces)
{	
	
	if(verbose)
    {
		cout << timestamp << " Start Remote Integrate..." << endl;
	}
	
	int degeneratedFaces = 0;
	MapIterator it;
	
	for(size_t i = 0; i < faces.size(); i++)
    {
		FFace* face = faces[i];
		
		for(int j = 0; j < 3; j++)
		{
			FVertex* v =  m_local_vertices[face->m_index[j]];
			std::pair<MapIterator,bool> const& r=global_vertices_map.insert(std::pair<VertexT, size_t>((VertexT)v->m_position, m_global_index));

				if (r.second) 
				{ 
					// && (global_vertices_map.count(v->m_position) == 1)) {
					// FEHLER: MANCHMAL WIRD NICHT ERKANNT DAS DIE VERTEX BEREITS IN DER MAP LIEGT
					addGlobalVertex(v);
					face->m_index[j] = v->m_self_index;
					face->vertices[j] = m_global_vertices[face->m_index[j]]; 
				}
				else
				{
					// value wasn't inserted because my_map[foo_obj] already existed.
					// note: the old value is available through r.first->second
					// and may not be "some value"
					 
					face->m_index[j] = r.first->second;
					
					if(face->m_index[j] >= m_global_vertices.size())
					{
					    cout << r.first->first << " " << r.first->second << endl;
						cout << "error: " <<  face->m_index[j] << " >=  " << m_global_vertices.size() << endl;
					}
					else {
						face->vertices[j] = m_global_vertices[face->m_index[j]];
					}
				}
		}
		//check for degenerated faces
		if (face->m_index[0] == face->m_index[1] || face->m_index[0] == face->m_index[2] || face->m_index[1] == face->m_index[2]) 
		{
			degeneratedFaces++;
		}
		else
		{
			m_global_faces.push_back(face);
		}
    }
    
    if(verbose)
    {
		cout << "Skipped " << degeneratedFaces << " Faces due to degeneration" << endl;
		cout << timestamp << " Finished Remote Integrate" << endl;
	}
}

/*template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addFacesToVertices()
{
	// currently only operating on elements in local Buffer
	for(size_t i = 0; i < m_local_faces.size(); i++)
    {
		FFace* face = m_local_faces[i];
		for(int j = 0; j < 3; j++)
		{
			FVertex* v =  m_local_vertices[face->m_index[j]];
			v->m_face_indices.push_back(i);
		}
	}
}*/

/*template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::sortClippingPoints(vector<const Point*> points, Point p, Point q, vector<const Point*>& sorted_points)
{
	sorted_points.clear();
	if (points.size() < 2) {
		sorted_points = points;
	}
	else {
		//cout << "points size " << points.size() << endl;
		//cout << "p: " << p.x() << ", " << p.y() << ", " << p.z() << endl;
		//cout << "q: " << q.x() << ", " << q.y() << ", " << q.z() << endl;
		//Segment seg1 = Segment(p, q);
		//FT dist1 = seg1.squared_length();
		//cout << "dist p q: " << dist1 << endl;
		while (points.size()>1) {
			FT min_dist = 9999999;
			int ind = 99;
			//find closest point to p
			for (int i = 0; i < points.size(); i++) {
				Point temp(points[i]->x(), points[i]->y(), points[i]->z());
				Segment seg = Segment(p, temp);
				FT dist = seg.squared_length();
				if (dist < min_dist) {
					min_dist = dist;
					ind = i;
				}
			}
			sorted_points.push_back(points[ind]);
			points.erase(points.begin()+ind);
		}
		sorted_points.push_back(points[0]);
		/*cout << "distances of sorted points: ";
		for (int i=0; i < sorted_points.size(); i++) {
			Point temp(sorted_points[i]->x(), sorted_points[i]->y(), sorted_points[i]->z());
			Segment seg = Segment(p, temp);
			FT dist = seg.squared_length();
			cout << dist << ", ";
		}
	cout << endl;	
	}
}*/

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::intersectIntegrate(vector<FFace*>& faces)
{
	if(verbose)
	{
		cout << timestamp << " Start Intersect Integrate..." << endl;
	}
	//addFacesToVertices();
	
	/*
	before Delauny try
	vector<int> outer_vertex; //face index of vertices that will be replaced
	vector<int> inner_vertex; //face index of vertices that will be kept
	vector<Point> old_vertex_pos;
	vector<const Point*> intersect_points;
	vector<const Segment*> intersect_segments;
	list<Object_and_primitive_id> intersections;
	vector<FFace*> faces_to_add;
	int count_changed_faces = 0;
	int count_new_faces = 0;
	bool add = false; 
	*/
	
	vector<Point> new_vertex_pos;
	vector<Point> tri_points;
	vector<const Segment*> intersect_segments;
	list<Object_and_primitive_id> intersections;
	vector<FFace*> new_faces;
	
	for(size_t i = 0; i < faces.size(); i++)
    { 
		FFace* face = faces[i];
		tri_points.clear();
		for (int j = 0; j < 3; j++) {
			FVertex* v = m_local_vertices[face->m_index[j]];
			Point a(v->m_position.x, v->m_position.y, v->m_position.z);
			FT dist = tree.squared_distance(a);
			tri_points.push_back(a);
			//check wether this vertex will be kept
			if (dist > threshold) {
				new_vertex_pos.push_back(a);
			}
		}
		//determine intersection points and add to new_vertex_pos
		Triangle tri = Triangle(tri_points[0], tri_points[1], tri_points[2]);
		intersections.clear();
		intersect_segments.clear();
		try {
				tree.all_intersections(tri, back_inserter(intersections));
		} catch (...)
		{
				cout << "tree.do_intersect() fails" << endl;
		}
		while (!intersections.empty()) {
			Object_and_primitive_id op = intersections.front();
			intersections.pop_front();
			CGAL::Object object = op.first;
			//check wether intersection object is a segment
			if (const Segment* s = CGAL::object_cast<Segment>(&object)){
				intersect_segments.push_back(s);
			}
			else if (const Point* p = CGAL::object_cast<Point>(&object)){
				cout << "intersection is a point not a segment" << endl;
			}
		}
		for (int j = 0; j < intersect_segments.size(); j++) {
			const Segment* seg = intersect_segments[j];
			Point p = seg->source();
			new_vertex_pos.push_back(p);
			p = seg->target();
			new_vertex_pos.push_back(p);
		}
		/* 
		before Delauny try
		FFace* face = faces[i];
		outer_vertex.clear();
		inner_vertex.clear();
		old_vertex_pos.clear();
		for(int j = 0; j < 3; j++)
		{
			FVertex* v =  m_local_vertices[face->m_index[j]];
			Point a(v->m_position.x, v->m_position.y, v->m_position.z);
			old_vertex_pos.push_back(a);
			FT dist = tree.squared_distance(a);
			//check wether this vertex lies on the boundary
			if (dist <= threshold) {
				outer_vertex.push_back(j);
			}
			else {
				inner_vertex.push_back(j);
			}
		}
		Triangle tri = Triangle(old_vertex_pos[0], old_vertex_pos[1], old_vertex_pos[2]);
		intersections.clear();
		intersect_segments.clear();
		//find all intersections for current triangle
		try {
				tree.all_intersections(tri, back_inserter(intersections));
		} catch (...)
		{
				cout << "tree.do_intersect() fails" << endl;
		}
		while (!intersections.empty()) {
			Object_and_primitive_id op = intersections.front();
			intersections.pop_front();
			CGAL::Object object = op.first;
			//check wether intersection object is a segment
			if (const Segment* s = CGAL::object_cast<Segment>(&object)){
				intersect_segments.push_back(s);
			}
			else if (const Point* p = CGAL::object_cast<Point>(&object)){
				cout << "intersection is a point not a segment" << endl;
			}
		}
		if (intersect_segments.size() < 1) {
			cout << "no intersection found, wrong sorting" << endl;
		}
		else {
			// one vertex will keep its position
			if (inner_vertex.size() == 1) {
				for (int j = 0; j < 1; j++) {
					const Segment* seg = intersect_segments[j];
					Segment temp = Segment(seg->source(), old_vertex_pos[outer_vertex[0]]);
					Segment temp1 = Segment(seg->target(), old_vertex_pos[outer_vertex[0]]);
					if (temp.squared_length() < temp1.squared_length()) {
						//change outer_vertex[0] position to seg.source
						FVertex* v =  m_local_vertices[face->m_index[outer_vertex[0]]];
						v->m_position.x = seg->source().x();
						v->m_position.y = seg->source().y();
						v->m_position.z = seg->source().z();
						v = m_local_vertices[face->m_index[outer_vertex[1]]];
						v->m_position.x = seg->target().x();
						v->m_position.y = seg->target().y();
						v->m_position.z = seg->target().z();
						cout << seg->vertex(1) << seg->vertex(2) << endl;
					}
					else {
						//change outer_vertex[0] position to seg.target
						FVertex* v =  m_local_vertices[face->m_index[outer_vertex[1]]];
						v->m_position.x = seg->source().x();
						v->m_position.y = seg->source().y();
						v->m_position.z = seg->source().z();
						v = m_local_vertices[face->m_index[outer_vertex[0]]];
						v->m_position.x = seg->target().x();
						v->m_position.y = seg->target().y();
						v->m_position.z = seg->target().z();	
					}
				}
				count_changed_faces++;
				faces_to_add.push_back(face); 
			}
			// two vertices will keep their position
			else if (inner_vertex.size() == 2) {
			}
		}	*/
			/*} //very old
			//create new faces by dividing current face
			//first update face thats already in local buffer
			cout << "local index before change " << m_local_index << endl;
			addVertex(VertexT(sorted_points[0]->x(), sorted_points[0]->y(), sorted_points[0]->z()));
			face->m_index[bound_vertex[1]] = m_local_index-1;
			cout << "local index after change "<< m_local_index << endl;
			face->r = 200;
			face->g = 200;
			face->b = 200;
			count_changed_faces++;
			// from now on add new faces to local buffer
			for (int l = 1; l < sorted_points.size(); l++) {
				addTriangle(m_local_index-1, m_local_index, face->m_index[inner_vertex]);
				addVertex(VertexT(sorted_points[l]->x(), sorted_points[l]->y(), sorted_points[l]->z()));
				// push back face so that it can be integrated into global buffer
				faces.push_back(m_local_faces[m_local_faces.size()-1]);
				count_new_faces++;
				FFace* colorFace = m_local_faces[m_local_faces.size()-1];
				colorFace->r = 200;
				colorFace->g = 0;
				colorFace->b = 0;
			}
			*/
	}
	//Delauny Triangulation
	/*
	vector<Point>::iterator begin;
	vector<Point>::iterator end;
	begin = new_vertex_pos.begin();
	end = new_vertex_pos.end();
	Delaunay dt(begin,end);
	*/
	Delaunay dt;
	for (int i = 0; i < new_vertex_pos.size(); i++) {
		Point2 p(new_vertex_pos[i].x(), new_vertex_pos[i].y(), new_vertex_pos[i].z());
		dt.push_back(p);
	}
	//add vertices and faces to local buffer
	Delaunay::Finite_faces_iterator it;	
	for (it = dt.finite_faces_begin(); it != dt.finite_faces_end(); it++)
	{
		Triangle2 tri = dt.triangle(it);
		int count_to_close_vertices = 0;
		for (int i = 0; i < 3; i++) {
			Point2 p = tri.vertex(i);
			Point a(p.x(), p.y(), p.z());
			FT dist = tree.squared_distance(a);
			if (dist <= threshold) {
				count_to_close_vertices++;
			}
			addVertex(VertexT(p.x(), p.y(), p.z()));
		}
		if (count_to_close_vertices < 3) {
			addTriangle(m_local_index-3, m_local_index-2, m_local_index-1);
			new_faces.push_back(m_local_faces[m_local_faces.size()-1]);
			FFace* colorFace = m_local_faces[m_local_faces.size()-1];
			cout << m_local_vertices[colorFace->m_index[0]]->m_position << m_local_vertices[colorFace->m_index[1]]->m_position << m_local_vertices[colorFace->m_index[2]]->m_position << endl;
				colorFace->r = 200;
				colorFace->g = 0;
				colorFace->b = 0;
		}
	}
	remoteIntegrate(new_faces);
	
		//OutputltFaces fit;
		//Face_handle start;
		//dt.get_conflicts(p, fit, start);
	
	//cout << "changed " << count_changed_faces << " faces" << endl;
	//cout << "added " << count_new_faces << " faces" << endl;
	
	//remoteIntegrate(faces_to_add);
	if(verbose)
	{
		cout << timestamp << " Finished Intersect Integrate ..." << endl;
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::buildTree()
{
	size_t num_current_global_vertices = m_global_vertices.size();
	size_t num_current_global_faces = m_global_faces.size();

	std::list<Triangle> triangles;	

	if(num_current_global_faces > 0)
	{
		for(size_t i = 0; i < num_current_global_faces; i++)
		{
			FVertex* v0 = m_global_vertices[m_global_faces[i]->m_index[0]];
			FVertex* v1 = m_global_vertices[m_global_faces[i]->m_index[1]];
			FVertex* v2 = m_global_vertices[m_global_faces[i]->m_index[2]];
			
			Point a(v0->m_position.x, v0->m_position.y, v0->m_position.z);
			Point b(v1->m_position.x, v1->m_position.y, v1->m_position.z);
			Point c(v2->m_position.x, v2->m_position.y, v2->m_position.z);
			
			triangles.push_back(Triangle(a,b,c));
		}
		
		tree.clear();
		tree.insert(triangles.begin(), triangles.end());
		
		bool result = tree.accelerate_distance_queries();
		if (result && verbose) 
		{
			cout << "successfully accelerated_distance_queries" << endl; 
		}
		else
		{
			cout << "could not accelerate aabb tree" << endl;
		}
	
	}	
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::buildVertexMap()
{
	size_t num_current_global_vertices = m_global_vertices.size();
	
	if(num_current_global_vertices > 0)
	{
		global_vertices_map.clear();
		for(size_t i = 0; i < num_current_global_vertices; i++)
		{
			global_vertices_map.insert(std::pair<VertexT, size_t>(m_global_vertices[i]->m_position, i));
		}
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::sortFaces(vector<FFace*>& remote_faces, vector<FFace*>& intersection_faces, vector<FFace*>& closeby_faces)
{	
	if(verbose)
	{
		cout << timestamp << "Start Sorting Faces... " << endl <<endl;
	
		cout << "Distance Threshold: " << sqrt(threshold) << endl;
		cout << "Squared Distance Threshold: " << threshold << endl;
	}
	
	redundant_faces = 0;
	special_case_faces = 0;
	int far_tree_intersect_fails = 0;
	int close_tree_intersect_fails = 0;
	
	FFace* face;
	FVertex* v0;
	FVertex* v1;
	FVertex* v2;
	FT dist_a, dist_b, dist_c;
	Triangle temp;
		
	for(size_t i = 0; i < m_local_faces.size(); i++)
	{
		face = m_local_faces[i];

		v0 = m_local_vertices[face->m_index[0]];
		v1 = m_local_vertices[face->m_index[1]];
		v2 = m_local_vertices[face->m_index[2]];
		
		Point a(v0->m_position.x, v0->m_position.y, v0->m_position.z);
		Point b(v1->m_position.x, v1->m_position.y, v1->m_position.z);
		Point c(v2->m_position.x, v2->m_position.y, v2->m_position.z);
		
		dist_a = tree.squared_distance(a);
		dist_b = tree.squared_distance(b);
		dist_c = tree.squared_distance(c);
		
		temp = Triangle(a,b,c);
		
		if (dist_a > threshold && dist_b > threshold && dist_c > threshold)
		{
			bool result = true;
			try {
				result = tree.do_intersect(temp);
			} catch (...)
		    {
				far_tree_intersect_fails++;
				//cout << "For face: " << i << " tree.do_intersect() fails" << endl;
			}
			if (result)
			{
				// unhandled exceptional situation
				//cout << "found intersection out of distance" << endl;
				//find solution
				face->r = 200;
				face->g = 200;
				face->b = 200;	
				special_case_faces++;
				// lassen wir erstmal ganz weg
			}
			//detected non overlapping local face
			else
			{
				/*face->r = 0;
				face->g = 0;
				face->b = 200;*/
				
				remote_faces.push_back(face);	
			}
		}
		else if(dist_a <= threshold && dist_b <= threshold && dist_c <= threshold)
		{	
			// Delete Case: redundant faces
			face->r = 200;
			face->g = 200;
			face->b = 0;
			redundant_faces++;
			// lassen wir ganz weg
		}
		else
		{	
			  //  face->r = 100;
			  //	face->g = 100;
			  //	face->b = 0;
			
			bool result = false;
			try {
				result = tree.do_intersect(temp);
			} catch (...)
			{
				close_tree_intersect_fails++;
				//cout << "For face: " << i << " three.do_intersect() fails" << endl;
			}
			if(result) {
				//cout << "found intersection within distance" << endl;
				//find solution
				face->r = 0;
				face->g = 0;
				face->b = 200;
				
				intersection_faces.push_back(face);
			}
			else {
				//partial overlaping, gaps etc. case
				//cout << "found within distance" << endl;
				//ggf. hier intersection erzwingen ?! (wall method s. paper)
				closeby_faces.push_back(face);
			}
		}
	}
	
	if(verbose)
	{
		printFaceSortingStatus();

		cout << "For " << far_tree_intersect_fails << " of Special Case Faces call to tree.intersect() failed" << endl;
		cout << "For " << close_tree_intersect_fails << " of Closeby Faces call to tree.intersect() failed" << endl << endl; 
		cout << timestamp << "Finished Sorting Faces..." << endl;
	}
	
	/*face->m_index[0] = face->m_index[0] + increment;
	face->m_index[1] += increment;
	face->m_index[2] += increment;
	
	m_global_faces.push_back(face);*/
	// cout << "Adding Tree Face - " << face->m_index[0] << " " << face->m_index[1] << " " << face->m_index[2] << " " << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::printFaceSortingStatus()
{
	size_t num_current_local_faces  = m_local_faces.size();
	
		double remote_ratio = ((double) remote_faces.size() / (double) num_current_local_faces) * 100;
		double intersection_ratio = ((double) intersection_faces.size() / (double) num_current_local_faces) * 100;
		double closeby_ratio = ((double) closeby_faces.size() / (double) num_current_local_faces) * 100;
		double redundant_ratio = ((double) redundant_faces / (double) num_current_local_faces) * 100;
		double special_case_ratio = ((double) special_case_faces  / (double) num_current_local_faces) * 100;
		cout << endl;
		cout << "Found # " <<  remote_faces.size() << " Remote Faces... " << remote_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  intersection_faces.size()  << " Intersection Faces... " << intersection_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  closeby_faces.size() << " Closeby but not intersecting Faces... " << closeby_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  redundant_faces  << " Redundant Faces... " << redundant_ratio << "% of all incoming" << endl;
		cout << "Found # " <<  special_case_faces  << " Special Case Faces... " <<  special_case_ratio << "% of all incoming" << endl;		
		cout << endl;
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::integrate()
{
	if(verbose)
	{
		cout <<endl << timestamp << "Start Integrating... " << endl;
	
		printLocalBufferStatus();
		printGlobalBufferStatus(); 
	}

	if (m_global_vertices.size() == 0)
	{
		lazyIntegrate(); //shorten
	}
	else
	{
		buildTree();
		buildVertexMap();
		// for all faces in local buffer
		// check whether to add face to tree/global buffer
		
		sortFaces(remote_faces, intersection_faces, closeby_faces);
		remoteIntegrate(remote_faces);
		intersectIntegrate(intersection_faces);
	}

	// for all faces in local buffer
	// add face to globalbuffer()
	/* {
		 * int status = checkOverlappingStatus(face)
		 * 
		 * if(0 = completely redundant) break;
		 * if(1 = border overlap) --> clip(boarder_faces, face)
		 * if(2 = interior overlap) --> später 
		 * 
	   }
    */

	clearLocalBuffer();

	if(verbose)
	{
		cout << endl << "Face Errors" << endl;

		for(unsigned int i = 0; i < m_global_faces.size(); i++)
		{
			if (m_global_faces[i]->m_index[0] >= m_global_index || m_global_faces[i]->m_index[1] >= m_global_index || m_global_faces[i]->m_index[2] >= m_global_index) {
				cout << "Vertex Indices for Face[" << i << "]: " << m_global_faces[i]->m_index[0] << ", " << m_global_faces[i]->m_index[1] << ", " << m_global_faces[i]->m_index[0] << endl;
				cout << "m_global_index: " << m_global_index << endl;
			}
		}

		cout << endl << "Vertice Errors" << endl;

		for(unsigned int i = 0; i < m_global_vertices.size(); i++)
		{
			if(i != m_global_vertices[i]->m_self_index) 
			{
				cout << "Index[" <<  i << "] " << m_global_vertices[i]->m_self_index << endl; 
			}
		}
	}
	
	if(verbose)
	{
		cout << endl << timestamp << "Finished Integrating..." << endl << endl;
		
		printLocalBufferStatus();
		printGlobalBufferStatus();
	}
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMeshAndIntegrate(MeshBufferPtr mesh)
{
	addMesh(mesh);
	integrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::addMeshAndLazyIntegrate(MeshBufferPtr mesh)
{
	addMesh(mesh);
	lazyIntegrate();
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::finalize()
{
	if(verbose)
	{
		cout << endl << timestamp << "Start Finalizing mesh..." << endl;
	}
	
	//boost::unordered_map<FusionVertex<VertexT, NormalT>*, int> index_map;

    int numVertices = m_global_vertices.size();
    int numFaces 	= m_global_faces.size();
    
    // Default Color values. Used if regions should not be colored.
    float r=0, g=255, b=0;
    std::vector<uchar> faceColorBuffer;

    floatArr vertexBuffer( new float[3 * numVertices] );
    floatArr normalBuffer( new float[3 * numVertices] );
    ucharArr colorBuffer(  new uchar[3 * numVertices] );
    uintArr  indexBuffer(  new unsigned int[3 * numFaces] );

    // Set the Vertex and Normal Buffer for every Vertex.
    typename vector<FVertex*>::iterator vertices_iter = m_global_vertices.begin();
    typename vector<FVertex*>::iterator vertices_end  = m_global_vertices.end();
    
    for(size_t i = 0; vertices_iter != vertices_end; ++i, ++vertices_iter)
    {
        vertexBuffer[3 * i] =     (*vertices_iter)->m_position[0];
        vertexBuffer[3 * i + 1] = (*vertices_iter)->m_position[1];
        vertexBuffer[3 * i + 2] = (*vertices_iter)->m_position[2];

        normalBuffer [3 * i] =     -(*vertices_iter)->m_normal[0];
        normalBuffer [3 * i + 1] = -(*vertices_iter)->m_normal[1];
        normalBuffer [3 * i + 2] = -(*vertices_iter)->m_normal[2];

        // Map the vertices to a position in the buffer.
        // This is necessary since the old indices might have been compromised.
        //index_map[*vertices_iter] = i;
    }

    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_iter = m_global_faces.begin();
    typename vector<FusionFace<VertexT, NormalT>*>::iterator face_end  = m_global_faces.end();

    for(size_t i = 0; face_iter != face_end; ++i, ++face_iter)
    {
		
		r=(float) (*face_iter)->r;
		g=(float) (*face_iter)->g;
		b=(float) (*face_iter)->b;
		/*
        indexBuffer[3 * i]      = index_map[m_global_vertices[(*face_iter)->m_index[0]]];
        indexBuffer[3 * i + 1]  = index_map[m_global_vertices[(*face_iter)->m_index[1]]];
        indexBuffer[3 * i + 2]  = index_map[m_global_vertices[(*face_iter)->m_index[2]]];
		*/

		indexBuffer[3 * i]      = (*face_iter)->m_index[0];
        indexBuffer[3 * i + 1]  = (*face_iter)->m_index[1];
        indexBuffer[3 * i + 2]  = (*face_iter)->m_index[2];

	
        colorBuffer[indexBuffer[3 * i]  * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i]  * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i]  * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 2] = b;
        
        /// TODO: Implement materials
        // faceColorBuffer.push_back( r );
        // faceColorBuffer.push_back( g );
        // faceColorBuffer.push_back( b );
    
    }
    // Hand the buffers over to the Model class for IO operations.

    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }   
    
    this->m_meshBuffer->setVertexArray( vertexBuffer, numVertices );
    this->m_meshBuffer->setVertexColorArray( colorBuffer, numVertices );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer, numVertices  );
    this->m_meshBuffer->setFaceArray( indexBuffer, numFaces );
    //this->m_meshBuffer->setFaceColorArray( faceColorBuffer );
    this->m_finalized = true;
	
	if(verbose)
	{
		cout << endl << timestamp << "Ended Finalizing mesh..." << endl;
	
	}
	
}

template<typename VertexT, typename NormalT> void FusionMesh<VertexT, NormalT>::flipEdge(uint v1, uint v2)
{
	cout << "No Edge no Flip!" << endl;
	cout << "But these are two nice uints" << v1 << ", " << v2 << endl;
}

} // namespace lvr
