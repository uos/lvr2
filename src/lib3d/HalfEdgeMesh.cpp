/*
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#include "HalfEdgeMesh.h"



HalfEdgeMesh::HalfEdgeMesh() {
	global_index = 0;
	biggest_polygon = 0;
	biggest_size = -1;
}

HalfEdgeMesh::~HalfEdgeMesh() {

}

int HalfEdgeMesh::classifyFace(HalfEdgeFace* f)
{
	Normal n = f->getInterpolatedNormal();
	Normal n_ceil(0.0, 1.0, 0.0);
	Normal n_floor(0.0, -1.0, 0.0);

	if(n_ceil * n > 0.98) return 1;
	if(n_floor * n > 0.98) return 2;

	float radius = sqrt(n.x * n.x + n.z * n.z);

	if(radius > 0.95) return 3;

	return 0;
}

void HalfEdgeMesh::finalize(){

	cout << "HEM::finalize()" << endl;

	number_of_vertices = (int)he_vertices.size();
	number_of_faces = (int)he_faces.size();

	vertices = new float[3 * number_of_vertices];
	normals = new float[3 * number_of_vertices];
	colors = new float[3 * number_of_vertices];

	m_indices = new unsigned int[3 * number_of_faces];

	for(size_t i = 0; i < he_vertices.size(); i++){
		vertices[3 * i] =     he_vertices[i]->position.x;
		vertices[3 * i + 1] = he_vertices[i]->position.y;
		vertices[3 * i + 2] = he_vertices[i]->position.z;

		normals [3 * i] =     -he_vertices[i]->normal.x;
		normals [3 * i + 1] = -he_vertices[i]->normal.y;
		normals [3 * i + 2] = -he_vertices[i]->normal.z;

		colors  [3 * i] = 0.8;
		colors  [3 * i + 1] = 0.8;
		colors  [3 * i + 2] = 0.8;
	}

	for(size_t i = 0; i < he_faces.size(); i++){
		m_indices[3 * i]      = he_faces[i]->index[0];
		m_indices[3 * i + 1]  = he_faces[i]->index[1];
		m_indices[3 * i + 2]  = he_faces[i]->index[2];

		int surface_class = classifyFace(he_faces[i]);

		switch(surface_class)
		{
		case 1:
			colors[m_indices[3 * i]  * 3 + 0] = 0.0;
			colors[m_indices[3 * i]  * 3 + 1] = 0.0;
			colors[m_indices[3 * i]  * 3 + 2] = 1.0;

			colors[m_indices[3 * i + 1] * 3 + 0] = 0.0;
			colors[m_indices[3 * i + 1] * 3 + 1] = 0.0;
			colors[m_indices[3 * i + 1] * 3 + 2] = 1.0;

			colors[m_indices[3 * i + 2] * 3 + 0] = 0.0;
			colors[m_indices[3 * i + 2] * 3 + 1] = 0.0;
			colors[m_indices[3 * i + 2] * 3 + 2] = 1.0;

			break;
		case 2:
			colors[m_indices[3 * i] * 3 + 0] = 1.0;
			colors[m_indices[3 * i] * 3 + 1] = 0.0;
			colors[m_indices[3 * i] * 3 + 2] = 0.0;

			colors[m_indices[3 * i + 1] * 3 + 0] = 1.0;
			colors[m_indices[3 * i + 1] * 3 + 1] = 0.0;
			colors[m_indices[3 * i + 1] * 3 + 2] = 0.0;

			colors[m_indices[3 * i + 2] * 3 + 0] = 1.0;
			colors[m_indices[3 * i + 2] * 3 + 1] = 0.0;
			colors[m_indices[3 * i + 2] * 3 + 2] = 0.0;

			break;
		case 3:
			colors[m_indices[3 * i] * 3 + 0] = 0.0;
			colors[m_indices[3 * i] * 3 + 1] = 1.0;
			colors[m_indices[3 * i] * 3 + 2] = 0.0;

			colors[m_indices[3 * i + 1] * 3 + 0] = 0.0;
			colors[m_indices[3 * i + 1] * 3 + 1] = 1.0;
			colors[m_indices[3 * i + 1] * 3 + 2] = 0.0;

			colors[m_indices[3 * i + 2] * 3 + 0] = 0.0;
			colors[m_indices[3 * i + 2] * 3 + 1] = 1.0;
			colors[m_indices[3 * i + 2] * 3 + 2] = 0.0;

			break;
		}

	}

	finalized = true;
}

void HalfEdgeMesh::finalize(vector<planarCluster> &planes)
{
	if(!finalized) finalize();

	// Create a color gradient
	float r[255];
	float g[255];
	float b[255];

	for(int i = 0; i < 255; i++)
	{
		 r[i] = (252 - i % 64 * 4) / 255.0;
		 g[i] =  (32 + i % 32 * 6) / 255.0;
		 b[i] =  (64 + i % 64 * 3) / 255.0;
	}

	// Change colors according to clustering
	int count = 0;
	for(size_t i = 0; i < planes.size(); i++)
	{
		planarCluster cluster = planes[i];
		for(size_t j = 0; j < cluster.face_count; j++)
		{
			HalfEdgeFace* f = cluster.faces[j];

			// Get vertex indices
			int _a = f->index[0];
			int _b = f->index[1];
			int _c = f->index[2];

			cout << r[count % 255] << " "
			     << g[count % 255] << " "
			     << b[count % 255] << endl;

			colors[3 * _a    ] = r[count % 255];
			colors[3 * _a + 1] = g[count % 255];
			colors[3 * _a + 2] = b[count % 255];

			colors[3 * _b    ] = r[count % 255];
			colors[3 * _b + 1] = g[count % 255];
			colors[3 * _b + 2] = b[count % 255];

			colors[3 * _c    ] = r[count % 255];
			colors[3 * _c + 1] = g[count % 255];
			colors[3 * _c + 2] = b[count % 255];

		}
		count++;
	}
}

bool HalfEdgeMesh::isFlatFace(HalfEdgeFace* face){

	int index = face->mcIndex;

	//WALL
	if(index == 240 || index == 15 || index == 153 || index == 102){

		return true;

	}
	//FLOOR
	else if(index == 204){

		return true;

	}
	//CEIL
	else if (index == 51){

		return true;

	}
	//DOORS
	else if (index == 9 || index == 144 || index == 96 || index == 6){

		return true;

	}
	//OTHER FLAT POLYGONS
	else if(index ==  68 || index == 136 || index ==  17 || index ==  34 || //Variants of MC-Case 2
			index == 192 || index ==  48 || index ==  12 || index ==   3 ){

		return true;

	} else if (index ==  63 || index == 159 || index == 207 || index == 111 || //Variants of MC-Case 2 (compl)
			index == 243 || index == 249 || index == 252 || index == 246 ||
			index == 119 || index == 187 || index == 221 || index == 238){
		return true;

	}

	return false;
}


void HalfEdgeMesh::getArea(set<HalfEdgeFace*> &faces, HalfEdgeFace* face, int depth, int max){

	vector<HalfEdgeFace*> adj;
	face->getAdjacentFaces(adj);

	vector<HalfEdgeFace*>::iterator it;
	for(it = adj.begin(); it != adj.end(); it++){
		faces.insert(*it);
		if(depth < max){
			getArea(faces, *it, depth + 1, max);
		}
	}

}

void HalfEdgeMesh::shiftIntoPlane(HalfEdgeFace* f){

	HalfEdge* edge  = f->edge;
	HalfEdge* start = edge;

	do{
		float d = (current_v - edge->end->position) * current_n;
		edge->end->position = edge->end->position + (current_n * d);
		edge = edge -> next;
	} while(edge != start);

}

bool HalfEdgeMesh::check_face(HalfEdgeFace* f0, HalfEdgeFace* current){

	//Calculate Plane representation
	Normal n_0 = f0->getInterpolatedNormal();
	Vertex p_0 = f0->getCentroid();

	//Calculate needed parameters
	float  d = p_0 * n_0;
	float  distance = fabs(current->getCentroid() * n_0 - d);
	float  cos_angle = n_0 * current->getInterpolatedNormal();

	//Decide using given thresholds
	//if(distance < 8.0 && cos_angle > 0.98) return true;
	if(cos_angle > 0.98) return true;

	//Return false if face is not in plane
	return false;
}

void HalfEdgeMesh::cluster(vector<planarCluster> &planes)
{
	for(size_t i = 0; i < he_faces.size(); i++)
	{
		HalfEdgeFace* current_face = he_faces[i];

		if(!current_face->used)
		{

			planarCluster cluster;
			cluster.face_count = 0;
			cluster.faces = 0;

			vector<HalfEdgeFace*> faces;

			check_next_neighbor(current_face, current_face, 0, faces);

			// Copy faces into cluster struct
			cluster.face_count = faces.size();
			cluster.faces = new HalfEdgeFace*[faces.size()];

			for(size_t i = 0; i < faces.size(); i++)
			{
				cluster.faces[i] = faces[i];
			}

			planes.push_back(cluster);
		}

	}
}

void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0,
		                               HalfEdgeFace* face,
		                               HalfEdge* edge,
		                               HalfEdgePolygon* polygon){

	face->used = true;
	polygon->add_face(face, edge);

    //Iterate through all surrounding faces
	HalfEdge* start_edge   = face->edge;
	HalfEdge* current_edge = face->edge;
	HalfEdge* pair         = current_edge->pair;
	HalfEdgeFace* current_neighbor;

	do{
		pair = current_edge->pair;
		if(pair != 0){
			current_neighbor = pair->face;
			if(current_neighbor != 0){
				if(check_face(f0, current_neighbor) && !current_neighbor->used){
					check_next_neighbor(f0, current_neighbor, current_edge, polygon);
				}
			}
		}
		current_edge = current_edge->next;
	} while(start_edge != current_edge);


}

void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0,
		                               HalfEdgeFace* face,
		                               HalfEdge* edge,
		                               vector<HalfEdgeFace*> &faces){

	face->used = true;
	faces.push_back(face);

    //Iterate through all surrounding faces
	HalfEdge* start_edge   = face->edge;
	HalfEdge* current_edge = face->edge;
	HalfEdge* pair         = current_edge->pair;
	HalfEdgeFace* current_neighbor;

	do{
		pair = current_edge->pair;
		if(pair != 0){
			current_neighbor = pair->face;
			if(current_neighbor != 0){
				if(check_face(f0, current_neighbor) && !current_neighbor->used){
					check_next_neighbor(f0, current_neighbor, current_edge, faces);
				}
			}
		}
		current_edge = current_edge->next;
	} while(start_edge != current_edge);


}


void HalfEdgeMesh::generate_polygons(){
//
//	vector<HalfEdgePolygon*>::iterator it;
//	HalfEdgePolygon* polygon;
//
//	for(it =  hem_polygons.begin();
//		it != hem_polygons.end();
//		it++)
//	{
//		polygon = *it;
//		polygon->fuse_edges();
//	}

}

void HalfEdgeMesh::extract_borders(){

	HalfEdgeFace*       current_face;
	HalfEdgePolygon*    current_polygon;
	vector<HalfEdgeFace*>::iterator face_iterator;

	unsigned int biggest_size = 0;

	int c = 0;
	for(face_iterator = he_faces.begin(); face_iterator != he_faces.end(); face_iterator++){
		if(c % 10000 == 0) cout << "Extracting Borders: " << c << " / " << he_faces.size() << endl;
		current_face = *face_iterator;
		if(!current_face->used){

			current_n = current_face->normal;
			current_d = current_face->edge->start->position * current_n;
			current_v = current_face->edge->start->position;

			current_polygon = new HalfEdgePolygon();
			check_next_neighbor(current_face, current_face, 0, current_polygon);
			current_polygon->generate_list();
			//current_polygon->fuse_edges();
			//current_polygon->test();

			hem_polygons.push_back(current_polygon);
			if(current_polygon->faces.size() > biggest_size){
				biggest_size = current_polygon->faces.size();
				biggest_polygon = current_polygon;
			}

		}
		c++;
	}

	cout << "BIGGEST POLYGON: " << biggest_polygon << endl;

}

void HalfEdgeMesh::create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges){


}

void HalfEdgeMesh::write_polygons(string filename){

	cout << "WRITE" << endl;

	ofstream out(filename.c_str());

	vector<HalfEdgePolygon*>::iterator p_it;
	//multiset<HalfEdge*>::iterator it;
	EdgeMapIterator it;

	for(it  = biggest_polygon->edges.begin();
		it != biggest_polygon->edges.end();
		it++)
	{
		HalfEdge* e = it->second;
		out << "BEGIN" << endl;
		out << e->start->position.x << " " << e->start->position.y << " " << e->start->position.z << endl;
		out << e->end->position.x   << " " << e->end->position.y   << " " << e->end->position.z   << endl;
		out << "END" << endl;
	}

	biggest_polygon->fuse_edges();

//	for(p_it =  hem_polygons.begin();
//		p_it != hem_polygons.end();
//		p_it++)
//	{
//		HalfEdgePolygon* polygon = *p_it;
//		for(it  = polygon->edges.begin();
//			it != polygon->edges.end();
//			it++)
//		{
//			HalfEdge* e = *it;
//			out << "BEGIN" << endl;
//			out << e->start->position.x << " " << e->start->position.y << " " << e->start->position.z << endl;
//			out << e->end->position.x   << " " << e->end->position.y   << " " << e->end->position.z   << endl;
//			out << "END" << endl;
//		}
//	}



}

void HalfEdgeMesh::write_face_normals(string filename){

	ofstream out(filename.c_str());

	HalfEdgeFace* face;

	Normal n;
	Vertex v;

	int c = 0;

	vector<HalfEdgeFace*>::iterator face_iterator;
	for(face_iterator = he_faces.begin();
		face_iterator != he_faces.end();
		face_iterator++)
	{
		if(c % 10000 == 0){
			cout << "Write Face Normals: " << c << " / " << he_faces.size() << endl;
		}
		face = *face_iterator;
		//n = face->getFaceNormal();
		n = face->getInterpolatedNormal();
		v = face->getCentroid();

		out << v.x << " " << v.y << " " << v.z << " "
		    << n.x << " " << n.y << " " << n.z << endl;

		c++;
	}

}

void HalfEdgeMesh::printStats(){
	if(finalized){
		cout << "##### HalfEdge Mesh (S): " << number_of_vertices << " Vertices / "
		                                    << number_of_faces    << " Faces.   " << endl;
	} else {
		cout << "##### HalfEdge Mesh (D): " << he_vertices.size() << " Vertices / "
		                                    << he_faces.size() / 3 << " Faces." << endl;
	}
}
