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

void HalfEdgeMesh::finalize(){

	number_of_vertices = (int)he_vertices.size();
	number_of_faces = (int)he_faces.size();

	vertices = new float[3 * number_of_vertices];
	normals = new float[3 * number_of_vertices];
	colors = new float[3 * number_of_vertices];

	indices = new unsigned int[3 * number_of_faces];

	for(size_t i = 0; i < he_vertices.size(); i++){
		vertices[3 * i] =     he_vertices[i]->position.x;
		vertices[3 * i + 1] = he_vertices[i]->position.y;
		vertices[3 * i + 2] = he_vertices[i]->position.z;

		normals [3 * i] =     -he_vertices[i]->normal.x;
		normals [3 * i + 1] = -he_vertices[i]->normal.y;
		normals [3 * i + 2] = -he_vertices[i]->normal.z;

		colors  [3 * i] = 0.0;
		colors  [3 * i + 1] = 1.0;
		colors  [3 * i + 2] = 0.0;
	}

	for(size_t i = 0; i < he_faces.size(); i++){
		indices[3 * i]      = he_faces[i]->index[0];
		indices[3 * i + 1]  = he_faces[i]->index[1];
		indices[3 * i + 2]  = he_faces[i]->index[2];
	}

	finalized = true;
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
	if(distance < 5.0 && cos_angle > 0.99) return true;

	//Return false if face is not in plane
	return false;
}

void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0, HalfEdgeFace* face, HalfEdgePolygon* polygon){

	face->used = true;
	polygon->add_face(face);

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
					check_next_neighbor(f0, current_neighbor, polygon);
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

	int c = 0;
	for(face_iterator = he_faces.begin(); face_iterator != he_faces.end(); face_iterator++){
		if(c % 10000 == 0) cout << "Extracting Borders: " << c << " / " << he_faces.size() << endl;
		current_face = *face_iterator;
		if(!current_face->used){

			current_n = current_face->normal;
			current_d = current_face->edge->start->position * current_n;
			current_v = current_face->edge->start->position;

			current_polygon = new HalfEdgePolygon();
			check_next_neighbor(current_face, current_face, current_polygon);
			current_polygon->generate_list();
			//current_polygon->fuse_edges();
			if((int)current_polygon->edge_list.size() > biggest_size){
				biggest_size = (int)current_polygon->edge_list.size();
				biggest_polygon = current_polygon;
			}
			hem_polygons.push_back(current_polygon);

		}
		c++;
	}

	cout << "BIGGEST POLYGON: " << biggest_polygon << endl;
}

void HalfEdgeMesh::create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges){


}

void HalfEdgeMesh::write_polygons(string filename){

	ofstream out(filename.c_str());

	HalfEdgePolygon* polygon;
	HalfEdge*        edge;

	vector<HalfEdgePolygon*>::iterator polygon_it;

//	int c = 0;
//	for(polygon_it  = hem_polygons.begin();
//	    polygon_it != hem_polygons.end();
//	    polygon_it++)
//	{
//		if(c % 10000 == 0) cout << "Writing Polygons: " << c << " / " << hem_polygons.size() << endl;
//		polygon = *polygon_it;
//		map<HalfEdgeVertex* , HalfEdge*>::iterator edge_it;
//		for(edge_it  = polygon->edge_list.begin();
//		    edge_it != polygon->edge_list.end();
//		    edge_it++)
//		{
//			edge = edge_it->second;
//			out << "BEGIN" << endl;
//
//			out << edge->start->position.x << " ";
//			out << edge->start->position.y << " ";
//			out << edge->start->position.z << endl;
//
//			out << edge->end->position.x << " ";
//			out << edge->end->position.y << " ";
//			out << edge->end->position.z << endl;
//
//			out << "END" << endl;
//		}
//		c++;
//	}

	int c = 0;

	polygon = biggest_polygon;
	map<HalfEdgeVertex* , HalfEdge*>::iterator edge_it;
	for(edge_it  = polygon->edge_list.begin();
	edge_it != polygon->edge_list.end();
	edge_it++)
	{
		edge = edge_it->second;
		out << "BEGIN" << endl;

		out << edge->start->position.x << " ";
		out << edge->start->position.y << " ";
		out << edge->start->position.z << endl;

		out << edge->end->position.x << " ";
		out << edge->end->position.y << " ";
		out << edge->end->position.z << endl;

		out << "END" << endl;
	}
	c++;


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
