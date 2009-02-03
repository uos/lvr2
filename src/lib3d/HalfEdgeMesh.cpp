/*
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#include "HalfEdgeMesh.h"



HalfEdgeMesh::HalfEdgeMesh() {
	global_index = 0;
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


bool HalfEdgeMesh::check_face(HalfEdgeFace* f0, HalfEdgeFace* current){

	float distance_to_plane = fabs(current->edge->start->position * current_n - current_d);

	if(distance_to_plane < 0.1) return true;


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


void HalfEdgeMesh::extract_borders(){

	HalfEdgeFace*       current_face;
	HalfEdgePolygon*    current_polygon;
	vector<HalfEdgeFace*>::iterator face_iterator;

	for(face_iterator = he_faces.begin(); face_iterator != he_faces.end(); face_iterator++){
		current_face = *face_iterator;
		if(!current_face->used){

			current_n = current_face->getFaceNormal();
			current_d = current_face->edge->start->position * current_n;

			current_polygon = new HalfEdgePolygon();
			check_next_neighbor(current_face, current_face, current_polygon);
			current_polygon->generate_list();
			hem_polygons.push_back(current_polygon);

		}
	}
}

void HalfEdgeMesh::create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges){


}

void HalfEdgeMesh::write_polygons(string filename){

	ofstream out(filename.c_str());

	HalfEdgePolygon* polygon;
	HalfEdge*        edge;

	vector<HalfEdgePolygon*>::iterator polygon_it;

	for(polygon_it  = hem_polygons.begin();
	    polygon_it != hem_polygons.end();
	    polygon_it++)
	{
		polygon = *polygon_it;
		vector<HalfEdge*>::iterator edge_it;
		for(edge_it  = polygon->edge_list.begin();
		    edge_it != polygon->edge_list.end();
		    edge_it++)
		{
			edge = *edge_it;
			out << "BEGIN" << endl;

			out << edge->start->position.x << " ";
			out << edge->start->position.y << " ";
			out << edge->start->position.z << endl;

			out << edge->end->position.x << " ";
			out << edge->end->position.y << " ";
			out << edge->end->position.z << endl;

			out << "END" << endl;
		}
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
