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

void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0, HalfEdgeFace* face, HalfEdgePolygon* polygon){


}


void HalfEdgeMesh::extract_borders(){


}

void HalfEdgeMesh::create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges){

	stack<int> polygon_vertex_indices;

	hash_map<unsigned int, HalfEdge*>::iterator it;

	HalfEdge* start_edge = (*edges->begin()).second;
	HalfEdge* current_edge = start_edge;


	cout << "BEGIN LOOP: " << edges->size() << endl;
	do{
		it = edges->find(current_edge->end->index);
		if(it == edges->end()){
			cout << "Edge not found" << endl;
		} else {
			current_edge = (*it).second;
			cout << current_edge->start->index << " "
			     << current_edge->end->index   << " "
			     << start_edge->start->index   << endl;
		}
	} while(current_edge != start_edge);
	cout <<  "END LOOP" << endl;
}

void HalfEdgeMesh::write_polygons(string filename){

//	ofstream out;
//
//	out.open(filename.c_str(), ios::out);
//	if (!out) {
//		cerr << "*** error: can't create file "  << " ***\n";
//		exit(2);
//	}
//
//	vector<HalfEdgePolygon*>::iterator it = hem_polygons.begin();
//	vector<int> vertex_list;
//	while(it != hem_polygons.end()){
//		vertex_list.clear();
//		HalfEdgePolygon* p = *it;
//		p->generate_list(vertex_list);
//		if(vertex_list.size() > 0){
//			out << "BEGIN" << endl;
//			for(size_t i = 0; i < vertex_list.size(); i++){
//				HalfEdgeVertex* v = he_vertices[vertex_list[i]];
//				Vertex pos = v->position;
//				out << pos.x << " " << pos.y << " " << pos.z << "0.0 1.0 0.0" << endl;
//			}
//			out << "END" << endl;
//		}
//		it++;
//	}

	ofstream out;

	out.open(filename.c_str());

	if(!out.good()){

		cerr << "ERROR: Could not open file '" << filename << "'." << endl;
		exit(-1);

	}

	vector<HalfEdgePolygon*>::iterator it;
	vector<int>::iterator vit;

	HalfEdgePolygon* p;
	HalfEdgeVertex* v;

	int index;

	for(it = hem_polygons.begin(); it != hem_polygons.end(); it++){

		p = (*it);

		out << "BEGIN" << endl;

		for(vit = p->indices.begin(); vit != p->indices.end(); vit++){
			index = (*vit);
			v = he_vertices[index];
			out << v->position.x << " " << v->position.y << " " << v->position.z << " ";
			out << 0.0 << " " << 1.0 << " " << 0.0 << endl;
		}

		out << "END" << endl;

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
