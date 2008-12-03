/*
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#include "HalfEdgeMesh.h"


HePlane::HePlane(Normal n0, Vertex v0){
	n = n0;
	d = n * v0;
}

HePlane::HePlane(const HePlane &o){
	n = o.n;
	d = o.d;
}

void HePlane::interpolate(HePlane p){
	n += p.n;

	n.x = n.x / 2.0;
	n.y = n.y / 2.0;
	n.z = n.z / 2.0;

	n.normalize();

	d = 0.5 * (d + p.d);
}

float HePlane::distance(Vertex v){
	return n * v - d;
}





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

void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0, HePlane &p, HalfEdgeFace* face, hash_map<unsigned int, HalfEdge*>* edges){

	face->used = true;

	HalfEdge* edge = face->edge;
	HalfEdge* pair = face->edge->pair;

	HalfEdgeFace* nb = 0;

	hash_map<int, HalfEdge*>::iterator it;

	if(pair != 0) nb = pair->face;

	do{

		edge = edge->next;
		pair = edge->pair;
		if(pair != 0) nb = pair->face;

		if(nb != 0){

			if(!nb->used){

				if(classifyFace(f0, nb, p) >= 0)
					check_next_neighbor(f0, p, nb, edges);
				else
					(*edges)[edge->start->index] = edge;
			}

		} else {
			//If no neighbor exists, current edge is a border
			(*edges)[edge->start->index] = edge;
		}


//		if(nb != 0 &&
//				!nb->used &&
//				classifyFace(nb, n_0) >= 0) check_next_neighbor(n_0, nb, edges);
//		else {
//
//			//If neighbor doesn't exist, current edges is a border
//			if(nb == 0){
//				(*edges)[edge->start->index] = edge;
//			//Current edge is also a border if normal criterium isn't fullfilled
//			} else if(classifyFace(nb, n_0) < 0){
//				(*edges)[edge->start->index] = edge;
//			}
//
//		}

	} while(edge != face->edge);


}


void HalfEdgeMesh::extract_borders(){


	hash_map<unsigned int, HalfEdge*> border_edges;
	hash_map<unsigned int, HalfEdge*>::iterator pit;

	vector<HalfEdgeVertex*> vertices;
	vector<HalfEdgeVertex*>::iterator vit;

	vector<HalfEdgeFace*>::iterator it;

	for(it = he_faces.begin(); it !=  he_faces.end(); it++){
		HalfEdgeFace* f = *it;
		f->interpolate_normal();
	}


	it = he_faces.begin();

	//if(edge->used) cout << "Used edge found!" << endl;

	HalfEdgeFace* face;

	vector<int> polygon;

	//Alles Faces durchlaufen
	while(it != he_faces.end()){
		face = *it;

		HalfEdgeVertex* v1 = he_vertices[face->index[0]];
		HalfEdgeVertex* v2 = he_vertices[face->index[1]];
		HalfEdgeVertex* v3 = he_vertices[face->index[2]];

		Vertex center;
		center += v1->position;
		center += v2->position;
		center += v3->position;

		center.x /= 3.0;
		center.y /= 3.0;
		center.z /= 3.0;

		HePlane plane(face->normal, center);

		if(!face->used) check_next_neighbor(face, plane, face, &border_edges);

		it++;
		if(!border_edges.empty()){
			//create_polygon(polygon, &border_edges);
			for(pit = border_edges.begin(); pit != border_edges.end(); pit++){
				HalfEdgePolygon* poly = new HalfEdgePolygon;
				HalfEdge* e = (*pit).second;
				poly->indices.push_back(e->start->index);
				poly->indices.push_back(e->end->index);
				hem_polygons.push_back(poly);
			}
		}

		border_edges.clear();
	}


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
