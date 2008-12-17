/*
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: twiemann
 */

#include "HalfEdgeMesh.h"

HalfEdgeVertex::HalfEdgeVertex(){
	index = -1;
}

HalfEdgeVertex::HalfEdgeVertex(const HalfEdgeVertex &o){

	index = o.index;
	position = o.position;
	normal = o.normal;

	vector<HalfEdge*>::iterator it;

	in.clear();
	out.clear();

	for(size_t i = 0; i < o.in.size(); i++) in.push_back(o.in[i]);
	for(size_t i = 0; i < o.out.size(); i++) out.push_back(o.out[i]);
}

HalfEdgeFace::HalfEdgeFace(const HalfEdgeFace &o){
	edge = o.edge;
	used = o.used;

	for(size_t i = 0; i < o.indices.size(); i++) indices.push_back(o.indices[i]);
	for(int i = 0; i < 3; i++) index[i] = o.index[i];
}

HalfEdge::HalfEdge(){
	start = end = 0;
	next = pair = 0;
	face = 0;
	used = false;
}

HalfEdge::~HalfEdge(){
	delete next;
	delete pair;
}

HalfEdgeFace::HalfEdgeFace(){
	edge = 0;
	index[0] = index[1] = index[2] = 0;
}

void HalfEdgeFace::calc_normal(){

	Vertex vertices[3];
	HalfEdgeVertex* start = edge->start;
	HalfEdge* current_edge = edge;

	int c = 0;
	while(current_edge->end != start){
		vertices[c] = current_edge->start->position;
		current_edge = current_edge->next;
		c++;
	}
	Vertex diff1 = vertices[0] - vertices[1];
	Vertex diff2 = vertices[0] - vertices[2];
	normal = Normal(diff1.cross(diff2));
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

void HalfEdgeMesh::check_next_neighbor(float d_0, HalfEdgeFace* face, HalfEdgePolygon* polygon){

	face->used = true;

	HalfEdge* edge = face->edge;
	HalfEdge* pair = face->edge->pair;

	HalfEdgeFace* neighbor = 0;

	//Go to first neighbor face
	if(pair != 0) neighbor = pair->face;



	//Iterate through neighbor faces
	do{
		//Go to next neighbor
		if(pair != 0) neighbor = pair->face;

		//If neighbor exists, check is nb-face is
		//in current plane
		if(neighbor != 0 && !neighbor->used && classifyFace(neighbor, d_0) > 0){
			polygon->add_face(face, pair);
			check_next_neighbor(d_0, neighbor, polygon);
		}

		edge = edge->next;
		pair = edge->pair;

	} while(edge != face->edge);

}


void HalfEdgeMesh::extract_borders(){

	HalfEdgeFace* face;
	vector<HalfEdgeFace*>::iterator face_iterator;
	vector<int> vertex_list;

	for(face_iterator = he_faces.begin(); face_iterator != he_faces.end(); face_iterator++){

		//Get current face
		face = *face_iterator;

		//Calculate d_0

		//Do not double check!
		if(!face->used){

			Vertex cog;
			for(int i = 0; i < 3; i++) cog += he_vertices[face->index[i]]->position;

			cog.x = cog.x / 3.0;
			cog.y = cog.y / 3.0;
			cog.z = cog.z / 3.0;

			float d_0 = face->normal * cog;

			HalfEdgePolygon* polygon = new HalfEdgePolygon(face);
			check_next_neighbor(d_0, face, polygon);
			polygon->generate_list(vertex_list);
			hem_polygons.push_back(polygon);
		}
	}


}

void HalfEdgeMesh::write_polygons(string filename){

	ofstream out;

	out.open(filename.c_str(), ios::out);
	if (!out) {
		cerr << "*** error: can't create file "  << " ***\n";
		exit(2);
	}

	vector<HalfEdgePolygon*>::iterator it = hem_polygons.begin();
	vector<int> vertex_list;
	while(it != hem_polygons.end()){
		vertex_list.clear();
		HalfEdgePolygon* p = *it;
		p->generate_list(vertex_list);
		if(vertex_list.size() > 0){
			out << "BEGIN" << endl;
			for(size_t i = 0; i < vertex_list.size(); i++){
				HalfEdgeVertex* v = he_vertices[vertex_list[i]];
				Vertex pos = v->position;
				out << pos.x << " " << pos.y << " " << pos.z << "0.0 1.0 0.0" << endl;
			}
			out << "END" << endl;
		}
		it++;
	}

//	ofstream out;
//
//	out.open(filename.c_str());
//
//	if(!out.good()){
//
//		cerr << "ERROR: Could not open file '" << filename << "'." << endl;
//		exit(-1);
//
//	}
//
//	vector<HalfEdgePolygon*>::iterator it;
//	vector<int>::iterator vit;
//
//	HalfEdgePolygon* p;
//	HalfEdgeVertex* v;
//
//	int index;
//
//	for(it = hem_polygons.begin(); it != hem_polygons.end(); it++){
//
//		p = (*it);
//
//		out << "BEGIN" << endl;
//
//		for(vit = p->indices.begin(); vit != p->indices.end(); vit++){
//			index = (*vit);
//			v = he_vertices[index];
//			out << v->position.x << " " << v->position.y << " " << v->position.z << " ";
//			out << 0.0 << " " << 1.0 << " " << 0.0 << endl;
//		}
//
//		out << "END" << endl;
//
//	}

}

void HalfEdgeMesh::analize(){

	vector<HalfEdgeFace*>::iterator it;

	int n_index_miss = 0;
	int n_null_pairs = 0;

	for(it = he_faces.begin(); it != he_faces.end(); it++){

		HalfEdge* edge = (*it)->edge;
		HalfEdge* pair = edge->pair;

		do{
			if(pair != 0){
				if(edge->start->index != pair->end->index) n_index_miss++;
				if(edge->end->index != pair->start->index) n_index_miss++;
				if(edge->next->start->index != edge->end->index) n_index_miss++;
				if(pair->face != 0){
					if(edge->pair->next->start->index != edge->pair->end->index) n_index_miss++;
					if(edge->pair->next->start->index != edge->start->index) n_index_miss++;
				}
			} else {
				n_null_pairs++;
			}
			edge = edge->next;
			pair = edge->pair;
		} while(edge != (*it)->edge);

	}

	cout << "Number of index misses : " << n_index_miss << endl;
	cout << "NUmber of NULL pointers: " << n_null_pairs << endl;

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
