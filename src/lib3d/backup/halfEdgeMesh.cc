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

void HalfEdgeMesh::check_next_neighbor(int startIndex, HalfEdgeFace* face, hash_map<unsigned int, HalfEdge*>* edges){

	face->used = true;

	HalfEdge* edge = face->edge;
	HalfEdge* pair = face->edge->pair;

	HalfEdgeFace* nb = 0;

	hash_map<int, HalfEdge*>::iterator it;

	unsigned int hashValue;

	if(pair != 0) nb = pair->face;

	do{

		edge = edge->next;
		pair = edge->pair;
		if(pair != 0) nb = pair->face;

		if(nb != 0 &&
				!nb->used &&
				nb->mcIndex == startIndex) check_next_neighbor(startIndex, nb, edges);
		else {

			hashValue = edge->start->index;

			if(nb != 0){

				if( nb->mcIndex != startIndex){

					(*edges)[hashValue] = edge;

				}
			} else {

				(*edges)[hashValue] = edge;

			}

		}

	} while(edge != face->edge);


}


void HalfEdgeMesh::extract_borders(){

	hash_map<unsigned int, HalfEdge*> points;
	hash_map<unsigned int, HalfEdge*>::iterator pit;

	vector<HalfEdgeVertex*> vertices;
	vector<HalfEdgeVertex*>::iterator vit;

	vector<HalfEdgeFace*>::iterator it = he_faces.begin();

	//if(edge->used) cout << "Used edge found!" << endl;


	HalfEdgeFace* face;

	HalfEdge* startEdge;
	HalfEdge* edge;

	HalfEdgeVertex* start;
	HalfEdgeVertex* end;

	HalfEdgePolygon* polygon;

	int hashValue;

	unsigned int c;

	bool error = false;
	int mcIndexOK = false;

	int code;
	int lastCode;

	//Alles Faces durchlaufen
	while(it != he_faces.end()){
		face = *it;
		mcIndexOK = classifyFace(face);

		if(mcIndexOK >= 0)
			face->texture_index = mcIndexOK; //Klingt irgendwie doof...
		else
			face->texture_index = 0;

		if(!face->used && mcIndexOK >= 0) check_next_neighbor(face->mcIndex, face, &points);

		cout << "NVERTICES: " << points.size() << endl;

		if(mcIndexOK < 0){
			HalfEdge* e = face->edge;
			//out << "BEGIN" << endl;

			polygon = new HalfEdgePolygon;

			do{
				//polygon->setTexture(face->texture_index);
				//polygon->addVertex(e->start);
				//polygon->setRGBColor(0.0f, 1.0f, 0.0f);
				polygon->texture_index = face->texture_index;
				polygon->vertices.push_back(e->start);
				polygon->indices.push_back(e->start->index);
				e = e->next;
			} while(e != face->edge);
			//out << "END" << endl;
			hem_polygons.push_back(polygon);

		}

		it++;

		//cout << "POINTS:SIZE: " << points.size() << endl;

		HalfEdgeVertex* lastEnd;

		if(!points.empty()){
			//cout << "START NEW POLYGON" << endl;
			pit = points.begin();
			startEdge = (*pit).second;
			hashValue = startEdge->end->index;
			edge = startEdge;

			start = startEdge->start;
			end = startEdge->end;
			lastEnd = end;

			vertices.push_back(edge->start);

			code = freeman3D(edge);
			lastCode = code;
			c = 0;
			error = false;

			do{

				//cout << hashValue << endl;
				pit = points.find(hashValue);

				c++;

				if(pit != points.end() ){
					edge = (*pit).second;

					code = freeman3D(edge);

					if(code == lastCode && lastEnd == edge->start)
						end = edge->end;
					else{
						vertices.push_back(end);
						start = edge->start;
						end = edge->end;
					}
					lastEnd = edge->end;
					lastCode = code;

					hashValue = edge->end->index;

				}

				if(pit == points.end()){
					cout << "WARNING: No matching edge found..." << endl;
					error = true;
					break;
				}

				if(c > 2 * points.size() + 2){
					cout << "WARNING: Loop detected..." << endl;
					error = true;
					break;
				}

				if(edge == startEdge){
					//Polygon schliessen...
					//vertices.push_back(*edge->start);
					break;
				}

				//cout << c << endl;

			} while(true);



			if(!error){
				polygon = new HalfEdgePolygon;

				HalfEdgeVertex* ver;

				for(vit = vertices.begin(); vit != vertices.end(); vit++){

					ver = *vit;
					polygon->vertices.push_back(ver);
					polygon->texture_index = face->texture_index;
					polygon->indices.push_back(ver->index);

				}
				hem_polygons.push_back(polygon);
				cout << "PVERTICES: " << polygon->vertices.size() << endl;

			} else {

				//hem_polygons.push_back(polygon);

			}

			vertices.clear();
			points.clear();
			//cout << "POLYGON DONE!" <<endl;
		}
	}


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
