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

	float c_r, c_g, c_b;

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
			if(cluster.face_count > 50)
			{
				c_r = 0.6;
				c_g = 0.0;
				c_b = 0.0;
			}
			else
			{
				c_r = 0.0;
				c_g = 0.6;
				c_b = 0.0;

			}
			HalfEdgeFace* f = cluster.faces[j];

			// Get vertex indices
			int _a = f->index[0];
			int _b = f->index[1];
			int _c = f->index[2];

			colors[3 * _a    ] = c_r;
			colors[3 * _a + 1] = c_g;
			colors[3 * _a + 2] = c_b;

			colors[3 * _b    ] = c_r;
			colors[3 * _b + 1] = c_g;
			colors[3 * _b + 2] = c_b;

			colors[3 * _c    ] = c_r;
			colors[3 * _c + 1] = c_g;
			colors[3 * _c + 2] = c_b;


		}
		count++;
	}
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

void HalfEdgeMesh::optimizeClusters(vector<planarCluster> &clusters)
{
	vector<planarCluster>::iterator start, end, it;
	start = clusters.begin();
	end = clusters.end();

	Normal mean_normal;
	Vertex centroid;

	for(it = start; it != end; it++)
	{
		// Calculated centroid and mean normal of
		// current cluster

		mean_normal = Normal(0, 0, 0);
		centroid = Vertex(0, 0, 0);

		size_t count = (*it).face_count;
		if(count > 50)
		{
			HalfEdgeFace** faces = (*it).faces;

			for(size_t i = 0; i < count; i++)
			{
				HalfEdgeFace* face = faces[i];
				HalfEdge* start_edge, *current_edge;
				start_edge = face->edge;
				current_edge = start_edge;

				mean_normal += face->getInterpolatedNormal();
				//mean_normal += face->getFaceNormal();


				do
				{
					centroid += current_edge->end->position;
					current_edge = current_edge->next;
				} while(start_edge != current_edge);
			}

			//mean_normal /= count;
			mean_normal.normalize();
			//centroid /= 3 * count;

			centroid.x = centroid.x / (3 * count);
			centroid.y = centroid.y / (3 * count);
			centroid.z = centroid.z / (3 * count);

			//cout << mean_normal << " " << centroid << endl;

			// Shift all effected vertices into the calculated
			// plane
			for(size_t i = 0; i < count; i++)
			{
				HalfEdgeFace* face = faces[i];
				HalfEdge* start_edge, *current_edge;
				start_edge = face->edge;
				current_edge = start_edge;

				do
				{

					float distance = (current_edge->end->position - centroid) * mean_normal;
					current_edge->end->position = current_edge->end->position - (mean_normal * distance);

					current_edge = current_edge->next;
				} while(start_edge != current_edge);
			}
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
