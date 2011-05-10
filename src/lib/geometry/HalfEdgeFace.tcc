 /*
 * HalfEdgeFace.cpp
 *
 *  Created on: 03.12.2008
 *      Author: Thomas Wiemann
 */
#include "HalfEdgeVertex.hpp"


// Forward declaration
template<typename A, typename B>
class HalfEdgeVertex;

namespace lssr
{

template<typename VertexT, typename NormalT>
HalfEdgeFace<VertexT, NormalT>::HalfEdgeFace(const HalfEdgeFace<VertexT, NormalT> &o){
	m_edge = o.m_edge;
	m_used = o.m_used;

	for(size_t i = 0; i < o.m_indices.size(); i++) m_indices.push_back(o.m_indices[i]);
	for(int i = 0; i < 3; i++) m_index[i] = o.m_index[i];
}

template<typename VertexT, typename NormalT>
float HalfEdgeFace<VertexT, NormalT>::getArea()
{
    vector<VertexT> vertices;
    getVertices(vertices);

    float a = (vertices[0] - vertices[1]).length();
    float b = (vertices[0] - vertices[2]).length();
    float c = (vertices[1] - vertices[2]).length();

    float s = (a + b + c) / 2.0;

    return sqrt(s * (s - a) * (s - b) * (s -c));
}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::calc_normal(){

	VertexT vertices[3];
	HalfEdgeVertex<VertexT, NormalT>* start = m_edge->start;
	HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;

	int c = 0;
	while(current_edge->end != start){
		vertices[c] = current_edge->start->m_position;
		current_edge = current_edge->next;
		c++;
	}
	VertexT diff1 = vertices[0] - vertices[1];
	VertexT diff2 = vertices[0] - vertices[2];
	m_normal = NormalT(diff1.cross(diff2));
}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::interpolate_normal(){

	//reset current normal
	m_normal = NormalT();

	HalfEdgeVertex<NormalT, VertexT>* start = m_edge->start;
	HalfEdge<NormalT, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;

	int c = 0;
	while(current_edge->end != start){
		m_normal += current_edge->start->normal;
		current_edge = current_edge->next;
		c++;
	}

	m_normal.x = m_normal.x / 3.0f;
	m_normal.y = m_normal.y / 3.0f;
	m_normal.z = m_normal.z / 3.0f;

	m_normal.normalize();
}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::getVertexNormals(vector<NormalT> &n){

	HalfEdgeVertex<VertexT, NormalT>* start = m_edge->start;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;
	while(current_edge->end != start){
		n.push_back(current_edge->end->normal);
		//normal += current_edge->start->normal;
		current_edge = current_edge->next;
	}

}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::getVertices(vector<VertexT> &v){

	HalfEdgeVertex<VertexT, NormalT>* start = m_edge->start;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;
	do{
		v.push_back(current_edge->end->position);
		current_edge = current_edge->next;
	} while(current_edge->end != start);
	v.push_back(current_edge->end->position);
}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::getAdjacentFaces(vector<HalfEdgeFace<VertexT, NormalT>*> &adj){

	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current = m_edge;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* pair;
	HalfEdgeFace<VertexT, NormalT>* neighbor;

	do{
		pair = current->pair;
		if(pair != 0){
			neighbor = pair->face;
			if(neighbor != 0){
				adj.push_back(neighbor);
			}
		}
		current = current->next;
	} while(m_edge != current);

}

template<typename VertexT, typename NormalT>
NormalT HalfEdgeFace<VertexT, NormalT>::getFaceNormal(){

	VertexT vertices[3];
	HalfEdgeVertex<VertexT, NormalT>* start = m_edge->start;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;

	int c = 0;
	while(current_edge->end != start){
		vertices[c] = current_edge->start->position;
		current_edge = current_edge->next;
		c++;
	}
	VertexT diff1 = vertices[0] - vertices[1];
	VertexT diff2 = vertices[0] - vertices[2];

	return NormalT(diff1.cross(diff2));

}

template<typename VertexT, typename NormalT>
NormalT HalfEdgeFace<VertexT, NormalT>::getInterpolatedNormal(){

	NormalT return_normal;

	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* start = m_edge;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;

	int c = 0;
	do{
		return_normal += current_edge->start->normal;

		current_edge = current_edge->next;
		c++;
	} while(current_edge != start);

	return_normal.normalize();
	return return_normal;

}

template<typename VertexT, typename NormalT>
VertexT HalfEdgeFace<VertexT, NormalT>::getCentroid(){
	vector<VertexT> vert;
	getVertices(vert);

	VertexT centroid;

	for(size_t i = 0; i < vert.size(); i++){
		centroid += vert[i];
	}

	if(vert.size() > 0){
		centroid.x = centroid.x / vert.size();
		centroid.y = centroid.y / vert.size();
		centroid.z = centroid.z / vert.size();
	} else {
		cout << "Warning: HalfEdgeFace::getCentroid: No vertices found." << endl;
		return VertexT();
	}

	return centroid;
}

} // namespace lssr

