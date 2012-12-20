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
 * HalfEdgeFace.tcc
 *
 *  @date 03.12.2008
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
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
}

template<typename VertexT, typename NormalT>
HalfEdgeFace<VertexT, NormalT>::~HalfEdgeFace()
{
	if(m_region != 0)
	{
		m_region->removeFace(this);
	}
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
	while(current_edge->end != start)
	{
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
	while(current_edge->end != start)
	{
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
	while(current_edge->end != start)
	{
		n.push_back(current_edge->end->normal);
		current_edge = current_edge->next;
	}

}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::getVertices(vector<VertexT> &v){

	HalfEdgeVertex<VertexT, NormalT>* start = m_edge->start;
	HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* current_edge = m_edge;
	do
	{
		v.push_back(current_edge->end->m_position);
		current_edge = current_edge->next;
	} while(current_edge->end != start);
	v.push_back(current_edge->end->m_position);
}

template<typename VertexT, typename NormalT>
void HalfEdgeFace<VertexT, NormalT>::getAdjacentFaces(vector<HalfEdgeFace<VertexT, NormalT>*> &adj){

	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* current = m_edge;
	HalfEdge<VertexT, HalfEdgeFace<VertexT, NormalT> >* pair;
	HalfEdgeFace<VertexT, NormalT>* neighbor;

	do
	{
		pair = current->pair;
		if(pair != 0)
		{
			neighbor = pair->face;
			if(neighbor != 0)
			{
				adj.push_back(neighbor);
			}
		}
		current = current->next;
	} while(m_edge != current);

}

template<typename VertexT, typename NormalT>
NormalT HalfEdgeFace<VertexT, NormalT>::getFaceNormal(){

	VertexT p0 = (*this)(0)->m_position;
	VertexT p1 = (*this)(1)->m_position;
	VertexT p2 = (*this)(2)->m_position;

	VertexT diff1 = p0 - p1;
	VertexT diff2 = p0 - p2;

	return NormalT(diff1.cross(diff2));
}

template<typename VertexT, typename NormalT>
NormalT HalfEdgeFace<VertexT, NormalT>::getInterpolatedNormal(){
	NormalT return_normal = NormalT();

	for (int i = 0; i < 3; i++)
	{
		return_normal += (*this)(i)->m_normal;
	}

	return_normal /= 3.0f;

	return_normal.normalize();
	return return_normal;
}

template<typename VertexT, typename NormalT>
VertexT HalfEdgeFace<VertexT, NormalT>::getCentroid(){
	vector<VertexT> vert;
	getVertices(vert);

	VertexT centroid;

	for(size_t i = 0; i < vert.size(); i++)
	{
		centroid += vert[i];
	}

	if(vert.size() > 0)
	{
		centroid.x = centroid.x / vert.size();
		centroid.y = centroid.y / vert.size();
		centroid.z = centroid.z / vert.size();
	}
	else
	{
		cout << "Warning: HalfEdgeFace::getCentroid: No vertices found." << endl;
		return VertexT();
	}

	return centroid;
}

template<typename VertexT, typename NormalT>
HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* 
HalfEdgeFace<VertexT, NormalT>::operator[](const int &index) const{
	switch(index)
	{
	case 0:
		return this->m_edge;
	case 1:
		return this->m_edge->next;
	case 2:
	    if(!this->m_edge->next)
	    {
	        cout << timestamp << "Degerated Face!" << endl;
	        return 0;
	    }
		return this->m_edge->next->next;
	}
    return 0;
}

template<typename VertexT, typename NormalT>
HalfEdgeVertex<VertexT, NormalT>* 
HalfEdgeFace<VertexT, NormalT>::operator()(const int &index) const{
	switch(index)
	{
	case 0:
		return this->m_edge->end;
	case 1:
        if(!this->m_edge->next)
        {
            cout << timestamp << "Degerated Face!" << endl;
            return 0;
        }
		return this->m_edge->next->end;
	case 2:

        if(!this->m_edge->next)
        {
            cout << timestamp << "Degerated Face!" << endl;
            return 0;
        }
        if(!this->m_edge->next->next)
        {
            cout << timestamp << "Degerated Face!" << endl;
            return 0;
        }
		return this->m_edge->next->next->end;
	}
    return 0;
}
template<typename VertexT, typename NormalT>
float HalfEdgeFace<VertexT, NormalT>::getD()
{
	NormalT normal = getFaceNormal();
	VertexT vertex = this->m_edge->end->m_position;

	return -(normal * vertex);
}


template<typename VertexT, typename NormalT>
bool HalfEdgeFace<VertexT, NormalT>::isBorderFace()
{
	if(this->m_edge->pair->face == 0) return true;
	if(this->m_edge->next->pair->face == 0) return true;
	if(this->m_edge->next->next->pair->face == 0) return true;
	return false;
}

} // namespace lssr

