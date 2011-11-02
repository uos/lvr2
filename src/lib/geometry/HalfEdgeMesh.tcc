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
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: Thomas Wiemann
 */
#include "ColorVertex.hpp"
namespace lssr
{



    template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::HalfEdgeMesh(PointCloudManager<VertexT, NormalT>* pm)
{
    m_globalIndex = 0;
    m_colorRegions = false;
    m_planesOptimized = false;
    m_pointCloudManager = pm;
}


    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addVertex(VertexT v)
{
    // Create new HalfEdgeVertex and increase vertex counter
    m_vertices.push_back(new HalfEdgeVertex<VertexT, NormalT>(v));
    m_globalIndex++;
}



    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteVertex(HVertex* v)
{
    // Delete HalfEdgeVertex and decrease vertex counter
    m_vertices.erase(find(m_vertices.begin(), m_vertices.end(), v));
    m_globalIndex--;
    delete v;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addNormal(NormalT n)
{
    // Is a vertex exists at globalIndex, save normal
    assert(m_globalIndex == m_vertices.size());
    m_vertices[m_globalIndex - 1]->m_normal = n;
}

    template<typename VertexT, typename NormalT>
HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* HalfEdgeMesh<VertexT, NormalT>::halfEdgeToVertex(HVertex *v, HVertex* next)
{
    HEdge* edge = 0;
    HEdge* cur = 0;

    typename vector<HEdge*>::iterator it;

    for(it = v->in.begin(); it != v->in.end(); it++)
    {
        // Check all incoming edges, if start and end vertex
        // are the same. If they are, save this edge.
        cur = *it;
        if(cur->end == v && cur->start == next)
        {
            edge = cur;
        }

    }

    return edge;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c)
{
    // Create a new face
    HFace* face = new HFace;

    // Create a list of HalfEdges that will be connected
    // with this here. Here we need only to alloc space for
    // three pointers, allocation and linking will be done
    // later.
    HEdge* edges[3];
    edges[0] = edges[1] = edges[2] = 0;

    // Traverse face triangles
    for(int k = 0; k < 3; k++)
    {
        // Pointer to start and end vertex of an edge
        HVertex* current;
        HVertex* next;

        // Map k values to parameters
        switch(k)
        {
            case 0:
                current = m_vertices[a];
                next 	= m_vertices[b];
                break;
            case 1:
                current = m_vertices[b];
                next 	= m_vertices[c];
                break;
            case 2:
                current = m_vertices[c];
                next 	= m_vertices[a];
                break;
        }

        // Try to find an pair edges of an existing face,
        // that points to the current vertex. If such an
        // edge exists, the pair-edge of this edge is the
        // one we need. Update link. If no edge is found,
        // create a new one.
        HEdge* edgeToVertex = halfEdgeToVertex(current, next);

        // If a fitting edge was found, save the pair edge
        // and let it point the the new face
        if(edgeToVertex != 0)
        {
            edges[k] = edgeToVertex->pair;
            edges[k]->face = face;
        }
        else
        {
            // Create new edge and pair
            HEdge* edge = new HEdge;
            edge->face = face;
            edge->start = current;
            edge->end = next;

            HEdge* pair = new HEdge;
            pair->start = next;
            pair->end = current;
            pair->face = 0;

            // Link Half edges
            edge->pair = pair;
            pair->pair = edge;

            // Save outgoing edge
            current->out.push_back(edge);
            next->in.push_back(edge);

            // Save incoming edges
            current->in.push_back(pair);
            next->out.push_back(pair);

            // Save pointer to new edge
            edges[k] = edge;
        }
    }

    for(int k = 0; k < 3; k++)
    {
        edges[k]->next = edges[(k+1) % 3];
    }
    //cout << ":: " << face->index[0] << " " << face->index[1] << " " << face->index[2] << endl;

    face->m_edge = edges[0];
    face->calc_normal();
    m_faces.push_back(face);
    face->m_face_index = m_faces.size();
    //face->m_index[0] = a;
    //face->m_index[1] = b;
    //face->m_index[2] = c;

    //	if(a == 0) {
    //		last_normal = face->normal;
    //	} else {
    //		if(last_normal * face->normal < 0){
    //			face->normal = face->normal * -1;
    //		}
    //	}

}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addFace(HVertex* v1, HVertex* v2, HVertex* v3)
{
    HFace* f = new HFace;

    HEdge* v1v2 = 0;
    HEdge* v2v3 = 0;
    HEdge* v3v1 = 0;

    HEdge* current = 0;

    //check if edge exists between v1, v2 if not add a new one
    if((current = halfEdgeToVertex(v1, v2)) == 0)
    {
        v1v2 = new HEdge;
        HEdge* v2v1 = new HEdge;
        typename vector<HEdge*>::iterator it;

        v1v2->start = v1;
        v1v2->end = v2;
        v1->out.push_back(v1v2);
        v2->in.push_back(v1v2);

        v2v1->start = v2;
        v2v1->end = v1;
        v1->in.push_back(v2v1);
        v2->out.push_back(v2v1);

        v1v2->pair = v2v1;
        v2v1->pair = v1v2;
    }
    else
    {
        if(current->face == 0)
        {
            v1v2 = current;
        }
        else
        {
            v1v2 = current->pair;
        }
    }

    //check if edge exists between v2, v3 if not add a new one
    if((current = halfEdgeToVertex(v2, v3)) == 0)
    {
        v2v3 = new HEdge;
        HEdge* v3v2 = new HEdge;

        v2v3->start = v2;
        v2v3->end = v3;
        v2->out.push_back(v2v3);
        v3->in.push_back(v2v3);

        v3v2->start = v3;
        v3v2->end = v2;
        v2->in.push_back(v3v2);
        v3->out.push_back(v3v2);

        v2v3->pair = v3v2;
        v3v2->pair = v2v3;
    }
    else
    {
        if(current->face == 0)
            v2v3 = current;
        else v2v3 = current->pair;
    }

    //check if edge exists between v3, v1 if not add a new one
    if((current = halfEdgeToVertex(v3, v1)) == 0)
    {
        v3v1 = new HEdge;
        HEdge* v1v3 = new HEdge;

        v3v1->start = v3;
        v3v1->end = v1;
        v3->out.push_back(v3v1);
        v1->in.push_back(v3v1);

        v1v3->start = v1;
        v1v3->end = v3;
        v3->in.push_back(v1v3);
        v1->out.push_back(v1v3);

        v3v1->pair = v1v3;
        v1v3->pair = v3v1;
    }
    else
    {
        if(current->face == 0)
        {
            v3v1 = current;
        }
        else
        {
            v3v1 = current->pair;
        }
    }

    // set next pointers
    typename vector<HEdge*>::iterator it;
    it = v1v2->end->out.begin();
    while(it != v1v2->end->out.end() && *it != v2v3) it++;
    if(it != v1v2->end->out.end())
        v1v2->next = v2v3;
    else
        v1v2->next = v2v3->pair;

    it = v1v2->next->end->out.begin();
    while(it != v1v2->next->end->out.end() && *it != v3v1) it++;

    if(it != v1v2->next->end->out.end())
    {
        v1v2->next->next = v3v1;
    }
    else
    {
        v1v2->next->next = v3v1->pair;
    }

    v1v2->next->next->next = v1v2;

    //set face->m_edge
    f->m_edge = v1v2;

    //set face pointers
    current = v1v2;
    for(int k = 0; k<3; k++,current = current->next)
        current->face = f;

    f->calc_normal();
    m_faces.push_back(f);
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteFace(HFace* f)
{
    //save references to edges and vertices
    HEdge* startEdge = (*f)[0];
    HEdge* nextEdge  = (*f)[1];
    HEdge* lastEdge  = (*f)[2];
    HVertex* p1 = (*f)(0);
    HVertex* p2 = (*f)(1);
    HVertex* p3 = (*f)(2);

    startEdge->face = 0;
    startEdge->next = 0;
    nextEdge->face = 0;
    nextEdge->next = 0;
    lastEdge->face = 0;
    lastEdge->next = 0;

    if(startEdge->pair->face == 0)
    {
        //delete edge and pair
        deleteEdge(startEdge);

        if(p1->out.size()==0) deleteVertex(p1);
        if(p3->out.size()==0) deleteVertex(p3);
    }

    if(nextEdge->pair->face == 0)
    {
        //delete edge and pair
        deleteEdge(nextEdge);

        if(p1->out.size()==0) deleteVertex(p1);
        if(p2->out.size()==0) deleteVertex(p2);
    }

    if(lastEdge->pair->face == 0)
    {
        //delete edge and pair
        deleteEdge(lastEdge);

        if(p3->out.size()==0) deleteVertex(p3);
        if(p2->out.size()==0) deleteVertex(p2);
    }

    //delete face
    m_faces.erase(find(m_faces.begin(), m_faces.end(), f));
    delete f;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteEdge(HEdge* edge, bool deletePair)
{
    //delete references from start point to outgoing edge
    edge->start->out.erase(find(edge->start->out.begin(),edge->start->out.end(), edge));

    //delete references from end point to incoming edge
    edge->end->in.erase(find(edge->end->in.begin(),edge->end->in.end(), edge));

    if(deletePair)
    {
        //delete references from start point to outgoing edge
        edge->pair->start->out.erase(find(edge->pair->start->out.begin(),edge->pair->start->out.end(), edge->pair));

        //delete references from end point to incoming edge
        edge->pair->end->in.erase(find(edge->pair->end->in.begin(),edge->pair->end->in.end(), edge->pair));

        delete edge->pair;
    }

    delete edge;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::collapseEdge(HEdge* edge)
{
    // Save start and end vertex
    HVertex* p1 = edge->start;
    HVertex* p2 = edge->end;

    // Move p1 to the center between p1 and p2 (recycle p1)
    p1->m_position = (p1->m_position + p2->m_position)*0.5;

    //Delete redundant edges
    if (edge->face != 0)
    {
        edge->next->next->pair->pair = edge->next->pair;
        edge->next->pair->pair = edge->next->next->pair;
        deleteEdge(edge->next->next, false);
        deleteEdge(edge->next, false);
    }

    if (edge->pair->face != 0)
    {
        edge->pair->next->next->pair->pair = edge->pair->next->pair;
        edge->pair->next->pair->pair = edge->pair->next->next->pair;
        deleteEdge(edge->pair->next->next, false);
        deleteEdge(edge->pair->next, false);
    }

    // Delete faces
    if(edge->pair->face != 0)
    {
        m_faces.erase(find(m_faces.begin(), m_faces.end(), edge->pair->face));
        delete edge->pair->face;
    }

    if(edge->face != 0)
    {
        m_faces.erase(find(m_faces.begin(), m_faces.end(), edge->face));
        delete edge->face;
    }

    //Delete edge and its' pair
    deleteEdge(edge);

    //Update incoming and outgoing edges of p1
    typename vector<HEdge*>::iterator it;
    it = p2->out.begin();
    while(it != p2->out.end())
    {
        (*it)->start = p1;
        p1->out.push_back(*it);
        it++;
    }

    it = p2->in.begin();
    while(it != p2->in.end())
    {
        (*it)->end = p1;
        p1->in.push_back(*it);
        it++;
    }

    //Delete p2
    deleteVertex(p2);
}


    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::flipEdge(HFace* f1, HFace* f2)
{
    HEdge* commonEdge = 0;
    HEdge* current = f1->m_edge;

    //search the common edge between the two faces
    for(int k = 0; k < 3; k++)
    {
        if (current->pair->face == f2) commonEdge = current;
        current = current->next;
    }

    //return if f1 and f2 are not adjacent in the grid
    if(commonEdge == 0)
    {
        return;
    }

    //flip the common edge
    this->flipEdge(commonEdge);
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::flipEdge(HEdge* edge)
{
    if (edge->pair->face != 0 && edge->face != 0)
    {
        HVertex* newEdgeStart = edge->next->end;
        HVertex* newEdgeEnd = edge->pair->next->end;

        //update next pointers
        edge->next->next->next = edge->pair->next;
        edge->pair->next->next->next = edge->next;

        //create the new edge
        HEdge* newEdge = new HEdge();
        newEdge->start = newEdgeStart;
        newEdge->end = newEdgeEnd;
        newEdge->pair = 0;
        newEdge->next = edge->pair->next->next;
        newEdge->face = edge->pair->next->next->face;
        newEdge->start->out.push_back(newEdge);
        newEdge->end->in.push_back(newEdge);

        HEdge* newPair = new HEdge();
        newPair->start = newEdgeEnd;
        newPair->end = newEdgeStart;
        newPair->pair = newEdge;
        newPair->next = edge->next->next;
        newPair->face = edge->next->next->face;
        newPair->start->out.push_back(newPair);
        newPair->end->in.push_back(newPair);

        newEdge->pair = newPair;

        //update face->edge pointers
        newEdge->face->m_edge = newEdge;
        newPair->face->m_edge = newPair;

        //update next pointers
        edge->next->next = newEdge;
        edge->pair->next->next = newPair;

        //update edge->face pointers
        newEdge->next->face = newEdge->face;
        newEdge->next->next->face = newEdge->face;
        newPair->next->face = newPair->face;
        newPair->next->next->face = newPair->face;

        //recalculate face normals
        newEdge->face->calc_normal();
        newPair->face->calc_normal();

        //delete the old edge
        deleteEdge(edge);
    }
}

    template<typename VertexT, typename NormalT>
int HalfEdgeMesh<VertexT, NormalT>::regionGrowing(HFace* start_face, Region<VertexT, NormalT>* region)
{
    //Mark face as used
    start_face->m_used = true;

    //Add face to region
    region->addFace(start_face);

    int neighbor_cnt = 0;

    //Get the unmarked neighbor faces and start the recursion
    for(int k=0; k<3; k++)
    {
        if((*start_face)[k]->pair->face != 0 && (*start_face)[k]->pair->face->m_used == false)
            ++neighbor_cnt += regionGrowing((*start_face)[k]->pair->face, region);
    }

    return neighbor_cnt;
}

    template<typename VertexT, typename NormalT>
int HalfEdgeMesh<VertexT, NormalT>::regionGrowing(HFace* start_face, NormalT &normal, float &angle, Region<VertexT, NormalT>* region)
{
    //Mark face as used
    start_face->m_used = true;

    //Add face to region
    region->addFace(start_face);

    int neighbor_cnt = 0;

    //Get the unmarked neighbor faces and start the recursion
    for(int k=0; k<3; k++)
    {
        if((*start_face)[k]->pair->face != 0 && (*start_face)[k]->pair->face->m_used == false
                && fabs((*start_face)[k]->pair->face->getFaceNormal() * normal) > angle )
            ++neighbor_cnt += regionGrowing((*start_face)[k]->pair->face, normal, angle, region);
    }

    return neighbor_cnt;
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlanes(
        int iterations,
        float angle,
        int min_region_size,
        int small_region_size,
        bool remove_flickering)
{
    cout << endl << timestamp << "Starting plane optimization with threshold " << angle << endl;

    // Magic numbers
    int default_region_threshold = (int)10*log(m_faces.size());

    // Regions that will be deleted due to size
    vector<Region<VertexT, NormalT>*> smallRegions;

    int region_size = 0;
    m_regions.clear();
    int region_number = 0;

    for(int j = 0; j < iterations; j++)
    {
        cout << timestamp << "Optimizing planes. " <<  j+1 << "th iteration." << endl;

        // Reset all used variables
        for(size_t i=0; i < m_faces.size(); i++)
        {
            m_faces[i]->m_used = false;
        }

        // Find all regions by regionGrowing with normal criteria
        for(size_t i=0; i < m_faces.size(); i++)
        {
            if(m_faces[i]->m_used == false)
            {
                NormalT n = m_faces[i]->getFaceNormal();

                Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(region_number);
                region_size = regionGrowing(m_faces[i], n, angle, region) + 1;

                // Fit big regions into the regression plane
                if(region_size > max(min_region_size, default_region_threshold))
                {
                    region->regressionPlane();
                }

                if(j == iterations-1)
                {
                    // Save too small regions with size smaller than small_region_size
                    if (region_size < small_region_size)
                    {
                        smallRegions.push_back(region);
                    }
                    else
                    {
                        // Save pointer to the region
                        m_regions.push_back(region);
                        region_number++;
                    }
                }
                else
                {
                    delete region;
                }
            }
        }
    }

    // Delete too small regions
    if(small_region_size)
    {
        string msg = timestamp.getElapsedTime() + "Deleting small regions ";
        ProgressBar progress(smallRegions.size(), msg);
        for(size_t i=0; i< smallRegions.size(); i++)
        {
            deleteRegion(smallRegions[i]);
            ++progress;
        }
    }
    cout << endl;

    //Delete flickering faces
    if(remove_flickering)
    {
        vector<HFace*> flickerer;
        for(size_t i=0; i< m_faces.size(); i++)
            if(m_faces[i]->m_region->detectFlicker(m_faces[i]))
            {
                flickerer.push_back(m_faces[i]);
            }

        while(!flickerer.empty())
        {
            deleteFace(flickerer.back());
            flickerer.pop_back();
        }
    }

    m_planesOptimized = true;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteRegion(Region<VertexT, NormalT>* region)
{
    while(! region->m_faces.empty())
        deleteFace(region->m_faces.front());
    delete region;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::removeDanglingArtifacts(int threshold)
{
    vector<Region<VertexT, NormalT>*> todelete;

    for(size_t i=0; i<m_faces.size(); i++)
    {
        if(m_faces[i]->m_used == false)
        {
            Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(0);
            int region_size = regionGrowing(m_faces[i], region) + 1;
            if(region_size <= threshold)
                todelete.push_back(region);
            else
            {
                delete region;
            }
        }
    }

    ///delete dangling artifacts
    string msg = timestamp.getElapsedTime() + "Removing dangling artifacts ";
    ProgressBar progress(todelete.size(), msg);
    for(size_t i=0; i<todelete.size(); i++ )
    {
        deleteRegion(todelete[i]);
        ++progress;
    }

    //reset all used variables
    for(size_t i=0; i<m_faces.size(); i++)
        m_faces[i]->m_used = false;
}

    template<typename VertexT, typename NormalT>
bool HalfEdgeMesh<VertexT, NormalT>::safeCollapseEdge(HEdge* edge)
{
    //try to reject all huetchen
    if(edge->face != 0 && edge->next->pair->face !=0 && edge->next->next->pair->face!=0)
        if(edge->next->pair->next->next == edge->next->next->pair->next->pair)
            return false;
    if(edge->pair->face && edge->pair->next->pair->face && edge->pair->next->next->pair->face)
        if(edge->pair->next->pair->next->next == edge->pair->next->next->pair->next->pair)
            return false;

    //Check for redundant edges
    int edgeCnt = 0;
    for (size_t i = 0; i<edge->start->out.size(); i++)
        if (edge->start->out[i]->end == edge->end)
            edgeCnt++;
    if(edgeCnt != 1)
        return false;

    //Avoid creation of edges without faces
    if( ( edge->face != 0 && edge->next->pair->face == 0 && edge->next->next->pair->face == 0 )
            || ( edge->pair->face != 0 && edge->pair->next->pair->face == 0 && edge->pair->next->next->pair->face == 0 ) )
        return false;

    //Check for triangle hole
    for(size_t o1 = 0; o1<edge->end->out.size(); o1++)
        for(size_t o2 = 0; o2 < edge->end->out[o1]->end->out.size(); o2++)
            if(edge->end->out[o1]->face == 0 && edge->end->out[o1]->end->out[o2]->face == 0 && edge->end->out[o1]->end->out[o2]->end == edge->start)
                return false;

    //Check for flickering
    //Move edge->start and check for flickering
    VertexT origin = edge->start->m_position;
    edge->start->m_position = (edge->start->m_position + edge->end->m_position)*0.5;
    for(size_t o = 0; o<edge->start->out.size(); o++)
        if(edge->start->out[o]->pair->face != edge->pair->face)
            if (edge->start->out[o]->pair->face != 0 && edge->start->out[o]->pair->face->m_region->detectFlicker(edge->start->out[o]->pair->face))
            {
                edge->start->m_position = origin;
                return false;
            }

    //Move edge->end and check for flickering
    origin = edge->end->m_position;
    edge->end->m_position = (edge->start->m_position + edge->end->m_position)*0.5;
    for(size_t o = 0; o<edge->end->out.size(); o++)
        if(edge->end->out[o]->pair->face != edge->pair->face)
            if (edge->end->out[o]->pair->face != 0 && edge->end->out[o]->pair->face->m_region->detectFlicker(edge->end->out[o]->pair->face))
            {
                edge->end->m_position = origin;
                return false;
            }

    //finally collapse the edge
    collapseEdge(edge);

    return true;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::fillHoles(size_t max_size)
{
    //holds all holes to close
    vector<vector<HEdge*> > holes;

    //walk through all edges and start hole finding
    //when pair has no face
    for(size_t i=0; i < m_faces.size(); i++)
    {
        for(int k=0; k<3; k++)
        {
            HEdge* current = (*m_faces[i])[k]->pair;
            if(current->used == false && current->face == 0)
            {
                //needed for contour tracking
                vector<HEdge*> contour;
                HEdge* next = 0;

                //while the contour is not closed
                while(current != 0)
                {
                    next = 0;
                    contour.push_back(current);
                    //to ensure that there is no way back to the same vertex
                    for (size_t e = 0; e<current->start->out.size(); e++)
                    {
                        if (current->start->out[e]->end == current->end)
                        {
                            current->start->out[e]->used = true;
                            current->start->out[e]->pair->used = true;
                        }
                    }
                    current->used = true;

                    typename vector<HEdge*>::iterator it = current->end->out.begin();
                    while(it != current->end->out.end())
                    {
                        //found a new possible edge to trace
                        if ((*it)->used == false && (*it)->face == 0)
                        {
                            next = *it;
                        }
                        it++;
                    }

                    current = next;
                }
                if (2 < contour.size() && contour.size() < max_size)
                {
                    holes.push_back(contour);
                }
            }
        }
    }

    //collapse the holes
    string msg = timestamp.getElapsedTime() + "Filling holes ";
    ProgressBar progress(holes.size(), msg);

    for(size_t h = 0; h<holes.size(); h++)
    {
        vector<HEdge*> current_hole = holes[h];

        //collapse as much edges as possible
        bool collapsedSomething = true;
        while(collapsedSomething)
        {
            collapsedSomething = false;
            for(size_t e = 0; e<current_hole.size() && ! collapsedSomething; e++)
                if(safeCollapseEdge(current_hole[e]))
                {
                    collapsedSomething = true;
                    current_hole.erase(current_hole.begin()+e);
                }
        }

        //add new faces
        while(current_hole.size()>0)
        {
            bool stop = false;
            for(size_t i = 0; i<current_hole.size() && !stop; i++)
                for(size_t j = 0; j<current_hole.size() && !stop; j++)
                    if(current_hole.back()->end == current_hole[i]->start)
                        if(current_hole[i]->end == current_hole[j]->start)
                            if(current_hole[j]->end == current_hole.back()->start)
                            {
                                HFace* f = new HFace();
                                f->m_edge = current_hole.back();
                                current_hole.back()->next = current_hole[i];
                                current_hole[i]->next = current_hole[j];
                                current_hole[j]->next = current_hole.back();
                                for(int e = 0; e<3; e++)
                                {
                                    (*f)[e]->face = f;
                                    current_hole.erase(find(current_hole.begin(), current_hole.end(), (*f)[e]));
                                }
                                (*f)[0]->pair->face->m_region->addFace(f);
                                m_faces.push_back(f);
                                stop = true;
                            }
            if(!stop)
                current_hole.pop_back();
        }
        ++progress;
    }
    cout << endl;
}


    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::dragOntoIntersection(Region<VertexT, NormalT>* plane, Region<VertexT, NormalT>* neighbor_region, VertexT& x, VertexT& direction)
{
    for (size_t i = 0; i<plane->m_faces.size(); i++)
    {
        for(int k=0; k<=2; k++)
        {
            if((*(plane->m_faces[i]))[k]->pair->face != 0 && (*(plane->m_faces[i]))[k]->pair->face->m_region == neighbor_region)
            {
                (*(plane->m_faces[i]))[k]->start->m_position = x + direction * (((((*(plane->m_faces[i]))[k]->start->m_position)-x) * direction) / (direction.length() * direction.length()));
                (*(plane->m_faces[i]))[k]->end->m_position   = x + direction * (((((*(plane->m_faces[i]))[k]->end->m_position  )-x) * direction) / (direction.length() * direction.length()));
            }
        }
    }
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlaneIntersections()
{
    string msg = timestamp.getElapsedTime() + "Optimizing plane intersections ";
    ProgressBar progress(m_regions.size(), msg);

    for (size_t i = 0; i<m_regions.size(); i++)
    {
        if (m_regions[i]->m_inPlane)
            for(size_t j = i+1; j<m_regions.size(); j++)
                if(m_regions[j]->m_inPlane)
                {
                    //calculate intersection between plane i and j

                    NormalT n_i = m_regions[i]->m_normal;
                    NormalT n_j = m_regions[j]->m_normal;

                    //don't improve almost parallel regions - they won't cross in a reasonable distance
                    if (fabs(n_i*n_j) < 0.9)
                    {

                        float d_i = n_i * m_regions[i]->m_stuetzvektor;
                        float d_j = n_j * m_regions[j]->m_stuetzvektor;

                        VertexT direction = n_i.cross(n_j);

                        float denom = direction * direction;
                        VertexT x = ((n_j*d_i - n_i*d_j).cross(direction)) * (1/denom);

                        //drag all points at the border between planes i and j onto the intersection
                        dragOntoIntersection(m_regions[i], m_regions[j], x, direction);
                        dragOntoIntersection(m_regions[j], m_regions[i], x, direction);
                    }
                }
        ++progress;
    }
}

    template<typename VertexT, typename NormalT>
vector<vector<HalfEdgeVertex<VertexT, NormalT>* > > HalfEdgeMesh<VertexT, NormalT>::findAllContours(float epsilon)
{
    vector<vector<HalfEdgeVertex<VertexT, NormalT>* > > contours;
    for (size_t i = 0; i< m_regions.size(); i++)
    {
        if(m_regions[i]->m_inPlane)
        {
            vector<vector<HalfEdgeVertex<VertexT, NormalT>* > > current_contours = m_regions[i]->getContours(epsilon);
            contours.insert(contours.end(), current_contours.begin(), current_contours.begin() + 1);
        }
    }
    return  contours;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::restorePlanes(int min_region_size)
{
    for(size_t r=0; r<m_regions.size(); r++)
        //drag points into the regression plane
        if( m_regions[r]->m_inPlane)
            for(size_t i=0; i<m_regions[r]->m_faces.size(); i++)
            {
                for(int p=0; p<3; p++)
                {
                    float v = ((m_regions[r]->m_stuetzvektor - (*(m_regions[r]->m_faces[i]))(p)->m_position) * m_regions[r]->m_normal) / (m_regions[r]->m_normal * m_regions[r]->m_normal);
                    if(v != 0)
                        (*(m_regions[r]->m_faces[i]))(p)->m_position = (*(m_regions[r]->m_faces[i]))(p)->m_position + (VertexT)m_regions[r]->m_normal * v;
                }
            }

    //start the last region growing
    int region_size = 0;
    m_regions.clear();
    int region_number = 0;
    int default_region_threshold = (int)10*log(m_faces.size());

    // Reset all used variables
    for(size_t i=0; i < m_faces.size(); i++)
    {
    	m_faces[i]->m_used = false;
    }

    // Find all regions by regionGrowing with normal criteria
    for(size_t i=0; i < m_faces.size(); i++)
    {
    	if(m_faces[i]->m_used == false)
    	{
    		NormalT n = m_faces[i]->getFaceNormal();

    		Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(region_number);
    		float almostone = 0.999;
    		region_size = regionGrowing(m_faces[i], n, almostone, region) + 1;

    		if(region_size > max(min_region_size, default_region_threshold))
            {
            	region->regressionPlane();
            }

			// Save pointer to the region
    		m_regions.push_back(region);
    		region_number++;
    	}
    }
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::tester()
{
    cout << "--------------------------------TESTER" << endl;
    //	for(int r=0; r<m_regions.size(); r++)
    //	//		if( m_regions[r]->detectFlicker()) cout << "still flickering" << endl;
    //	for(int r=0; r<m_regions.size(); r++)
    //		if( m_regions[r]->m_inPlane) cout << r << ": " << m_regions[r]->m_regionNumber << endl;
    cout << "----------------------------END TESTER" << endl;

    //    Reset all used variables
    for(int i=0; i<m_faces.size(); i++)
        for(int k=0; k<3; k++)
            (*m_faces[i])[k]->used=false;

    vector<vector<HalfEdgeVertex<VertexT, NormalT>* > > contours = findAllContours(0.01);
    fstream filestr;
    filestr.open ("contours.pts", fstream::out);
    filestr<<"#X Y Z"<<endl;
    for (int i = 0; i<contours.size(); i++)
    {
        vector<HalfEdgeVertex<VertexT, NormalT>* > contour = contours[i];

        HalfEdgeVertex<VertexT, NormalT> first = *(contour.back());

        while (!contour.empty())
        {
            filestr << contour.back()->m_position[0] << " " << contour.back()->m_position[1] << " " << contour.back()->m_position[2] << endl;
            contour.pop_back();
        }

        filestr << first.m_position[0] << " " << first.m_position[1] << " " << first.m_position[2] << endl;

        filestr<<endl<<endl;

    }
    filestr.close();


    //    Reset all used variables
    for(int i=0; i<m_faces.size(); i++)
        for(int k=0; k<3; k++)
            (*m_faces[i])[k]->used=false;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::finalize()
{
    cout << timestamp << "Finalizing mesh." << endl;
    cout << timestamp << "Number of vertices: " << m_vertices.size() << endl;
    cout << timestamp << "Number of faces: " << m_faces.size() << endl;

    boost::unordered_map<HalfEdgeVertex<VertexT, NormalT>*, int> index_map;

    this->m_nVertices 		= (uint32_t)m_vertices.size();
    this->m_nFaces 			= (uint32_t)m_faces.size();

    this->m_vertexBuffer 	= new float[3 * this->m_nVertices];
    this->m_normalBuffer 	= new float[3 * this->m_nVertices];
    this->m_colorBuffer 	= new uchar[3 * this->m_nVertices];

    this->m_indexBuffer 	= new unsigned int[3 * this->m_nFaces];

    typename vector<HVertex*>::iterator vertices_iter = m_vertices.begin();
    typename vector<HVertex*>::iterator vertices_end = m_vertices.end();
    for(size_t i = 0; vertices_iter != vertices_end; ++i, ++vertices_iter)
    {
        this->m_vertexBuffer[3 * i] =     (*vertices_iter)->m_position[0];
        this->m_vertexBuffer[3 * i + 1] = (*vertices_iter)->m_position[1];
        this->m_vertexBuffer[3 * i + 2] = (*vertices_iter)->m_position[2];

        this->m_normalBuffer [3 * i] =     -(*vertices_iter)->m_normal[0];
        this->m_normalBuffer [3 * i + 1] = -(*vertices_iter)->m_normal[1];
        this->m_normalBuffer [3 * i + 2] = -(*vertices_iter)->m_normal[2];

        this->m_colorBuffer  [3 * i] = 0.8;
        this->m_colorBuffer  [3 * i + 1] = 0.8;
        this->m_colorBuffer  [3 * i + 2] = 0.8;

        // map the old index to the new index in the vertexBuffer
        index_map[*vertices_iter] = i;
    }

    typename vector<HalfEdgeFace<VertexT, NormalT>*>::iterator face_iter = m_faces.begin();
    typename vector<HalfEdgeFace<VertexT, NormalT>*>::iterator face_end  = m_faces.end();

    for(size_t i = 0; face_iter != face_end; ++i, ++face_iter)
    {
        this->m_indexBuffer[3 * i]      = index_map[(*(*face_iter))(0)];
        this->m_indexBuffer[3 * i + 1]  = index_map[(*(*face_iter))(1)];
        this->m_indexBuffer[3 * i + 2]  = index_map[(*(*face_iter))(2)];

        int surface_class = 1;
        if ((*face_iter)->m_region != 0)
            surface_class = (*face_iter)->m_region->m_regionNumber;

        float r, g, b;
        if(m_colorRegions)
        {
            r = (uchar) (255 * fabs(cos(surface_class)));
            g = (uchar) (255 * fabs(sin(surface_class * 30)));
            b = (uchar) (255 * fabs(sin(surface_class * 2)));
        }
        else
        {
            r = 0;
            g = 200;
            b = 0;
        }
        this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 0] = r;
        this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 1] = g;
        this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 2] = b;

        this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 0] = r;
        this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 1] = g;
        this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 2] = b;

        this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 0] = r;
        this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 1] = g;
        this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 2] = b;


    }

   this->m_finalized = true;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::regionsToBuffer(
        float **vertex, 
        float **normal,
        uchar **color,
        float **texture,
        unsigned int **index, 
        unsigned int **textureIndex, 
        size_t     &vncSize, 
        size_t   &indexSize, 
        vector<int> &regions)
{
	this->m_nRegions = 0;
	this->m_regionSizeBuffer = new uint[regions.size()];

    /* xxxUsed variables are used to keep track of the array length
     * that was used until now.
     */
    int vncUsed=0, indexUsed=0, facesUsed=0;
    for(size_t h=0; h<regions.size(); ++h)
    {
        int i = regions[h];
        float r = 0, g = 0, b = 0;
        int surface_class = m_regions[regions[h]]->m_regionNumber;

        if(this->m_colorRegions)
        {
            r = (uchar)(255 * fabs(cos(surface_class)));
            g = (uchar)(255 * fabs(sin(surface_class * 30)));
            b = (uchar)(255 * fabs(sin(surface_class * 2)));
        }
        else
        {
        	//calculate an average color for the region
        	vector<VertexT> cv;
        	for(int m=0; m<m_regions[regions[h]]->m_faces.size(); m++)
        	{
        		for(int n=0; n<3; n++)
        		{
        			VertexT current_position = (*m_regions[regions[h]]->m_faces[m])(n)->m_position;
        			int one = 1;
        			this->m_pointCloudManager->getkClosestVertices(current_position, one, cv);
        			r += cv[0].r;
        			g += cv[0].g;
        			b += cv[0].b;
        		}
        	}
        	r /= 3 * m_regions[regions[h]]->m_faces.size();
        	g /= 3 * m_regions[regions[h]]->m_faces.size();
        	b /= 3 * m_regions[regions[h]]->m_faces.size();
        }

        facesUsed = 0;
        for(size_t j=0; j<m_regions[i]->m_faces.size(); ++j)
        {
        	facesUsed++;

            /* a small check to see whether we will run out of memory. */
            if((float)vncUsed / (float)vncSize >= 0.80)
            {
                *vertex = (float*)realloc((*vertex), vncSize*2*sizeof(float));
                *normal = (float*)realloc((*normal), vncSize*2*sizeof(float));
                *color  = (uchar*)realloc((*color),  vncSize*2*sizeof(uchar));
                *texture =(float*)realloc((*texture),vncSize*2*sizeof(float)); 
                vncSize *= 2;
            }

            /* a small check to see whether we will run out of memory. */
            if((float)indexUsed / (float)indexSize >= 0.80)
            {
                *index        = (unsigned int*)realloc((*index), indexSize*2*sizeof(float)); 
                *textureIndex = (unsigned int*)realloc((*textureIndex), indexSize*2*sizeof(float)); 
                indexSize *= 2;
            }

            int oldVncUsed = vncUsed;
            bool skip = false;

            for(int k=0; k<3; k++)
            {
                /*
                 * loops over all 3 vertices of a face and stores the coordinates, normales
                 * and colors in the corresponding arrays for the IO handling.
                 */
                (*vertex)[vncUsed + 0] = (*m_regions[regions[h]]->m_faces[j])(k)->m_position.x;
                (*vertex)[vncUsed + 1] = (*m_regions[regions[h]]->m_faces[j])(k)->m_position.y;
                (*vertex)[vncUsed + 2] = (*m_regions[regions[h]]->m_faces[j])(k)->m_position.z;

                /* check for nan or ninf coordinates!
                */
                if( isnan((*vertex)[vncUsed + 0]) || isnan((*vertex)[vncUsed + 0]) || isnan((*vertex)[vncUsed + 0]) ||
                        isinf((*vertex)[vncUsed + 0]) || isinf((*vertex)[vncUsed + 0]) || isinf((*vertex)[vncUsed + 0]))
                {
                    cerr << "[nan] coordinates!:\n\t: RegNr(external): " << h
                      << "\n\t RegNr(internal): " << m_regions[i]->m_regionNumber
                      << "\n\t vncUsed: " << vncUsed 
                      << "\n\t face nr: " << j
                      << "\n\t" << k << "th vertex\n\t\tSKIPPING!\n"; 
                    skip = true;
                    break;
                }

                /*check for broken triangles! */
                if( k == 2 )
                {
                    /* get the the three points */
                    float x1 = (*vertex)[vncUsed - 6];
                    float y1 = (*vertex)[vncUsed - 5];
                    float z1 = (*vertex)[vncUsed - 4];

                    float x2 = (*vertex)[vncUsed - 3];
                    float y2 = (*vertex)[vncUsed - 2];
                    float z2 = (*vertex)[vncUsed - 1];

                    float x3 = (*vertex)[vncUsed  + 0];
                    float y3 = (*vertex)[vncUsed  + 1];
                    float z3 = (*vertex)[vncUsed  + 2];

                    /* get the distances between the three points */
                    float d12 = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
                    float d23 = sqrt(pow((x3 - x2), 2) + pow((y3 - y2), 2) + pow((z3 - z2), 2));
                    float d13 = sqrt(pow((x3 - x1), 2) + pow((y3 - y1), 2) + pow((z3 - z1), 2));

                    /* If the points lay to close together: Skip this triangle! */
                    if( d12 <= 0.001 || d23 <= 0.001 || d13 <= 0.001)
                    {
                        cerr << "Degenerated Face!:\n\t: RegNr(external): " << h
                          << "\n\t d12: " << d12 
                          << "\n\t d13: " << d13
                          << "\n\t d23: " << d23
                          << "\n\t\tSKIPPING!\n";
                        skip = true;
                        break;
                    }

                }

                (*normal)[vncUsed + 0] = (*m_regions[i]->m_faces[j])(k)->m_normal[0];
                (*normal)[vncUsed + 1] = (*m_regions[i]->m_faces[j])(k)->m_normal[1];
                (*normal)[vncUsed + 2] = (*m_regions[i]->m_faces[j])(k)->m_normal[2];

                (*color)[vncUsed + 0] = r;
                (*color)[vncUsed + 1] = g;
                (*color)[vncUsed + 2] = b;

                (*texture)[vncUsed + 0] = 0.0;
                (*texture)[vncUsed + 1] = 0.0;
                (*texture)[vncUsed + 2] = 0.0;
                vncUsed += 3;

                (*index)[indexUsed+k] = (vncUsed / 3) - 1;

                /* 
                 * texture does not exist because all regions that should be handled here
                 * dont belong to a regression plane. 
                 * the index UINT_MAX points to nowhere, this is necessary for the IO handling when
                 * the mesh is stored
                 */
                (*textureIndex)[indexUsed+k] = UINT_MAX; 
            }
            if( !skip )
            {
            	facesUsed++;
                indexUsed += 3;
            } else 
            {
                vncUsed = oldVncUsed;
            }
        }
        this->m_regionSizeBuffer[h] = facesUsed;
        this->m_nRegions++;
    }
    vncSize = vncUsed;
    indexSize = indexUsed;

    /*
     * Resize all the arrays to the used sizes.
     */
    *vertex = (float*)realloc((*vertex), vncUsed*sizeof(float));
    *normal = (float*)realloc((*normal), vncUsed*sizeof(float));
    *color  = (uchar*)realloc((*color),  vncUsed*sizeof(uchar));
    *texture =(float*)realloc((*texture),vncUsed*sizeof(float)); 
    *index        = (unsigned int*)realloc((*index), indexUsed*sizeof(float)); 
    *textureIndex = (unsigned int*)realloc((*textureIndex), indexUsed*sizeof(float)); 
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::retesselateRegionsToBuffer(
        size_t &vncBufferSize, size_t &indexBufferSize,
        /*int &vncOffset, int &indexOffset, */ vector<int> plane_regions , bool genTextures) {

    int *coordinatesLength = new int;
    int *indexLength = new int;
    float **v  = new float*;
    unsigned int **in = new unsigned int*;
    
    int pointsUsed = vncBufferSize;
    int indicesUsed = indexBufferSize;
    // check all regions if they are to be retesselated (only if they lie in a regression plane!)
    for(size_t h=0; h<plane_regions.size(); ++h)
    {
        int i = plane_regions[h];
        uchar r, g, b;
        int surface_class = m_regions[i]->m_regionNumber;

        if(this->m_colorRegions)
        {
            r = (uchar)(255 * fabs(cos(surface_class)));
            g = (uchar)(255 * fabs(sin(surface_class * 30)));
            b = (uchar)(255 * fabs(sin(surface_class * 2)));
        }
        else
        {
            r = 0;
            g = 200;
            b = 0;
        }

        // If memory used to 75% reallocate!
        if((double)pointsUsed / (double)vncBufferSize >= 0.45 || vncBufferSize==0)
        {
        	if(vncBufferSize==0)
        		vncBufferSize=1024;
            vncBufferSize *= 4*sizeof(float);
            this->m_vertexBuffer       = (float*)realloc(this->m_vertexBuffer, vncBufferSize); 
            this->m_colorBuffer        = (uchar*)realloc(this->m_colorBuffer, vncBufferSize);
            this->m_normalBuffer       = (float*)realloc(this->m_normalBuffer, vncBufferSize); 
            this->m_textureCoordBuffer = (float*)realloc(this->m_normalBuffer, vncBufferSize);
        }


        // If memory i used to 75% reallocate!
        if((double)indicesUsed / (double)indexBufferSize >= 0.45 || indexBufferSize==0)
        {
        	if(indexBufferSize==0)
        		indexBufferSize=1024;
            indexBufferSize *= 4*sizeof(unsigned int);
            this->m_indexBuffer = (unsigned int*)realloc(this->m_indexBuffer, indexBufferSize);
            this->m_textureIndexBuffer = (unsigned int*)realloc(this->m_textureIndexBuffer, indexBufferSize);
        }
        this->m_textureBuffer[this->m_nTextures++] = m_regions[i]->m_regionNumber;

        Tesselator<VertexT, NormalT>::init();
        vector<vector<HVertex*> > contours = m_regions[i]->getContours(0.01);

        Tesselator<VertexT, NormalT>::tesselate(contours);
        Tesselator<VertexT, NormalT>::getFinalizedTriangles(v, in, indexLength, coordinatesLength);

        if(*indexLength > 0 && *coordinatesLength > 0)
        {
            Texture<VertexT, NormalT>* t=NULL;
            if(genTextures)
            {
                t = new Texture<VertexT, NormalT>(m_pointCloudManager, m_regions[i], contours);
                t->save();
            }

            for(int j=0; j< (*coordinatesLength)/3; ++j)
            {
                this->m_vertexBuffer[j*3+pointsUsed+0] = (*v)[j*3+0];
                this->m_vertexBuffer[j*3+pointsUsed+1] = (*v)[j*3+1];
                this->m_vertexBuffer[j*3+pointsUsed+2] = (*v)[j*3+2];

                this->m_normalBuffer[j*3+pointsUsed+0] = m_regions[i]->m_normal[0]; 
                this->m_normalBuffer[j*3+pointsUsed+1] = m_regions[i]->m_normal[1]; 
                this->m_normalBuffer[j*3+pointsUsed+2] = m_regions[i]->m_normal[2]; 

                this->m_colorBuffer[ j*3+pointsUsed+0] = r; 
                this->m_colorBuffer[ j*3+pointsUsed+1] = g; 
                this->m_colorBuffer[ j*3+pointsUsed+2] = b; 

                float u1 = 0;
                float u2 = 0;
                if(t) t->textureCoords(VertexT((*v)[j*3+0], (*v)[j*3+1], (*v)[j*3+2]) ,u1 ,u2);
                this->m_textureCoordBuffer[j*3+pointsUsed+0] = u1;
                this->m_textureCoordBuffer[j*3+pointsUsed+1] = u2;
                this->m_textureCoordBuffer[j*3+pointsUsed+2] = 0;

                /* Check for degenerated Faces! */
                if( (j+1)%3==0 )
                {
                    float x1 = this->m_vertexBuffer[(j-2)*3+0];
                    float y1 = this->m_vertexBuffer[(j-2)*3+1];
                    float z1 = this->m_vertexBuffer[(j-2)*3+2]; 

                    float x2 = this->m_vertexBuffer[(j-1)*3+0];
                    float y2 = this->m_vertexBuffer[(j-1)*3+1];
                    float z2 = this->m_vertexBuffer[(j-1)*3+2]; 

                    float x3 = this->m_vertexBuffer[(j-0)*3+0];
                    float y3 = this->m_vertexBuffer[(j-0)*3+1];
                    float z3 = this->m_vertexBuffer[(j-0)*3+2]; 

                    float d12 = sqrt( pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2) );
                    float d13 = sqrt( pow((x1-x3),2) + pow((y1-y3),2) + pow((z1-z3),2) );
                    float d23 = sqrt( pow((x2-x3),2) + pow((y2-y3),2) + pow((z2-z3),2) );
                    if( d12 <= 0.001 || d13 <= 0.001 || d23 <= 0.001){
                        cout << "Damnit DEAD Face!: " << endl;
                    }

                }
            }

            for(int j=0; j < (*indexLength); ++j)
            {
                this->m_indexBuffer[j+indicesUsed] = ((*in)[j])+pointsUsed/3; 
                this->m_textureIndexBuffer[j+indicesUsed] = m_regions[i]->m_regionNumber;
            }
            pointsUsed  += (*coordinatesLength);
            indicesUsed += *indexLength;

            delete (*v);
            delete (*in);
            if(t) delete t;
        }
    }
    delete indexLength;
    delete coordinatesLength;
    vncBufferSize = pointsUsed;
    indexBufferSize = indicesUsed;
}

    template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::finalizeAndRetesselate(bool genTextures)
{

    cout << timestamp << "Finalizing mesh." << endl;

    this->m_nVertices    = (uint32_t)m_vertices.size();
    this->m_nFaces 	     = (uint32_t)m_faces.size();
    this->m_nTextures    = 0;
    size_t vncBufferSize   = 3 * this->m_nVertices;
    size_t indexBufferSize = 3 * this->m_nFaces;

    // Guess for the array sizes. The correct values are only known after the regions were tesselated.
    this->m_vertexBuffer       = new float[vncBufferSize];
    this->m_normalBuffer       = new float[vncBufferSize];
    this->m_colorBuffer        = new uchar[vncBufferSize];
    this->m_textureCoordBuffer = new float[vncBufferSize];
    this->m_indexBuffer        = new unsigned int[indexBufferSize];
    this->m_textureIndexBuffer = new unsigned int[indexBufferSize];
    this->m_textureBuffer 	   = new unsigned int[m_regions.size()]; 

    // Reset used variable. Necessarsy to use getContours() in tesselator-class. 
    for(size_t j=0; j<m_faces.size(); j++)
        for(int k=0; k<3; k++)
            (*m_faces[j])[k]->used=false;

    /*vector<Region<VertexT, NormalT> * > plane_regions;*/
    vector<int> plane_regions;
    vector<int> normal_regions;

    // check for all regions whether they should be retesselated.
    for(size_t i=0; i<m_regions.size(); ++i)
    {
        if(m_regions[i]->m_inPlane)
        {
            plane_regions.push_back(i);
        } else{
            normal_regions.push_back(i);
        }
    }

    regionsToBuffer(&this->m_vertexBuffer, 
            &this->m_normalBuffer,
            &this->m_colorBuffer,
            &this->m_textureCoordBuffer,
            &this->m_indexBuffer,
            &this->m_textureIndexBuffer,
            vncBufferSize, 
            indexBufferSize, 
            normal_regions);

    cout << timestamp << "Done copying non-plane regions" << endl;

    retesselateRegionsToBuffer(vncBufferSize, indexBufferSize, plane_regions, genTextures);
    cout << timestamp << "Done retesselated plane regions" << endl;

    this->m_nVertices = vncBufferSize/3; 
    this->m_nFaces 	  = indexBufferSize/3;
    this->m_vertexBuffer = (float*)realloc(this->m_vertexBuffer, this->m_nVertices*3*sizeof(float));
    this->m_colorBuffer  = (uchar*)realloc(this->m_colorBuffer,  this->m_nVertices*3*sizeof(uchar));
    this->m_normalBuffer = (float*)realloc(this->m_normalBuffer, this->m_nVertices*3*sizeof(float));
    this->m_textureCoordBuffer = (float*)realloc(this->m_textureCoordBuffer, this->m_nVertices*3*sizeof(float));
    this->m_indexBuffer = (unsigned int*)realloc(this->m_indexBuffer, this->m_nFaces*3*sizeof(unsigned int));
    this->m_textureIndexBuffer = (unsigned int*)realloc(this->m_textureIndexBuffer, this->m_nFaces*3*sizeof(unsigned int));
    this->m_textureBuffer = (unsigned int*)realloc(this->m_textureBuffer, this->m_nTextures*sizeof(unsigned int));

    this->m_finalized = true; 
    cout << timestamp << "Done retesselating: " 						  	 	<< endl;
    cout << timestamp << "[" << this->m_nVertices << "] Vertices." << endl;
    cout << timestamp << "[" << this->m_nFaces << "] Faces." 	   << endl;

} 

} // namespace lssr
