/*
 * HalfEdgeMesh.cpp
 *
 *  Created on: 13.11.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{



template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::HalfEdgeMesh()
{
	m_globalIndex = 0;
	m_colorRegions = false;
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
	typename vector<HVertex*>::iterator it = m_vertices.begin();
	while(*it != v) it++;
	m_vertices.erase(it);
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
	nextEdge->face = 0;
	lastEdge->face = 0;

	typename vector<HEdge*>::iterator it;

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
	typename vector<HalfEdgeFace<VertexT, NormalT>*>::iterator face_iter = m_faces.begin();
	while(*face_iter != f) face_iter++;
	m_faces.erase(face_iter);
	delete f;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteEdge(HEdge* edge, bool deletePair)
{
	typename vector<HEdge*>::iterator it;

	//delete references from start point to outgoing edge
	it = edge->start->out.begin();
	while(*it != edge) it++;
	edge->start->out.erase(it);

	//delete references from end point to incoming edge
	it = edge->end->in.begin();
	while(*it != edge) it++;
	edge->end->in.erase(it);

	if(deletePair)
	{
		//delete references from start point to outgoing edge
		it = edge->pair->start->out.begin();
		while(*it != edge->pair) it++;
		edge->pair->start->out.erase(it);

		//delete references from end point to incoming edge
		it = edge->pair->end->in.begin();
		while(*it != edge->pair) it++;
		edge->pair->end->in.erase(it);

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
	typename vector<HEdge*>::iterator it;

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
	typename	vector<HalfEdgeFace<VertexT, NormalT>*>::iterator face_iter;
	if(edge->pair->face != 0)
	{
		face_iter = m_faces.begin();
		while(*face_iter != edge->pair->face) face_iter++;
		m_faces.erase(face_iter);
		delete edge->pair->face;
	}

	if(edge->face != 0)
	{
		face_iter = m_faces.begin();
		while(*face_iter != edge->face) face_iter++;
		m_faces.erase(face_iter);
		delete edge->face;
	}

	//Delete edge and its' pair
	deleteEdge(edge);

	//Update incoming and outgoing edges of p1
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
int HalfEdgeMesh<VertexT, NormalT>::regionGrowing(HFace* start_face, int region)
{
	//Mark face as used
	start_face->m_region = region;

	int neighbor_cnt = 0;

	//Get the unmarked neighbor faces and start the recursion
	if((*start_face)[0]->pair->face != 0 && (*start_face)[0]->pair->face->m_region == 0)
		++neighbor_cnt += regionGrowing((*start_face)[0]->pair->face, region);

	if((*start_face)[1]->pair->face != 0 && (*start_face)[1]->pair->face->m_region == 0)
		++neighbor_cnt += regionGrowing((*start_face)[1]->pair->face, region);

	if((*start_face)[2]->pair->face != 0 && (*start_face)[2]->pair->face->m_region == 0)
		++neighbor_cnt += regionGrowing((*start_face)[2]->pair->face, region);

	return neighbor_cnt;
}

template<typename VertexT, typename NormalT>
int HalfEdgeMesh<VertexT, NormalT>::regionGrowing(HFace* start_face, NormalT &normal, float &angle, int region)
{
	//Mark face as used
	start_face->m_region = region;

	int neighbor_cnt = 0;

	//Get the unmarked neighbor faces and start the recursion
	if((*start_face)[0]->pair->face != 0 && (*start_face)[0]->pair->face->m_region == 0
			&& fabs((*start_face)[0]->pair->face->getFaceNormal() * normal) > angle )
		++neighbor_cnt += regionGrowing((*start_face)[0]->pair->face, normal, angle, region);

	if((*start_face)[1]->pair->face != 0 && (*start_face)[1]->pair->face->m_region == 0
			&& fabs((*start_face)[1]->pair->face->getFaceNormal() * normal) > angle)
		++neighbor_cnt += regionGrowing((*start_face)[1]->pair->face, normal, angle, region);

	if((*start_face)[2]->pair->face != 0 && (*start_face)[2]->pair->face->m_region == 0
			&& fabs((*start_face)[2]->pair->face->getFaceNormal() * normal) > angle)
		++neighbor_cnt += regionGrowing((*start_face)[2]->pair->face, normal, angle, region);

	return neighbor_cnt;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlanes(
        int iterations,
        float angle,
        int min_region_size,
        int small_region_size
        )
{
    cout << timestamp << "Starting plane optimization with threshold " << angle << endl;
    // Magic numbers
    int default_region_threshold = (int)10*log(m_faces.size());

	// Regions that will be deleted due to size
	vector<HalfEdgeFace<VertexT, NormalT>*> smallRegions;

	int region_size = 0;
	m_regions.clear();

	for(int j = 0; j < iterations; j++)
	{
		cout << timestamp << "Optimizing planes. " <<  j << "th iteration." << endl;
		int region = 1;

		// Reset all region variables
		for(int i=0; i < m_faces.size(); i++)
		{
			m_faces[i]->m_region = 0;
		}
		// Find all regions by regionGrowing with normal criteria
		for(int i=0; i < m_faces.size(); i++)
		{
			if(m_faces[i]->m_region == 0)
			{
				NormalT n = m_faces[i]->getFaceNormal();

				region_size = regionGrowing(m_faces[i], n, angle, region) + 1;

				// Fit big regions into the regression plane
				if(region_size > max(min_region_size, default_region_threshold))
				{
					regressionPlane(region);
				}

				if(j == iterations-1)
				{
					// Save too small regions with size smaller than 7
					if (region_size < small_region_size)
					{
						smallRegions.push_back(m_faces[i]);
					}
					else
					{
					    // Save pointer to the region for fast access
						m_regions.push_back(m_faces[i]);
					}
				}
				region++;
			}
		}
	}

	// Delete too small regions
	if(small_region_size)
	{
	    string msg = timestamp.getElapsedTime() + "Deleting small regions.";
	    ProgressBar progress(smallRegions.size(), msg);
	    for(int i=0; i< smallRegions.size(); i++)
	    {
	        deleteRegionRecursive(smallRegions[i]);
	        ++progress;
	    }
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::regressionPlane(int region)
{
	//collect all faces from the same region
	vector<HalfEdgeFace<VertexT, NormalT>*>    planeFaces;

	for(int i=0; i<m_faces.size(); i++)
	{
		if(m_faces[i]->m_region == region) planeFaces.push_back(m_faces[i]);
	}

//	srand ( time(NULL) );

	VertexT point1;
	VertexT point2;
	VertexT point3;

	//representation of best regression plane by point and normal
	VertexT bestpoint;
	NormalT bestNorm;

	float bestdist = FLT_MAX;
	float dist = 0;

	int iterations = 0;
	int nonimproving_iterations = 0;

	while((nonimproving_iterations < 20) && (iterations < 200))
	{
		//randomly choose 3 disjoint points
		do
		{
			point1 = (*planeFaces[rand() % planeFaces.size()])(0)->m_position;
			point2 = (*planeFaces[rand() % planeFaces.size()])(1)->m_position;
			point3 = (*planeFaces[rand() % planeFaces.size()])(2)->m_position;
		}
		while(point1 == point2 || point2 == point3 || point3 == point1);

		//compute normal of the plane given by the 3 points
		NormalT n0 = (point1 - point2).cross(point1 - point3);
		n0.normalize();

		//compute error to at most 50 other randomly chosen points
		dist = 0;
		for(int i=0; i < min(50, (int)planeFaces.size()); i++)
		{
			VertexT refpoint = (*planeFaces[rand() % planeFaces.size()])(0)->m_position;
			dist += fabs(refpoint * n0 - point1 * n0) / min(50, (int)planeFaces.size());
		}

		//a new optimum is found
		if(dist < bestdist)
		{
			bestdist = dist;

			bestpoint = point1;
			bestNorm = n0;

			nonimproving_iterations = 0;
		}
		else
		{
			nonimproving_iterations++;
		}

		iterations++;
	}

	//drag points into the regression plane
	for(int i=0; i<planeFaces.size(); i++)
	{

		float v = ((bestpoint - (*planeFaces[i])(0)->m_position) * bestNorm) / (bestNorm * bestNorm);
		if(v != 0)
			(*planeFaces[i])(0)->m_position = (*planeFaces[i])(0)->m_position + (VertexT)bestNorm * v;

		v = ((bestpoint - (*planeFaces[i])(1)->m_position) * bestNorm) / (bestNorm * bestNorm);
		if(v != 0)
			(*planeFaces[i])(1)->m_position = (*planeFaces[i])(1)->m_position + (VertexT)bestNorm * v;

		v = ((bestpoint - (*planeFaces[i])(2)->m_position) * bestNorm) / (bestNorm * bestNorm);
		if(v != 0)
			(*planeFaces[i])(2)->m_position = (*planeFaces[i])(2)->m_position + (VertexT)bestNorm * v;

		//change sign of all faces drawn into regression plane
		planeFaces[i]->m_region = -abs(planeFaces[i]->m_region);
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteRegion(int region)
{
	vector<HalfEdgeFace<VertexT, NormalT>*> todelete;

	for(int i=0; i<m_faces.size(); i++)
		if(m_faces[i]->m_region == region)
			todelete.push_back(m_faces[i]);

	for(int i=0; i<todelete.size(); i++)
		deleteFace(todelete[i]);
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteRegionRecursive(HFace* start_face)
{
	int region = start_face->m_region;

	//Mark face as used
	start_face->m_region = 0;

	//Get the unmarked neighbor faces and start the recursion
	for(int k = 0; k < 3; k++)
	{
		if((*start_face)[k]->pair->face != 0 && (*start_face)[k]->pair->face->m_region == region)
		{
			deleteRegionRecursive((*start_face)[k]->pair->face);
		}
	}

	deleteFace(start_face);
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::removeDanglingArtifacts(int threshold)
{
	vector<int> todelete;
	int region = 1;

	for(int i=0; i<m_faces.size(); i++)
	{
		if(m_faces[i]->m_region == 0)
		{
			int region_size = regionGrowing(m_faces[i], region) + 1;
			if(region_size <= threshold)
				todelete.push_back(region);
			region++;
		}
	}

	for(int i=0; i<todelete.size(); i++ )
		deleteRegion(todelete[i]);

	//reset all region variables
	for(int i=0; i<m_faces.size(); i++)
		m_faces[i]->m_region=0;
}

template<typename VertexT, typename NormalT>
vector<HalfEdgeVertex<VertexT, NormalT>* > HalfEdgeMesh<VertexT, NormalT>::simpleDetectHole(HEdge* start)
{
	int region = start->pair->face->m_region;

	HVertex* end = start->start;
	HEdge* current  = start;

	HEdge* next;

	vector<HalfEdgeVertex<VertexT, NormalT>* > contour;
	contour.push_back(current->start);
	while(current->end != end)
	{
		int i = 0;
		typename vector<HEdge*>::iterator it;
		it = current->end->out.begin();

		/* Search for edges without faces and count them */
		while(it != current->end->out.end())
		{
			if((*it)->face == 0)
			{
				next = *it;
				++i;
			}
			++it;
		}

		/* If there are more than one outgoing edges without a face return empty contour */
		if (i != 1 || next->pair->face->m_region != region)
		{
			return vector<HalfEdgeVertex<VertexT, NormalT>* >();
		}
		else
		{
			contour.push_back(current->end);
			current = next;
		}
	}
	return contour;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::fillHole(vector<HVertex*> contour)
{
	//Simulate a face by setting the face pointer of the edge next to the hole
	for(int p=0; p<contour.size(); p++)
	{
		HEdge* edge = halfEdgeToVertex(contour[p], contour[(p+1) % contour.size()]);
		if(edge->face == 0)
			edge->face = edge->pair->face;
		else
			edge->pair->face = edge->face;
	}

	//Just for testing purposes
	HalfEdgeVertex<VertexT, NormalT> newPoint;
	for (int i = 0; i<contour.size(); i++)
		newPoint.m_position += contour[i]->m_position;
	newPoint.m_position /= contour.size();

	for (int i = 0; i<contour.size(); i++)
		contour[i]->m_position = newPoint.m_position;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::dragOntoIntersection(HFace* planeFace, int neighbor_region, VertexT& x, VertexT& direction)
{
	planeFace->m_used = true;
	if(fabs(direction.cross(x-planeFace->getCentroid()).length() / direction.length()) < 10)
	{
		for(int k=0; k<=2; k++)
		{
			if((*planeFace)[k]->pair->face == 0 || (*planeFace)[k]->pair->face->m_region == neighbor_region)
			{
				(*planeFace)[k]->start->m_position = x + direction * (((((*planeFace)[k]->start->m_position)-x) * direction) / (direction.length() * direction.length()));
				(*planeFace)[k]->end->m_position   = x + direction * (((((*planeFace)[k]->end->m_position  )-x) * direction) / (direction.length() * direction.length()));
			}
		}
	}

	for(int k=0; k<=2; k++)
		if( (*planeFace)[k]->pair->face != 0 && planeFace->m_region == (*planeFace)[k]->pair->face->m_region && (*planeFace)[k]->pair->face->m_used == false)
			dragOntoIntersection((*planeFace)[k]->pair->face, neighbor_region, x, direction);
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlaneIntersections()
{
	for (int i = 0; i<m_regions.size(); i++)
		if (m_regions[i]->m_region < 0)
			for(int j = i+1; j<m_regions.size(); j++)
				if(m_regions[j]->m_region < 0)
				{
					//calculate intersection between plane i and j
					NormalT n_i = m_regions[i]->getFaceNormal();
					NormalT n_j = m_regions[j]->getFaceNormal();
					n_i.normalize();
					n_j.normalize();

					if (fabs(n_i*n_j) < 0.9)
					{

						float d_i = n_i * (*m_regions[i])(0)->m_position;
						float d_j = n_j * (*m_regions[j])(0)->m_position;
						float n_i1 = n_i.x;
						float n_i2 = n_i.y;
						float n_j1 = n_j.x;
						float n_j2 = n_j.y;

						float x1 = (d_i/n_i1 - ((n_i2*d_j)/(n_j2*n_i1)))/(1-((n_i2*n_j1)/(n_j2*n_i1)));
						float x2 = (d_j-n_j1*x1)/n_j2;
						float x3 = 0;
						VertexT x (x1, x2, x3);

						VertexT direction = n_i.cross(n_j);

						//drag all points of planes i and j in a certain radius onto the intersection
						for(int k=0; k<m_faces.size(); k++)
							m_faces[k]->m_used=false;
						dragOntoIntersection(m_regions[i], m_regions[j]->m_region, x, direction);

						for(int k=0; k<m_faces.size(); k++)
							m_faces[k]->m_used=false;
						dragOntoIntersection(m_regions[j], m_regions[i]->m_region, x, direction);
					}
				}
}

template<typename VertexT, typename NormalT>
stack<HalfEdgeVertex<VertexT, NormalT>* > HalfEdgeMesh<VertexT, NormalT>::getContour(HEdge* start, float epsilon)
{
	stack<HalfEdgeVertex<VertexT, NormalT>* > contour;

	//check for infeasible input
	if(start->face == 0 || start->pair->face && start->pair->face->m_region == start->face->m_region)
		return contour;

	int region = start->face->m_region;

	HEdge* current = start;
	HEdge* next = 0;

	while(current->used == false)
	{
		//mark edge as used
		current->used = true;
		next = 0;

		//push the next vertex
		contour.push(current->end);

		//find next edge
		for(int i = 0; i<current->end->out.size(); i++)
		{
			if(!current->end->out[i]->used
					&& current->end->out[i]->face && current->end->out[i]->face->m_region == region
					&& (current->end->out[i]->pair->face == 0
							||current->end->out[i]->pair->face	&& current->end->out[i]->pair->face->m_region != region))

				next = current->end->out[i];
		}

		if(next)
		{
			// calculate direction of the current edge
			NormalT currentDirection(current->end->m_position - current->start->m_position);

			//calculate direction of the next edge
			NormalT nextDirection(next->end->m_position - next->start->m_position);

			//Check if we have to remove the top vertex
			if(   fabs(nextDirection.x - currentDirection.x) <= epsilon
			   && fabs(nextDirection.y - currentDirection.y) <= epsilon
			   && fabs(nextDirection.z - currentDirection.z) <= epsilon
			   )
			{
				contour.pop();
			}
			current = next;
		}
	}

	return contour;
}

template<typename VertexT, typename NormalT>
vector<stack<HalfEdgeVertex<VertexT, NormalT>* > > HalfEdgeMesh<VertexT, NormalT>::findAllContours(float epsilon)
{
	vector<stack<HalfEdgeVertex<VertexT, NormalT>* > > contours;
	for(int i=0; i<m_faces.size(); i++){
		if (m_faces[i]->m_region < 0)
			for (int j = 0; j<3; j++)
				if ((*m_faces[i])[j]->used == false && ((*m_faces[i])[j]->pair->face == 0 || (*m_faces[i])[j]->pair->face->m_region != m_faces[i]->m_region))
					contours.push_back(getContour((*m_faces[i])[j], epsilon));
	}
	return  contours;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::finalize()
{
    cout << timestamp << "Finalizing mesh." << endl;
	cout << timestamp << "Number of vertices: " << (uint32_t) m_vertices.size() << endl;
	cout << timestamp << "Number of faces: " << (uint32_t)m_faces.size() << endl;

	boost::unordered_map<HalfEdgeVertex<VertexT, NormalT>*, int> index_map;

	this->m_nVertices 		= (uint32_t)m_vertices.size();
	this->m_nFaces 			= (uint32_t)m_faces.size();

	this->m_vertexBuffer 	= new float[3 * this->m_nVertices];
	this->m_normalBuffer 	= new float[3 * this->m_nVertices];
	this->m_colorBuffer 	= new float[3 * this->m_nVertices];

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
		
		// TODO: Think of classification
		//int surface_class = classifyFace(he_faces[i]);

		int surface_class = 1;
		surface_class = (*face_iter)->m_region;

		float r, g, b;
		if(m_colorRegions)
		{
		   r = fabs(cos(surface_class));
		   g = fabs(sin(surface_class * 30));
		   b = fabs(sin(surface_class * 2));
		}
		else
		{
		    r = 0.0;
		    g = 0.8;
		    b = 0.0;
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

} // namespace lssr
