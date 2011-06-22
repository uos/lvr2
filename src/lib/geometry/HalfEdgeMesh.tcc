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
}

//int HalfEdgeMesh::classifyFace(HalfEdgeFace* f)
//{
//	Normal n = f->getInterpolatedNormal();
//	Normal n_ceil(0.0, 1.0, 0.0);
//	Normal n_floor(0.0, -1.0, 0.0);
//
//	if(n_ceil * n > 0.98) return 1;
//	if(n_floor * n > 0.98) return 2;
//
//	float radius = sqrt(n.x * n.x + n.z * n.z);
//
//	if(radius > 0.95) return 3;
//
//	return 0;
//}

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

	for(it = v->in.begin(); it != v->in.end(); it++){
		// Check all incoming edges, if start and end vertex
		// are the same. If they are, save this edge.
		cur = *it;
		if(cur->end == v && cur->start == next){
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
	face->m_used = false;

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
		if(edgeToVertex != 0){
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


	for(int k = 0; k < 3; k++){
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
		//delete references from vertices to edges
		it = p1->in.begin();
		while(*it != startEdge) it++;
		p1->in.erase(it);

		it = p1->out.begin();
		while(*it != startEdge->pair) it++;
		p1->out.erase(it);

		it = p3->in.begin();
		while(*it != startEdge->pair) it++;
		p3->in.erase(it);

		it = p3->out.begin();
		while(*it != startEdge) it++;
		p3->out.erase(it);

		//delete edge and pair
		delete startEdge->pair;
		delete startEdge;

		if(p1->out.size()==0) deleteVertex(p1);
		if(p3->out.size()==0) deleteVertex(p3);
	}

	if(nextEdge->pair->face == 0)
	{
		//delete references from vertices to edges
		it = p2->in.begin();
		while(*it != nextEdge) it++;
		p2->in.erase(it);

		it = p2->out.begin();
		while(*it != nextEdge->pair) it++;
		p2->out.erase(it);

		it = p1->in.begin();
		while(*it != nextEdge->pair) it++;
		p1->in.erase(it);

		it = p1->out.begin();
		while(*it != nextEdge) it++;
		p1->out.erase(it);

		//delete edge and pair
		delete nextEdge->pair;
		delete nextEdge;

		if(p1->out.size()==0) deleteVertex(p1);
		if(p2->out.size()==0) deleteVertex(p2);
	}

	if(lastEdge->pair->face == 0)
	{
		//delete references from vertices to edges
		it = p3->in.begin();
		while(*it != lastEdge) it++;
		p3->in.erase(it);

		it = p3->out.begin();
		while(*it != lastEdge->pair) it++;
		p3->out.erase(it);

		it = p2->in.begin();
		while(*it != lastEdge->pair) it++;
		p2->in.erase(it);

		it = p2->out.begin();
		while(*it != lastEdge) it++;
		p2->out.erase(it);

		//delete edge and pair
		delete lastEdge->pair;
		delete lastEdge;

		if(p3->out.size()==0) deleteVertex(p3);
		if(p2->out.size()==0) deleteVertex(p2);
	}

	typename	vector<HalfEdgeFace<VertexT, NormalT>*>::iterator face_iter = m_faces.begin();
	while(*face_iter != f)
		face_iter++;

	m_faces.erase(face_iter);
	delete f;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::collapseEdge(HEdge* edge)
{
	edge = m_faces[21]->m_edge;

	// Save start and end vertex
	HVertex* p1 = edge->start;
	HVertex* p2 = edge->end;

	// Move p1 to the center between p1 and p2 (recycle p1)
	p1->m_position = (p1->m_position + p2->m_position)*0.5;

	// Delete faces
	if(edge->pair->face != 0) deleteFace(edge->pair->face);
	if(edge->face != 0) deleteFace(edge->face);

	//Delete redundant edges
	typename vector<HEdge*>::iterator it;

	it = edge->next->next->start->out.begin();
	while(*it != edge->next->next) it++;
	edge->next->next->start->out.erase(it);
	it = edge->next->next->end->in.begin();
	while(*it != edge->next->next) it++;
	edge->next->next->end->in.erase(it);
	edge->next->next->pair->pair = edge->next->pair;

	it = edge->next->start->out.begin();
	while(*it != edge->next) it++;
	edge->next->start->out.erase(it);
	it = edge->next->end->in.begin();
	while(*it != edge->next) it++;
	edge->next->end->in.erase(it);
	edge->next->pair->pair = edge->next->next->pair;
	delete (edge->next->next);
	delete (edge->next);


	it = edge->pair->next->next->start->out.begin();
	while(*it != edge->pair->next->next) it++;
	edge->pair->next->next->start->out.erase(it);
	it = edge->pair->next->next->end->in.begin();
	while(*it != edge->pair->next->next) it++;
	edge->pair->next->next->end->in.erase(it);
	edge->pair->next->next->pair->pair = edge->pair->next->pair;

	it = edge->pair->next->start->out.begin();
	while(*it != edge->pair->next) it++;
	edge->pair->next->start->out.erase(it);
	it = edge->pair->next->end->in.begin();
	while(*it != edge->pair->next) it++;
	edge->pair->next->end->in.erase(it);
	edge->pair->next->pair->pair = edge->pair->next->next->pair;
	delete (edge->pair->next->next);
	delete (edge->pair->next);

	it = edge->start->out.begin();
	while(*it != edge) it++;
	edge->start->out.erase(it);

	it = edge->end->in.begin();
	while(*it != edge) it++;
	edge->end->in.erase(it);

	it = edge->pair->start->out.begin();
	while(*it != edge->pair) it++;
	edge->pair->start->out.erase(it);

	it = edge->pair->end->in.begin();
	while(*it != edge->pair) it++;
	edge->pair->end->in.erase(it);

	delete edge->pair;
	delete edge;

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
	for(int k = 0; k < 3; k++){
		if (current->pair->face == f2) commonEdge = current;
		current = current->next;
	}

	//return if f1 and f2 are not adjacent in the grid
	if(commonEdge == 0)
		return;

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
		typename vector<HEdge*>::iterator it;

		it = edge->start->out.begin();
		while(*it != edge) it++;
		edge->start->out.erase(it);

		it = edge->end->in.begin();
		while(*it != edge) it++;
		edge->end->in.erase(it);

		it = edge->pair->start->out.begin();
		while(*it != edge->pair) it++;
		edge->pair->start->out.erase(it);

		it = edge->pair->end->in.begin();
		while(*it != edge->pair) it++;
		edge->pair->end->in.erase(it);

		delete edge->pair;
		delete edge;
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::finalize()
{
	cout<<"Number of vertices: "<<(uint32_t)m_vertices.size()<<" Number of faces: "<<(uint32_t)m_faces.size()<<endl;
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
		//cout << "VERTEX: -------------------------------" << endl;
		//cout << vertices_iter->second->m_position[0] << endl;
		//cout << vertices_iter->second->m_position[1] << endl;
		//cout << vertices_iter->second->m_position[2] << endl;

		//cout << vertices_iter->second->m_normal[0] << endl;
		//cout << vertices_iter->second->m_normal[1] << endl;
		//cout << vertices_iter->second->m_normal[2] << endl;

		//cout << (vertices_iter->first) << endl;
		//cout << "END: ----------------------------------" << endl;

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
		switch(surface_class)
		{
		case 1:
			this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i]  * 3 + 2] = 1.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 2] = 1.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 2] = 1.0;

			break;
		case 2:
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 0] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 2] = 0.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 0] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 2] = 0.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 0] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 1] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 2] = 0.0;

			break;
		case 3:
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 1] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i] * 3 + 2] = 0.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 1] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 1] * 3 + 2] = 0.0;

			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 0] = 0.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 1] = 1.0;
			this->m_colorBuffer[this->m_indexBuffer[3 * i + 2] * 3 + 2] = 0.0;

			break;
		}
	}

	this->m_finalized = true;
}




//void HalfEdgeMesh::finalize(vector<planarCluster> &planes)
//{
//	if(!finalized) finalize();
//
//	// Create a color gradient
//	float r[255];
//	float g[255];
//	float b[255];
//
//	float c_r, c_g, c_b;
//
//	for(int i = 0; i < 255; i++)
//	{
//		 r[i] = (252 - i % 64 * 4) / 255.0;
//		 g[i] =  (32 + i % 32 * 6) / 255.0;
//		 b[i] =  (64 + i % 64 * 3) / 255.0;
//	}
//
//	// Change colors according to clustering
//	int count = 0;
//	for(size_t i = 0; i < planes.size(); i++)
//	{
//		planarCluster cluster = planes[i];
//		for(size_t j = 0; j < cluster.face_count; j++)
//		{
//			if(cluster.face_count > 50)
//			{
//				c_r = r[count % 255];
//				c_g = g[count % 255];
//				c_b = b[count % 255];
////				c_r = 0.0;
////				c_g = 0.6;
////				c_b = 0.0;
//			}
//			else
//			{
//				c_r = 0.0;
//				c_g = 0.6;
//				c_b = 0.0;
//
//			}
//			HalfEdgeFace* f = cluster.faces[j];
//
//			// Get vertex indices
//			int _a = f->index[0];
//			int _b = f->index[1];
//			int _c = f->index[2];
//
////			cout << r[count % 255] << " "
////			     << g[count % 255] << " "
////			     << b[count % 255] << endl;
//
//			colors[3 * _a    ] = c_r;
//			colors[3 * _a + 1] = c_g;
//			colors[3 * _a + 2] = c_b;
//
//			colors[3 * _b    ] = c_r;
//			colors[3 * _b + 1] = c_g;
//			colors[3 * _b + 2] = c_b;
//
//			colors[3 * _c    ] = c_r;
//			colors[3 * _c + 1] = c_g;
//			colors[3 * _c + 2] = c_b;
//
//
//		}
//		count++;
//	}
//}
//
//bool HalfEdgeMesh::isFlatFace(HalfEdgeFace* face){
//
//	int index = face->mcIndex;
//
//	//WALL
//	if(index == 240 || index == 15 || index == 153 || index == 102){
//
//		return true;
//
//	}
//	//FLOOR
//	else if(index == 204){
//
//		return true;
//
//	}
//	//CEIL
//	else if (index == 51){
//
//		return true;
//
//	}
//	//DOORS
//	else if (index == 9 || index == 144 || index == 96 || index == 6){
//
//		return true;
//
//	}
//	//OTHER FLAT POLYGONS
//	else if(index ==  68 || index == 136 || index ==  17 || index ==  34 || //Variants of MC-Case 2
//			index == 192 || index ==  48 || index ==  12 || index ==   3 ){
//
//		return true;
//
//	} else if (index ==  63 || index == 159 || index == 207 || index == 111 || //Variants of MC-Case 2 (compl)
//			index == 243 || index == 249 || index == 252 || index == 246 ||
//			index == 119 || index == 187 || index == 221 || index == 238){
//		return true;
//
//	}
//
//	return false;
//}


//void HalfEdgeMesh::getArea(set<HalfEdgeFace*> &faces, HalfEdgeFace* face, int depth, int max){
//
//	vector<HalfEdgeFace*> adj;
//	face->getAdjacentFaces(adj);
//
//	vector<HalfEdgeFace*>::iterator it;
//	for(it = adj.begin(); it != adj.end(); it++){
//		faces.insert(*it);
//		if(depth < max){
//			getArea(faces, *it, depth + 1, max);
//		}
//	}
//
//}

//void HalfEdgeMesh::shiftIntoPlane(HalfEdgeFace* f){
//
//	HalfEdge* edge  = f->edge;
//	HalfEdge* start = edge;
//
//	do{
//		float d = (current_v - edge->end->position) * current_n;
//		edge->end->position = edge->end->position + (current_n * d);
//		edge = edge -> next;
//	} while(edge != start);
//
//}

//bool HalfEdgeMesh::check_face(HalfEdgeFace* f0, HalfEdgeFace* current){
//
//	//Calculate Plane representation
//	Normal n_0 = f0->getInterpolatedNormal();
//	Vertex p_0 = f0->getCentroid();
//
//	//Calculate needed parameters
//	float  cos_angle = n_0 * current->getInterpolatedNormal();
//
//	//Decide using given thresholds
//	//if(distance < 8.0 && cos_angle > 0.98) return true;
//	//if(cos_angle > 0.98) return true; <--- Standard lssr value
//	if(cos_angle > 0.88) return true;
//
//	//Return false if face is not in plane
//	return false;
//}

//void HalfEdgeMesh::cluster(vector<planarCluster> &planes))
//{
//	for(size_t i = 0; i < he_faces.size(); i++)
//	{
//		HalfEdgeFace* current_face = he_faces[i];
//
//		if(!current_face->used)
//		{
//
//			planarCluster cluster;
//			cluster.face_count = 0;
//			cluster.faces = 0;
//
//			vector<HalfEdgeFace*> faces;
//
//			check_next_neighbor(current_face, current_face, 0, faces);
//
//			// Copy faces into cluster struct
//			cluster.face_count = faces.size();
//			cluster.faces = new HalfEdgeFace*[faces.size()];
//
//			for(size_t i = 0; i < faces.size(); i++)
//			{
//				cluster.faces[i] = faces[i];
//			}
//
//			planes.push_back(cluster);
//		}
//
//	}
//}

//void HalfEdgeMesh::classifyCluster(vector<planarCluster> &planes, list<list<planarCluster> > &objectCandidates)
//{
//
//    // Tmp marker vector for checked cluster
//    vector<bool> markers(planes.size(), false);
//
//    // Iterate through all clusters and check the following
//    // constaints:
//    //
//    // (1) Cluster size is bigger than s_min (to filter outliers)
//    // (2) Cluster size is smaller than s_max (to filter floor and ceiling)
//    //
//    // Than for all clusters recursively check if there are
//    // other clusters with a maximum distance of d_max between their
//    // COGs.
//
//    list<planarCluster> clustercluster;
//    int c = 0;
//    for(size_t i = 0; i < planes.size(); i++)
//    {
//        //cout << i << " / " << planes.size() << endl;
//        planarCluster c = planes[i];
//        markers[i] = true;
//        clustercluster.clear();
//        findNextClusterInRange(i, planes, c, clustercluster, markers);
//
//        if(clustercluster.size())
//        {
//            cout << clustercluster.size() << endl;
//            objectCandidates.push_back(clustercluster);
//        }
//    }
//
//
//}

//void HalfEdgeMesh::findNextClusterInRange(int s, vector<planarCluster> &clusters, planarCluster &start, list<planarCluster> &clustercluster, vector<bool> &markers)
//{
//    float d_max = 30000000;    // Max distance between clusters
//    float a_min = 10000;   // Min cluster size
//    float a_max = 20000;   // Max cluster size
//
//    Normal start_normal;
//    Vertex start_centroid;
//    float start_area;
//
//    // Calc paramters of current cluster
//    start.calcParameters(start_area, start_centroid, start_normal);
//
//    //cout << start_area << endl;
//
//    // Ok, this check is redundant, but it is more comfartable for
//    // testing to have the magic numbers in just one method...
//    if(start_area > a_max && start_area > a_min ) return;
//
//    // Find next unused cluster that is in ranges
//    // and has a suitable size.
//    for(size_t i = s; i < clusters.size(); i++)
//    {
//        if(!markers[i])
//        {
//            Normal next_normal;
//            Vertex next_centroid;
//            float next_area;
//            clusters[i].calcParameters(next_area, next_centroid, next_normal);
//
//            if((next_centroid - start_centroid).length() < d_max) return;
//            // Check area criterion
//            if((next_area < a_max) && (next_area > a_min))
//            {
//
//                  markers[i] = true;
//                  clustercluster.push_back(clusters[i]);
//                  findNextClusterInRange(i, clusters, clusters[i], clustercluster, markers);
//            }
//            markers[i] = true;
//        }
//    }
//}

//void HalfEdgeMesh::optimizeClusters(vector<planarCluster> &clusters)
//{
//	vector<planarCluster>::iterator start, end, it;
//	start = clusters.begin();
//	end = clusters.end();
//
//	Normal mean_normal;
//	Vertex centroid;
//
//	for(it = start; it != end; it++)
//	{
//		// Calculated centroid and mean normal of
//		// current cluster
//
//		mean_normal = Normal(0, 0, 0);
//		centroid = Vertex(0, 0, 0);
//
//		size_t count = (*it).face_count;
//		if(count > 50)
//		{
//			HalfEdgeFace** faces = (*it).faces;
//
//			for(size_t i = 0; i < count; i++)
//			{
//				HalfEdgeFace* face = faces[i];
//				HalfEdge* start_edge, *current_edge;
//				start_edge = face->edge;
//				current_edge = start_edge;
//
//				mean_normal += face->getInterpolatedNormal();
//				//mean_normal += face->getFaceNormal();
//
//
//				do
//				{
//					centroid += current_edge->end->position;
//					current_edge = current_edge->next;
//				} while(start_edge != current_edge);
//			}
//
//			//mean_normal /= count;
//			mean_normal.normalize();
//			//centroid /= 3 * count;
//
//			centroid.x = centroid.x / (3 * count);
//			centroid.y = centroid.y / (3 * count);
//			centroid.z = centroid.z / (3 * count);
//
//			//cout << mean_normal << " " << centroid << endl;
//
//			// Shift all effected vertices into the calculated
//			// plane
//			for(size_t i = 0; i < count; i++)
//			{
//				HalfEdgeFace* face = faces[i];
//				HalfEdge* start_edge, *current_edge;
//				start_edge = face->edge;
//				current_edge = start_edge;
//
//				do
//				{
//
//					float distance = (current_edge->end->position - centroid) * mean_normal;
//					current_edge->end->position = current_edge->end->position - (mean_normal * distance);
//
//					current_edge = current_edge->next;
//				} while(start_edge != current_edge);
//			}
//		}
//	}
//}

//void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0,
//		                               HalfEdgeFace* face,
//		                               HalfEdge* edge,
//		                               HalfEdgePolygon* polygon){
//
//	face->used = true;
//	polygon->add_face(face, edge);
//
//    //Iterate through all surrounding faces
//	HalfEdge* start_edge   = face->edge;
//	HalfEdge* current_edge = face->edge;
//	HalfEdge* pair         = current_edge->pair;
//	HalfEdgeFace* current_neighbor;
//
//	do{
//		pair = current_edge->pair;
//		if(pair != 0){
//			current_neighbor = pair->face;
//			if(current_neighbor != 0){
//				if(check_face(f0, current_neighbor) && !current_neighbor->used){
//					check_next_neighbor(f0, current_neighbor, current_edge, polygon);
//				}
//			}
//		}
//		current_edge = current_edge->next;
//	} while(start_edge != current_edge);
//
//
//}

//void HalfEdgeMesh::check_next_neighbor(HalfEdgeFace* f0,
//		                               HalfEdgeFace* face,
//		                               HalfEdge* edge,
//		                               vector<HalfEdgeFace*> &faces){
//
//	face->used = true;
//	faces.push_back(face);
//
//    //Iterate through all surrounding faces
//	HalfEdge* start_edge   = face->edge;
//	HalfEdge* current_edge = face->edge;
//	HalfEdge* pair         = current_edge->pair;
//	HalfEdgeFace* current_neighbor;
//
//	do{
//		pair = current_edge->pair;
//		if(pair != 0){
//			current_neighbor = pair->face;
//			if(current_neighbor != 0){
//				if(check_face(f0, current_neighbor) && !current_neighbor->used){
//					check_next_neighbor(f0, current_neighbor, current_edge, faces);
//				}
//			}
//		}
//		current_edge = current_edge->next;
//	} while(start_edge != current_edge);
//
//
//}


//void HalfEdgeMesh::generate_polygons(){
//
//	vector<HalfEdgePolygon*>::iterator it;
//	HalfEdgePolygon* polygon;)
//
//	for(it =  hem_polygons.begin();
//		it != hem_polygons.end();
//		it++)
//	{
//		polygon = *it;
//		polygon->fuse_edges();
//	}

//}

//void HalfEdgeMesh::extract_borders(){
//
//	HalfEdgeFace*       current_face;
//	HalfEdgePolygon*    current_polygon;
//	vector<HalfEdgeFace*>::iterator face_iterator;
//
//	unsigned int biggest_size = 0;
//
//	int c = 0;
//	for(face_iterator = he_faces.begin(); face_iterator != he_faces.end(); face_iterator++){
//		if(c % 10000 == 0) cout << "Extracting Borders: " << c << " / " << he_faces.size() << endl;
//		current_face = *face_iterator;
//		if(!current_face->used){
//
//			current_n = current_face->normal;
//			current_d = current_face->edge->start->position * current_n;
//			current_v = current_face->edge->start->position;
//
//			current_polygon = new HalfEdgePolygon();
//			check_next_neighbor(current_face, current_face, 0, current_polygon);
//			current_polygon->generate_list();
//			//current_polygon->fuse_edges();
//			//current_polygon->test();
//
//			hem_polygons.push_back(current_polygon);
//			if(current_polygon->faces.size() > biggest_size){
//				biggest_size = current_polygon->faces.size();
//				biggest_polygon = current_polygon;
//			}
//
//		}
//		c++;
//	}
//
//	cout << "BIGGEST POLYGON: " << biggest_polygon << endl;
//
//}

//void HalfEdgeMesh::create_polygon(vector<int> &polygon, hash_map<unsigned int, HalfEdge*>* edges){
//
//
//}
//
//void HalfEdgeMesh::write_polygons(string filename){
//
//	cout << "WRITE" << endl;
//
//	ofstream out(filename.c_str());
//
//	vector<HalfEdgePolygon*>::iterator p_it;
//	//multiset<HalfEdge*>::iterator it;
//	EdgeMapIterator it;
//
//	for(it  = biggest_polygon->edges.begin();
//		it != biggest_polygon->edges.end();
//		it++)
//	{
//		HalfEdge* e = it->second;
//		out << "BEGIN" << endl;
//		out << e->start->position.x << " " << e->start->position.y << " " << e->start->position.z << endl;
//		out << e->end->position.x   << " " << e->end->position.y   << " " << e->end->position.z   << endl;
//		out << "END" << endl;
//	}
//
//	//biggest_polygon->fuse_edges();
//
//	for(p_it =  hem_polygons.begin();
//		p_it != hem_polygons.end();
//		p_it++)
//	{
//		HalfEdgePolygon* polygon = *p_it;
//		for(it  = polygon->edges.begin();
//			it != polygon->edges.end();
//			it++)
//		{
//			HalfEdge* e = it->second;
//			out << "BEGIN" << endl;
//			out << e->start->position.x << " " << e->start->position.y << " " << e->start->position.z << endl;
//			out << e->end->position.x   << " " << e->end->position.y   << " " << e->end->position.z   << endl;
//			out << "END" << endl;
//		}
//	}
//
//
//
//}

//void HalfEdgeMesh::write_face_normals(string filename){
//
//	ofstream out(filename.c_str());
//
//	HalfEdgeFace* face;
//
//	Normal n;
//	Vertex v;
//
//	int c = 0;
//
//	vector<HalfEdgeFace*>::iterator face_iterator;
//	for(face_iterator = he_faces.begin();
//		face_iterator != he_faces.end();
//		face_iterator++)
//	{
//		if(c % 10000 == 0){
//			cout << "Write Face Normals: " << c << " / " << he_faces.size() << endl;
//		}
//		face = *face_iterator;
//		//n = face->getFaceNormal();
//		n = face->getInterpolatedNormal();
//		v = face->getCentroid();
//
//		out << v.x << " " << v.y << " " << v.z << " "
//		    << n.x << " " << n.y << " " << n.z << endl;
//
//		c++;
//	}
//
//}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::printStats()
{
	if(this->m_finalized)
	{
		cout << "##### HalfEdge Mesh (S): " << this->m_nVertices << " Vertices / "
		                                    << this->m_nFaces    << " Faces.   " << endl;
	} else {
		cout << "##### HalfEdge Mesh (D): " << this->m_nVertices << " Vertices / "
		                                    << this->m_nFaces / 3 << " Faces." << endl;
	}
}

} // namespace lssr
