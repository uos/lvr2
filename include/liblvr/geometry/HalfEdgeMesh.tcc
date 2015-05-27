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
 * HalfEdgeMesh.tcc
 *
 *  @date 13.11.2008
 *  @author Florian Otte (fotte@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */


namespace lvr
{
	
template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::HalfEdgeMesh( )
{
    m_globalIndex = 0;
    m_regionClassifier = ClassifierFactory<VertexT, NormalT>::get("Default", this);
    m_classifierType = "Default";
    m_pointCloudManager = NULL;
    m_depth = 100;
}

template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::HalfEdgeMesh( 
        typename PointsetSurface<VertexT>::Ptr pm )
{
    m_globalIndex = 0;
    m_regionClassifier = ClassifierFactory<VertexT, NormalT>::get("Default", this);
    m_classifierType = "Default";
    m_pointCloudManager = pm;
    m_depth = 100;
}

template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::HalfEdgeMesh(
        MeshBufferPtr mesh)
{
    size_t num_verts, num_faces;
    floatArr vertices = mesh->getVertexArray(num_verts);

    // Add all vertices
    for(size_t i = 0; i < num_verts; i++)
    {
        addVertex(VertexT(vertices[3 * i], vertices[3 * i + 1], vertices[3 * i + 2]));
    }

    // Add all faces
    uintArr faces = mesh->getFaceArray(num_faces);
    for(size_t i = 0; i < num_faces; i++)
    {
        addTriangle(faces[3 * i], faces[3 * i + 1], faces[3 * i + 2]);
    }

    // Initial remaining stuff
    m_globalIndex = this->meshSize();
    m_regionClassifier = ClassifierFactory<VertexT, NormalT>::get("Default", this);
    m_classifierType = "Default";
    m_depth = 100;
}

template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>::~HalfEdgeMesh()
{

    this->m_meshBuffer.reset();
    if(this->m_pointCloudManager != NULL)
		this->m_pointCloudManager.reset();

    typename set<EdgePtr>::iterator e_it = m_garbageEdges.begin();
    for(; e_it != m_garbageEdges.end(); e_it++)
    {
        EdgePtr e = *e_it;
        delete e;
    }
    m_garbageEdges.clear();


    if(this->m_regionClassifier != 0)
    {
        delete this->m_regionClassifier;
        this->m_regionClassifier = 0;
    }

    for (int i = 0 ; i < m_vertices.size() ; i++)
    {
        delete m_vertices[i];
    }
    this->m_vertices.clear();

    typename set<Region<VertexT, NormalT>*>::iterator r_it;
    for(r_it = m_garbageRegions.begin(); r_it != m_garbageRegions.end(); r_it++)
    {
        RegionPtr r = *r_it;
        delete r;
    }
    m_garbageRegions.clear();

    typename set<FacePtr>::iterator f_it = m_garbageFaces.begin();
    for(; f_it != m_garbageFaces.end(); f_it++)
    {
        FacePtr f = *f_it;
        delete f;
    }
    m_garbageFaces.clear();

}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::setClassifier(string name)
{
	// Delete old classifier if present
	if(m_regionClassifier) delete m_regionClassifier;

	// Create new one
	m_regionClassifier = ClassifierFactory<VertexT, NormalT>::get(name, this);
	
	// update name
	m_classifierType = name;

	// Check if successful
	if(!m_regionClassifier)
	{
		cout << timestamp << "Warning: Unable to create classifier type '"
			 << name << "'. Using default." << endl;

		m_regionClassifier = ClassifierFactory<VertexT, NormalT>::get("Default", this);
		m_classifierType = "Default";
	}
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addVertex(VertexT v)
{
    // Create new HalfEdgeVertex and increase vertex counter
    m_vertices.push_back(new HVertex(v));
    m_globalIndex++;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addGiftVertex(VertexT v, bool& bullseye)
{
    // Create new HalfEdgeVertex and increase vertex counter
	for(int i = 300000; i < m_vertices.size(); i++){
		auto vertex  = m_vertices[i];
		if(vertex->m_position[0] == v[0] && vertex->m_position[1] == v[1]  && vertex->m_position[2] == v[2]){
			cout << "ICH BIN DOPPELT" << endl;
			cout << "i: "  << i <<  endl;
			cout <<"vorhanden " << vertex->m_position[0] << " " << vertex->m_position[1] << " " << vertex->m_position[2] << endl;
			cout <<"versuch " << v << endl;
			bullseye = true;
		}
	}
    m_vertices.push_back(new HVertex(v));
    m_globalIndex++;
}



template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteVertex(VertexPtr v)
{
    // Delete HalfEdgeVertex and decrease vertex counter
  /*  m_vertices.erase(find(m_vertices.begin(), m_vertices.end(), v));
    m_globalIndex--;*/
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addNormal(NormalT n)
{
    // Is a vertex exists at globalIndex, save normal
    assert(m_globalIndex == m_vertices.size());
    m_vertices[m_globalIndex - 1]->m_normal = n;
}

template<typename VertexT, typename NormalT>
HalfEdge<HalfEdgeVertex<VertexT, NormalT>, HalfEdgeFace<VertexT, NormalT> >* HalfEdgeMesh<VertexT, NormalT>::halfEdgeToVertex(VertexPtr v, VertexPtr next)
{
    EdgePtr edge = 0;
    EdgePtr cur;

    typename EdgeVector::iterator it;

    for(it = v->in.begin(); it != v->in.end(); it++)
    {
        // Check all incoming edges, if start and end vertex
        // are the same. If they are, save this edge.
        cur = *it;
        if(cur->end() == v && cur->start() == next)
        {
            edge = cur;
        }
    }

    return edge;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c, FacePtr &f)
{
    // Create a new face
    FacePtr face = new HFace;
    m_faces.push_back(face);
    m_garbageFaces.insert(face);

    // Create a list of HalfEdges that will be connected
    // with this here. Here we need only to alloc space for
    // three pointers, allocation and linking will be done
    // later.
    EdgePtr edges[3];

    // Traverse face triangles
    for(int k = 0; k < 3; k++)
    {
        // Pointer to start and end vertex of an edge
        VertexPtr current;
        VertexPtr next;

        // Map k values to parameters
        switch(k)
        {
        case 0:
            current = m_vertices[a];
            next    = m_vertices[b];
            break;
        case 1:
            current = m_vertices[b];
            next    = m_vertices[c];
            break;
        case 2:
            current = m_vertices[c];
            next    = m_vertices[a];
            break;
        }

        // Try to find an pair edges of an existing face,
        // that points to the current vertex. If such an
        // edge exists, the pair-edge of this edge is the
        // one we need. Update link. If no edge is found,
        // create a new one.
        EdgePtr edgeToVertex = halfEdgeToVertex(current, next);

        // If a fitting edge was found, save the pair edge
        // and let it point the the new face

        if(edgeToVertex)
        {
            try
            {
                edges[k] = edgeToVertex->pair();
            }
            catch (HalfEdgeAccessException &e)
            {
                cout << "HalfEdgeMesg::addTriangle: " << e.what() << endl;
                EdgePtr edge = new HEdge;
                m_garbageEdges.insert(edge);
                edge->setStart(edgeToVertex->end());
                edge->setEnd(edgeToVertex->start());
                edges[k] = edge;
            }

            edges[k]->setFace(face);
        }
        else
        {
            // Create new edge and pair
            EdgePtr edge = new HEdge;
            m_garbageEdges.insert(edge);
            edge->setFace(face);
            edge->setStart(current);
            edge->setEnd(next);

            EdgePtr pair = new HEdge;
            m_garbageEdges.insert(pair);
            pair->setStart(next);
            pair->setEnd(current);
            pair->setFace(0);

            // Link Half edges
            edge->setPair(pair);
            pair->setPair(edge);

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
        edges[k]->setNext(edges[(k + 1) % 3]);
    }

    //m_faces.push_back(face);
    face->m_edge = edges[0];
    face->calc_normal();
    face->m_face_index = m_faces.size();
    f = face;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::setFusionVertex(uint v)
{
	m_vertices[v]->m_fused = true;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::addTriangle(uint a, uint b, uint c)
{
    FacePtr face;
    addTriangle(a, b, c, face);
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::cleanContours(int iterations)
{
	for(int a = 0; a < iterations; a++)
	{
		FaceVector toDelete;
		toDelete.clear();
		for(int i = 0; i < m_faces.size(); i++)
		{


			// Get current face
			FacePtr f = m_faces[i];

			// Count border edges
			int bf = 0;

			EdgePtr e1 = f->m_edge;
			EdgePtr e2 = e1->next();
			EdgePtr e3 = e2->next();

			EdgePtr s = f->m_edge;
			EdgePtr e = s;
			do
			{
				if(e->pair())
				{
					EdgePtr p = e->pair();
					if(!(p->face())) bf++;
				}
				e = e->next();
			}
			while(s != e);

			// Mark face if is an artifact and store in removal list
			if(bf >= 2)
			{
				toDelete.push_back(f);
			}

			if(bf == 1)
			{
				if(f->getArea() < 0.0001) toDelete.push_back(f);
			}

		}

		// Delete all artifact faces
		for(int i = 0; i < (int)toDelete.size(); i++)
		{
			deleteFace(toDelete[i]);
		}
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteFace(FacePtr f, bool erase)
{
    f->m_invalid = true;
    m_regions[f->m_region]->deleteInvalidFaces();

    //save references to edges and vertices
    HEdge* startEdge = (*f)[0];
    HEdge* nextEdge  = (*f)[1];
    HEdge* lastEdge  = (*f)[2];
    HVertex* p1 = (*f)(0);
    HVertex* p2 = (*f)(1);
    HVertex* p3 = (*f)(2);

    startEdge->setFace(0);
    startEdge->setFace(0);
    nextEdge->setFace(0);
    nextEdge->setFace(0);
    lastEdge->setFace(0);
    lastEdge->setFace(0);

    try
    {
       startEdge->pair()->face();
    }
    catch(...)
    {
        deleteEdge(startEdge);
    }

    try
    {
        nextEdge->pair()->face();
    }
    catch(...)
    {
        deleteEdge(nextEdge);
    }

    try
    {
        lastEdge->pair()->face();
    }
    catch(...)
    {
        deleteEdge(lastEdge);
    }

    if(erase)
    {
        typename vector<FacePtr>::iterator it = find(m_faces.begin(), m_faces.end(), f);
        if(it != m_faces.end())
        {
            m_faces.erase(it);
        }
    }

}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::checkFaceIntegreties()
{
    int c = 0;
    HEdge* edge;
    typename vector<FacePtr>::iterator it;
    for(it = m_faces.begin(); it != m_faces.end(); it++)
    {
        bool face_ok = true;
        for(int i = 0; i < 3; i++)
        {
            try
            {
                edge = (*(*it))[i];
            }
            catch(HalfEdgeAccessException)
            {
                face_ok = false;
            }
        }
        if(!face_ok)
        {
            c++;
        }
    }
    cout << timestamp << "Checked face integreties. Found " << c << " degenerated faces from " << m_faces.size() << " checked." << endl;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteEdge(EdgePtr edge, bool deletePair)
{
    typename vector<EdgePtr>::iterator it;
   /* try
    {
        // Do not delete edges that still have a face!
        edge->face();
        return;
    } catch(...) {}*/



    try
    {
        //delete references from start point to outgoing edge
        it = find(edge->start()->out.begin(), edge->start()->out.end(), edge);
        if(it != edge->start()->out.end())
        {
            edge->start()->out.erase(it);
        }
    }
    catch (HalfEdgeAccessException &e)
    {
        cout << "HalfEdgeMesh::deleteEdge(): " << e.what() << endl;
    }

    try
    {
        it = find(edge->end()->in.begin(), edge->end()->in.end(), edge);
        //delete references from end point to incoming edge
        if(it !=  edge->end()->in.end())
        {
            edge->end()->in.erase(it);
        }
    }
    catch (HalfEdgeAccessException &e)
    {
        cout << "HalfEdgeMesh::deleteEdge(): " << e.what() << endl;
    }


    if(deletePair)
    {
        try
        {
            //delete references from start point to outgoing edge
            it = find(edge->pair()->start()->out.begin(), edge->pair()->start()->out.end(), edge->pair());
            if(it != edge->pair()->start()->out.end())
            {
                edge->pair()->start()->out.erase(it);
            }
        }
        catch (HalfEdgeAccessException )
        {
            //cout << "HalfEdgeMesh::deleteEdge(): " << e.what() << endl;
        }

        try
        {
            it = find(edge->pair()->end()->in.begin(), edge->pair()->end()->in.end(), edge->pair());
            if(it != edge->pair()->end()->in.end())
            {
                edge->pair()->end()->in.erase(it);
            }
            edge->pair()->setPair(0);
        }
        catch (HalfEdgeAccessException)
        {
            //cout << "HalfEdgeMesh::deleteEdge(): " << e.what() << endl;
        }
        edge->setPair(0);
    }
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::collapseEdge(EdgePtr edge)
{

    // Save start and end vertex
    VertexPtr p1 = edge->start();
    VertexPtr p2 = edge->end();

    // Don't collapse zero edges (need to fix them!!!)
    if(p1 == p2) return;

    // Move p1 to the center between p1 and p2 (recycle p1)
    p1->m_position = (p1->m_position + p2->m_position) * 0.5;

    // Reorganize the pointer structure between the edges.
    // If a face will be deleted after the edge is collapsed the pair pointers
    // have to be reseted.
    try
    {
        if (edge->face())
        {
            // reorganize pair pointers
            edge->next()->next()->pair()->setPair(edge->next()->pair());
            edge->next()->pair()->setPair(edge->next()->next()->pair());

            EdgePtr e1 = edge->next()->next();
            EdgePtr e2 = edge->next();

            //delete old edges
            deleteEdge(e1, false);
            deleteEdge(e2, false);
        }
    }
    catch(HalfEdgeAccessException)
    {
    }

    try
    {
        if (edge->pair()->face())
        {
            // reorganize pair pointers
            edge->pair()->next()->next()->pair()->setPair(edge->pair()->next()->pair());
            edge->pair()->next()->pair()->setPair(edge->pair()->next()->next()->pair());
            //delete old edges

            EdgePtr e1 = edge->pair()->next()->next();
            EdgePtr e2 = edge->pair()->next();

            deleteEdge(e1, false);
            deleteEdge(e2, false);
        }
    }
    catch(HalfEdgeAccessException )
    {

    }

    // Now really delete faces
    try
    {
        if(edge->pair()->face())
        {
            deleteFace(edge->pair()->face());
            edge->setPair(0);
        }
    }
    catch(HalfEdgeAccessException )
    {
    }

    try
    {
        if(edge->face())
        {
            deleteFace(edge->face());
            edge->setFace(0);
        }
    }
    catch(HalfEdgeAccessException )
    {
    }

    //Delete collapsed edge and its' pair
    deleteEdge(edge);

    //Update incoming and outgoing edges of p1 (the start point of the collapsed edge)
    typename vector<EdgePtr>::iterator it;
    it = p2->out.begin();

    while(it != p2->out.end())
    {
        (*it)->setStart(p1);
        p1->out.push_back(*it);
        it++;
    }

    //Update incoming and outgoing edges of p2 (the end point of the collapsed edge)
    it = p2->in.begin();
    while(it != p2->in.end())
    {
        (*it)->setEnd(p1);
        p1->in.push_back(*it);
        it++;
    }

    //Delete p2
    //deleteVertex(p2);
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::flipEdge(FacePtr f1, FacePtr f2)
{
    EdgePtr commonEdge;
    EdgePtr current = f1->m_edge;

    //search the common edge between the two faces
    for(int k = 0; k < 3; k++)
    {
        if (current->pair()->face() == f2)
        {
            commonEdge = current;
        }
        current = current->next();
    }

    //return if f1 and f2 are not adjacent in the grid
    if(!commonEdge)
    {
        return;
    }

    //flip the common edge
    this->flipEdge(commonEdge);
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::flipEdge(uint v1, uint v2)
{
    EdgePtr edge = halfEdgeToVertex(m_vertices[v1], m_vertices[v2]);
    if(edge)
    {
        flipEdge(edge);
    }
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::flipEdge(EdgePtr edge)
{
    // This can only be done if there are two faces on both sides of the edge
    if (edge->pair()->face() != 0 && edge->face() != 0)
    {
        //The old egde will be deleted while a new edge is created

        //save the start and end vertex of the new edge
        VertexPtr newEdgeStart = edge->next()->end();
        VertexPtr newEdgeEnd   = edge->pair()->next()->end();

        //update the next pointers of the remaining edges
        //Those next pointers pointed to the edge that will be deleted before.
        edge->next()->next()->setNext(edge->pair()->next());
        edge->pair()->next()->next()->setNext(edge->next());

        //create the new edge
        EdgePtr newEdge = new HEdge;
        m_garbageEdges.insert(newEdge);

        //set its' start and end vertex
        newEdge->setStart(newEdgeStart);
        newEdge->setEnd(newEdgeEnd);
        newEdge->setPair(0);

        //set the new edge's next pointer to the appropriate edge
        newEdge->setNext(edge->pair()->next()->next());

        //use one of the old faces for the new edge
        newEdge->setFace(edge->pair()->next()->next()->face());

        //update incoming and outgoing edges of the start and end vertex
        newEdge->start()->out.push_back(newEdge);
        newEdge->end()->in.push_back(newEdge);

        //create the new pair
        EdgePtr newpair = new HEdge;
        m_garbageEdges.insert(newpair);

        //set its' start and end vertex (complementary to new edge)
        newpair->setStart(newEdgeEnd);
        newpair->setEnd(newEdgeStart);

        //set the pair pointer to the new edge
        newpair->setPair(newEdge);

        //set the new pair's next pointer to the appropriate edge
        newpair->setNext(edge->next()->next());

        //use the other one of the old faces for the new pair
        newpair->setFace(edge->next()->next()->face());

        //update incoming and outgoing edges of the start and end vertex
        newpair->start()->out.push_back(newpair);
        newpair->end()->in.push_back(newpair);

        //set the new edge's pair pointer to the new pair
        newEdge->setPair(newpair);

        //update face()->edge pointers of the recycled faces
        newEdge->face()->m_edge = newEdge;
        newpair->face()->m_edge = newpair;

        //update next pointers of the edges pointing to new edge and new pair
        edge->next()->setNext(newEdge);
        edge->pair()->next()->setNext(newpair);

        //update edge->face() pointers
        newEdge->next()->setFace(newEdge->face());
        newEdge->next()->next()->setFace(newEdge->face());
        newpair->next()->setFace(newpair->face());
        newpair->next()->next()->setFace(newpair->face());

        //recalculate face normals
        newEdge->face()->calc_normal();
        newpair->face()->calc_normal();

        //delete the old edge
        deleteEdge(edge);
    }
}

template<typename VertexT, typename NormalT>
int HalfEdgeMesh<VertexT, NormalT>::stackSafeRegionGrowing(FacePtr start_face, NormalT &normal, float &angle, RegionPtr region)
{
    // stores the faces where we need to continue
    vector<FacePtr> leafs;
    int regionSize = 0;
    leafs.push_back(start_face);
    do
    {
        if(leafs.front()->m_used == false)
        {
            regionSize += regionGrowing(leafs.front(), normal, angle, region, leafs, m_depth);
        }
        leafs.erase(leafs.begin());
    }
    while(!leafs.empty());

    return regionSize;
}

template<typename VertexT, typename NormalT>
int HalfEdgeMesh<VertexT, NormalT>::regionGrowing(FacePtr start_face, NormalT &normal, float &angle, RegionPtr region, vector<FacePtr> &leafs, unsigned int depth)
{
    //Mark face as used
    start_face->m_used = true;

    //Add face to region
    region->addFace(start_face);

    int neighbor_cnt = 0;

    //Get the unmarked neighbor faces and start the recursion
    try
    {
        for(int k = 0; k < 3; k++)
        {
            if((*start_face)[k]->pair()->face() != 0 && (*start_face)[k]->pair()->face()->m_used == false
                    && fabs((*start_face)[k]->pair()->face()->getFaceNormal() * normal) > angle )
            {
                if(depth == 0)
                {
                    // if the maximum recursion depth is reached save the child faces to restart the recursion from
                    leafs.push_back((*start_face)[k]->pair()->face());
                }
                else
                {
                    // start the recursion
                    ++neighbor_cnt += regionGrowing((*start_face)[k]->pair()->face(), normal, angle, region, leafs, depth - 1);
                }
            }
        }
    }
    catch (HalfEdgeAccessException )
    {
        // Just ignore access to invalid elements
        //cout << "HalfEdgeMesh::regionGrowing(): " << e.what() << endl;
    }


    return neighbor_cnt;
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::clusterRegions(float angle, int minRegionSize)
{
	// Reset all used variables
	for(size_t i = 0; i < m_faces.size(); i++)
	{
		m_faces[i]->m_used = false;
	}
	m_regions.clear();

	int region_number = 0;
	int region_size = 0;
	// Find all regions by regionGrowing with normal criteria
	for(size_t i = 0; i < m_faces.size(); i++)
	{
		if(m_faces[i]->m_used == false)
		{
			NormalT n = m_faces[i]->getFaceNormal();

			RegionPtr region = new Region<VertexT, NormalT>(m_regions.size());
			m_garbageRegions.insert(region);
			stackSafeRegionGrowing(m_faces[i], n, angle, region);

			// Save pointer to the region
			m_regions.push_back(region);
		}
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlanes(
        int iterations,
        float angle,
        int min_region_size,
        int small_region_size,
        bool remove_flickering)
{
    cout << timestamp << "Starting plane optimization with threshold " << angle << endl;
    cout << timestamp << "Number of faces before optimization: " << m_faces.size() << endl;

    // Magic numbers
    int default_region_threshold = (int) 10 * log(m_faces.size());

    int region_size   = 0;
    int region_number = 0;
    m_regions.clear();

    for(int j = 0; j < iterations; j++)
    {
        cout << timestamp << "Optimizing planes. Iteration " <<  j + 1 << " / "  << iterations << endl;

        // Reset all used variables
        for(size_t i = 0; i < m_faces.size(); i++)
        {
            m_faces[i]->m_used = false;
        }

        // Find all regions by regionGrowing with normal criteria
        for(size_t i = 0; i < m_faces.size(); i++)
        {
            if(m_faces[i]->m_used == false)
            {
                NormalT n = m_faces[i]->getFaceNormal();

                Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(region_number);
                m_garbageRegions.insert(region);
                region_size = stackSafeRegionGrowing(m_faces[i], n, angle, region) + 1;

                // Fit big regions into the regression plane
                if(region_size > max(min_region_size, default_region_threshold))
                {
                    region->regressionPlane();
                }

                if(j == iterations - 1)
                {
                    // Save too small regions with size smaller than small_region_size
                    if (region_size < small_region_size)
                    {
                        region->m_toDelete = true;
                    }

                    // Save pointer to the region
                    m_regions.push_back(region);
                    region_number++;

                }
            }
        }
    }

    // Delete too small regions
    cout << timestamp << "Starting to delete small regions" << endl;
    if(small_region_size)
    {
        cout << timestamp << "Deleting small regions" << endl;
        deleteRegions();
    }

    //Delete flickering faces
    if(remove_flickering)
    {
   /*     vector<FacePtr> flickerer;
        for(size_t i = 0; i < m_faces.size(); i++)
        {
            if(m_faces[i]->m_region && m_faces[i]->m_region->detectFlicker(m_faces[i]))
            {
                flickerer.push_back(m_faces[i]);
            }
        }

        while(!flickerer.empty())
        {
            deleteFace(flickerer.back());
            flickerer.pop_back();
        }*/
    }

}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizeIterativePlanes(
        int iterations,
        float angle,
        int min_region_size,
        size_t oldSize)
{
    cout << timestamp << "Starting plane optimization with threshold " << angle << endl;
    cout << timestamp << "Number of faces before optimization: " << m_faces.size() << endl;

    // Magic numbers
    int default_region_threshold = (int) 10 * log(m_faces.size());

    int region_size   = 0;
    int region_number = 0;
    m_regions.clear();
	
	
    for(int j = 0; j < iterations; j++)
    {
        cout << timestamp << "Optimizing planes. Iteration " <<  j + 1 << " / "  << iterations << endl;

        // Reset all used variables
        // TODO Track Vertex Pose in new mesh
        for(size_t i = oldSize; i < m_faces.size(); i++)
        {
			FacePtr face = m_faces[i];
			if((*face)(0)->m_fused || (*face)(1)->m_fused || (*face)(2)->m_fused)
			{
				face->m_used = true;
			}
			else
				face->m_used = false;
        }

        // Find all regions by regionGrowing with normal criteria
        for(size_t i = oldSize; i < m_faces.size(); i++)
        {
            if(m_faces[i]->m_used == false)
            {
                NormalT n = m_faces[i]->getFaceNormal();

                Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(region_number);
                m_garbageRegions.insert(region);
                region_size = stackSafeRegionGrowing(m_faces[i], n, angle, region) + 1;

                // Fit big regions into the regression plane
                if(region_size > max(min_region_size, default_region_threshold))
                {
                    region->regressionPlane();
                }

                if(j == iterations - 1)
                {

                    // Save pointer to the region
                    m_regions.push_back(region);
                    region_number++;

                }
            }
        }
    }

}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::deleteRegions()
{
    for(int i = 0; i < m_faces.size(); i++)
    {
        if(m_faces[i]->m_region >= 0 && m_regions[m_faces[i]->m_region]->m_toDelete)
        {
            deleteFace(m_faces[i], false);
            m_faces[i] = 0;
        }
    }

    typename vector<FacePtr>::iterator f_iter = m_faces.begin();
    while (f_iter != m_faces.end())
    {
        if (!(*f_iter))
        {
            f_iter = m_faces.erase(f_iter);
        }
        else
        {
            ++f_iter;
        }
    }

    typename vector<RegionPtr>::iterator r_iter = m_regions.begin();
    while (r_iter != m_regions.end())
    {
        if (!(*r_iter))
        {
            r_iter = m_regions.erase(r_iter);
        }
        else
        {
            ++r_iter;
        }
    }
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::removeDanglingArtifacts(int threshold)
{
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        if(m_faces[i]->m_used == false)
        {
            RegionPtr region = new Region<VertexT, NormalT>(0);
            m_garbageRegions.insert(region);
            NormalT n = m_faces[i]->getFaceNormal();
            float angle = -1;
            int region_size = stackSafeRegionGrowing(m_faces[i], n, angle, region) + 1;
            if(region_size <= threshold)
            {
                region->m_toDelete = true;
            }
        }
    }

    //delete dangling artifacts
    cout << timestamp << "Removing dangling artifacts" << endl;
    deleteRegions();

    //reset all used variables
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        m_faces[i]->m_used = false;
    }
}

template<typename VertexT, typename NormalT>
bool HalfEdgeMesh<VertexT, NormalT>::safeCollapseEdge(EdgePtr edge)
{

    //try to reject all huetchen
    //A huetchen geometry must not be present at the edge or it's pair
    try
    {
        if(edge->face() != 0 && edge->next()->pair()->face() != 0 && edge->next()->next()->pair()->face() != 0)
        {
            if(edge->next()->pair()->next()->next() == edge->next()->next()->pair()->next()->pair())
            {
                return false;
            }
        }
    }
    catch(HalfEdgeAccessException )
    {

    }

    try
    {
        if(edge->pair()->face() && edge->pair()->next()->pair()->face() && edge->pair()->next()->next()->pair()->face())
        {
            if(edge->pair()->next()->pair()->next()->next() == edge->pair()->next()->next()->pair()->next()->pair())
            {
                return false;
            }
        }
    }
    catch(HalfEdgeAccessException )
    {

    }

    //Check for redundant edges i.e. more than one edge between the start and the
    //end point of the edge which is tried to collapse
    int edgeCnt = 0;
    for (size_t i = 0; i < edge->start()->out.size(); i++)
    {
        try
        {
            if (edge->start()->out[i]->end() == edge->end())
            {
                edgeCnt++;
            }
        }
        catch(HalfEdgeAccessException )
        {

        }
    }
    if(edgeCnt != 1)
    {
        return false;
    }

    //Avoid creation of edges without faces
    //Collapsing the edge in this constellation leads to the creation of edges without any face
    try
    {
        if( ( edge->face() != 0 && edge->next()->pair()->face() == 0 && edge->next()->next()->pair()->face() == 0 )
                || ( edge->pair()->face() != 0 && edge->pair()->next()->pair()->face() == 0 && edge->pair()->next()->next()->pair()->face() == 0 ) )
        {
            return false;
        }
    }
    catch(HalfEdgeAccessException)
    {

    }

    //Check for triangle hole
    //We do not want to close triangle holes as this can be achieved by adding a new face
    for(size_t o1 = 0; o1 < edge->end()->out.size(); o1++)
    {
        try
        {
            for(size_t o2 = 0; o2 < edge->end()->out[o1]->end()->out.size(); o2++)
            {
                if(edge->end()->out[o1]->face() == 0 && edge->end()->out[o1]->end()->out[o2]->face() == 0 && edge->end()->out[o1]->end()->out[o2]->end() == edge->start())
                {
                    return false;
                }
            }
        }
        catch(HalfEdgeAccessException )
        {

        }
    }

    //Check for flickering
    //Move edge->start() to its' theoretical position and check for flickering
    VertexT origin = edge->start()->m_position;
    edge->start()->m_position = (edge->start()->m_position + edge->end()->m_position) * 0.5;
    for(size_t o = 0; o < edge->start()->out.size(); o++)
    {
        try
        {
            if(edge->start()->out[o]->pair()->face() != edge->pair()->face())
            {
                // Safety First!
                if(edge->start()->out[o]->pair()->face()->m_region >= 0)
                {
                    if (edge->start()->out[o]->pair()->face() != 0 && m_regions[edge->start()->out[o]->pair()->face()->m_region]->detectFlicker(edge->start()->out[o]->pair()->face()))
                    {
                        edge->start()->m_position = origin;
                        return false;
                    }
                }
            }
        }
        catch(HalfEdgeAccessException)
        {

        }
    }

    //Move edge->end() to its' theoretical position and check for flickering
    origin = edge->end()->m_position;
    edge->end()->m_position = (edge->start()->m_position + edge->end()->m_position) * 0.5;
    for(size_t o = 0; o < edge->end()->out.size(); o++)
    {
        try
        {
            if(edge->end()->out[o]->pair()->face() != edge->pair()->face())
            {
                if(edge->end()->out[o]->pair()->face()->m_region >= 0)
                {
                    if (edge->end()->out[o]->pair()->face() != 0 && m_regions[edge->end()->out[o]->pair()->face()->m_region]->detectFlicker(edge->end()->out[o]->pair()->face()))
                    {
                        edge->end()->m_position = origin;
                        return false;
                    }
                }
            }
        }
        catch(HalfEdgeAccessException)
        {

        }
    }
    //finally collapse the edge
    collapseEdge(edge);

    return true;
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::fillHoles(size_t max_size)
{
    // Do'nt fill holes if contour size is zero
    if(max_size == 0) return;

    //holds all holes to close
    vector<vector<EdgePtr> > holes;

    //walk through all edges and start hole finding
    //when pair has no face
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        for(int k = 0; k < 3; k++)
        {
            try
            {
                EdgePtr current = (*m_faces[i])[k]->pair();
                if(current->used == false && !current->hasFace())
                {
                    //needed for contour tracking
                    vector<EdgePtr> contour;
                    EdgePtr next = 0;

                    //while the contour is not closed
                    while(current)
                    {
                        next = 0;
                        contour.push_back(current);
                        //to ensure that there is no way back to the same vertex
                        for (size_t e = 0; e < current->start()->out.size(); e++)
                        {
                            try
                            {
                                if (current->start()->out[e]->end() == current->end())
                                {
                                    current->start()->out[e]->used          = true;
                                    current->start()->out[e]->pair()->used  = true;
                                }
                            }
                            catch (HalfEdgeAccessException)
                            {
                                // cout << "HalfEdgeMesg::fillHoles: " << e.what() << endl;
                            }
                        }
                        current->used = true;

                        typename vector<EdgePtr>::iterator it = current->end()->out.begin();
                        while(it != current->end()->out.end())
                        {
                            //found a new possible edge to trace
                            if ((*it)->used == false && !(*it)->hasFace())
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
            catch(HalfEdgeAccessException)
            {
                // cout << "HalfEdgeMesg::fillHoles: " << e.what() << endl;
            }
        } // for
    }

    //collapse the holes
    string msg = timestamp.getElapsedTime() + "Filling holes ";
    ProgressBar progress(holes.size(), msg);

    for(size_t h = 0; h < holes.size(); h++)
    {
        vector<EdgePtr> current_hole = holes[h];

        //collapse as much edges as possible
        bool collapsedSomething = true;
        while(collapsedSomething)
        {
            collapsedSomething = false;
            for(size_t e = 0; e < current_hole.size() && ! collapsedSomething; e++)
            {
                if(safeCollapseEdge(current_hole[e]))
                {
                    collapsedSomething = true;
                    current_hole.erase(current_hole.begin() + e);
                }
            }
        }

        //add new faces
        while(current_hole.size() > 0)
        {
            bool stop = false;
            for(size_t i = 0; i < current_hole.size() && !stop; i++)
            {
                for(size_t j = 0; j < current_hole.size() && !stop; j++)
                {
                    if(current_hole.back()->end() == current_hole[i]->start())
                    {
                        if(current_hole[i]->end() == current_hole[j]->start())
                        {
                            if(current_hole[j]->end() == current_hole.back()->start())
                            {
                                FacePtr f = new HFace;
                                m_garbageFaces.insert(f);
                                f->m_edge = current_hole.back();
                                current_hole.back()->setNext(current_hole[i]);
                                current_hole[i]->setNext(current_hole[j]);
                                current_hole[j]->setNext(current_hole.back());
                                for(int e = 0; e < 3; e++)
                                {
                                    (*f)[e]->setFace(f);
                                    typename vector<EdgePtr>::iterator ch_it = find(current_hole.begin(), current_hole.end(), (*f)[e]);
                                    if(ch_it != current_hole.end())
                                    {
                                        current_hole.erase(ch_it);
                                    }
                                }
                                try
                                {
                                    if((*f)[0]->pair()->face()->m_region >= 0)
                                    {
                                        m_regions[(*f)[0]->pair()->face()->m_region]->addFace(f);
                                    }
                                }
                                catch(HalfEdgeAccessException)
                                {

                                }
                                m_faces.push_back(f);
                                stop = true;
                            }
                        }
                    }
                }
            }
            if(!stop)
            {
                current_hole.pop_back();
            }
        }
        ++progress;
    }
    cout << endl;
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::dragOntoIntersection(RegionPtr plane, RegionPtr neighbor_region, VertexT& x, VertexT& direction)
{
    for (size_t i = 0; i < plane->m_faces.size(); i++)
    {
        for(int k = 0; k <= 2; k++)
        {
            try
            {
                if((*(plane->m_faces[i]))[k]->pair()->face()->m_region >= 0)
                {
                    if((*(plane->m_faces[i]))[k]->pair()->face() != 0 && m_regions[(*(plane->m_faces[i]))[k]->pair()->face()->m_region]->m_regionNumber == neighbor_region->m_regionNumber)
                    {
                        (*(plane->m_faces[i]))[k]->start()->m_position = x + direction * (((((*(plane->m_faces[i]))[k]->start()->m_position) - x) * direction) / (direction.length() * direction.length()));
                        (*(plane->m_faces[i]))[k]->end()->m_position   = x + direction * (((((*(plane->m_faces[i]))[k]->end()->m_position  ) - x) * direction) / (direction.length() * direction.length()));
                    }
                }
            }
            catch(HalfEdgeAccessException)
            {
                // Just ignore access to invalid elements
                // cout << "HalfEdgeMesh::dragOntoIntersection(): " << e.what() << endl;
            }
        }
    }
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::optimizePlaneIntersections()
{
    string msg = timestamp.getElapsedTime() + "Optimizing plane intersections ";
    ProgressBar progress(m_regions.size(), msg);

    for (size_t i = 0; i < m_regions.size(); i++)
    {
        if (m_regions[i]->m_inPlane)
        {
            for(size_t j = i + 1; j < m_regions.size(); j++)
            {
                if(m_regions[j]->m_inPlane)
                {
                    //calculate intersection between plane i and j

                    NormalT n_i = m_regions[i]->m_normal;
                    NormalT n_j = m_regions[j]->m_normal;

                    //don't improve almost parallel regions - they won't cross in a reasonable distance
                    if (fabs(n_i * n_j) < 0.9)
                    {

                        float d_i = n_i * m_regions[i]->m_stuetzvektor;
                        float d_j = n_j * m_regions[j]->m_stuetzvektor;

                        VertexT direction = n_i.cross(n_j);

                        float denom = direction * direction;
                        VertexT x = ((n_j * d_i - n_i * d_j).cross(direction)) * (1 / denom);

                        //drag all points at the border between planes i and j onto the intersection
                        dragOntoIntersection(m_regions[i], m_regions[j], x, direction);
                        dragOntoIntersection(m_regions[j], m_regions[i], x, direction);
                    }
                }
            }
        }
        ++progress;
    }
    cout << endl;
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::restorePlanes(int min_region_size)
{
    for(size_t r = 0; r < m_regions.size(); r++)
    {
        //drag points into the regression plane
        if( m_regions[r]->m_inPlane)
        {
            for(size_t i = 0; i < m_regions[r]->m_faces.size(); i++)
            {
                for(int p = 0; p < 3; p++)
                {
                    try
                    {
                        float v = ((m_regions[r]->m_stuetzvektor - (*(m_regions[r]->m_faces[i]))(p)->m_position) * m_regions[r]->m_normal) / (m_regions[r]->m_normal * m_regions[r]->m_normal);
                        if(v != 0)
                        {
                            (*(m_regions[r]->m_faces[i]))(p)->m_position = (*(m_regions[r]->m_faces[i]))(p)->m_position + (VertexT)m_regions[r]->m_normal * v;
                        }
                    }
                    catch (HalfEdgeAccessException &e)
                    {
                        cout << "HalfEdgeMesh::restorePlanes(): " << e.what() << endl;
                    }
                }
            }
        }
    }

    //start the last region growing
    m_regions.clear();


    int region_size   = 0;
    int region_number = 0;
    int default_region_threshold = (int)10 * log(m_faces.size());

    // Reset all used variables
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        m_faces[i]->m_used = false;
    }

    // Find all regions by regionGrowing with normal criteria
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        if(m_faces[i]->m_used == false)
        {
            NormalT n = m_faces[i]->getFaceNormal();

            RegionPtr region = new Region<VertexT, NormalT>(region_number);
            m_garbageRegions.insert(region);
            float almostOne = 0.999f;
            region_size = stackSafeRegionGrowing(m_faces[i], n, almostOne, region) + 1;

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
void HalfEdgeMesh<VertexT, NormalT>::finalize()
{
	std::cout << timestamp << "Checking face integreties." << std::endl;
    checkFaceIntegreties();

    std::cout << timestamp << "Finalizing mesh with classifier \"" << m_classifierType << "\"." << std::endl;

    boost::unordered_map<VertexPtr, size_t> index_map;

    size_t numVertices = m_vertices.size();
    size_t numFaces    = m_faces.size();
    size_t numRegions  = m_regions.size();
	float r = 0.0f;
	float g = 255.0f;
	float b = 0.0f;
    std::vector<uchar> faceColorBuffer;

    floatArr vertexBuffer( new float[3 * numVertices] );
    floatArr normalBuffer( new float[3 * numVertices] );
    ucharArr colorBuffer(  new uchar[3 * numVertices] );
    uintArr  indexBuffer(  new unsigned int[3 * numFaces] );

    // Set the Vertex and Normal Buffer for every Vertex.
    typename vector<VertexPtr>::iterator vertices_iter = m_vertices.begin();
    typename vector<VertexPtr>::iterator vertices_end  = m_vertices.end();
    for(size_t i = 0; vertices_iter != vertices_end; ++i, ++vertices_iter)
    {
        vertexBuffer[3 * i] =     (*vertices_iter)->m_position[0];
        vertexBuffer[3 * i + 1] = (*vertices_iter)->m_position[1];
        vertexBuffer[3 * i + 2] = (*vertices_iter)->m_position[2];

        normalBuffer [3 * i] =     -(*vertices_iter)->m_normal[0];
        normalBuffer [3 * i + 1] = -(*vertices_iter)->m_normal[1];
        normalBuffer [3 * i + 2] = -(*vertices_iter)->m_normal[2];

        // Map the vertices to a position in the buffer.
        // This is necessary since the old indices might have been compromised.
        index_map[*vertices_iter] = i;
    }

    string msg = timestamp.getElapsedTime() + "Calculating region sizes";
       ProgressBar progress(m_regions.size(), msg);
       for(size_t i = 0; i < m_regions.size(); i++)
       {
           m_regions[i]->calcArea();
           ++progress;
       }
       cout << endl;


    typename vector<FacePtr>::iterator face_iter = m_faces.begin();
    typename vector<FacePtr>::iterator face_end  = m_faces.end();
    for(size_t i = 0; face_iter != face_end; i++, ++face_iter)
    {
        indexBuffer[3 * i]      = index_map[(*(*face_iter))(0)];
        indexBuffer[3 * i + 1]  = index_map[(*(*face_iter))(1)];
        indexBuffer[3 * i + 2]  = index_map[(*(*face_iter))(2)];


        int surface_class = 1;

        if ((*face_iter)->m_region > 0)
        {
            // get the faces surface class (region id)
            surface_class = (*face_iter)->m_region;
            // label the region if not already done
            if (m_regionClassifier->generatesLabel())
            {
                Region<VertexT, NormalT>* region = m_regions.at(surface_class);
                if (!region->hasLabel())
                {
                    region->setLabel(m_regionClassifier->getLabel(surface_class));
                }
            }

            r = m_regionClassifier->r(surface_class);
            g = m_regionClassifier->g(surface_class);
            b = m_regionClassifier->b(surface_class);
        }

        colorBuffer[indexBuffer[3 * i]  * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i]  * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i]  * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 1] * 3 + 2] = b;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 0] = r;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 1] = g;
        colorBuffer[indexBuffer[3 * i + 2] * 3 + 2] = b;

        // now store the MeshBuffer face id into the face
        (*face_iter)->setBufferID(i);

        /// TODO: Implement materials
        /*faceColorBuffer.push_back( r );
        faceColorBuffer.push_back( g );
        faceColorBuffer.push_back( b );*/
    }

    // Label regions with Classifier
    LabelRegions();

	// jetzt alle regions labeln, falls der classifier das kann
    assignLBI();

    // Hand the buffers over to the Model class for IO operations.
    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }
    this->m_meshBuffer->setVertexArray( vertexBuffer, numVertices );
    this->m_meshBuffer->setVertexColorArray( colorBuffer, numVertices );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer, numVertices  );
    this->m_meshBuffer->setFaceArray( indexBuffer, numFaces );
	this->m_meshBuffer->setLabeledFacesMap( labeledFaces );
    //this->m_meshBuffer->setFaceColorArray( faceColorBuffer );
    this->m_finalized = true;

    // clean up
    labeledFaces.clear();
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::LabelRegions()
{

	typename vector<FacePtr>::iterator face_iter = m_faces.begin();
	typename vector<FacePtr>::iterator face_end  = m_faces.end();
	for(size_t i = 0; face_iter != face_end; i++, ++face_iter)
	{
		int surface_class = 1;

		if ((*face_iter)->m_region >= 0)
		{
			// get the faces surface class (region id)
			surface_class = (*face_iter)->m_region;

			// label the region if not already done
			if (m_regionClassifier->generatesLabel())
			{
				Region<VertexT, NormalT>* region = m_regions.at(surface_class);
				if (!region->hasLabel())
				{
					region->setLabel(m_regionClassifier->getLabel(surface_class));
				}
			}
		}
	}
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::assignLBI()
{
	// jetzt alle regions labeln, falls der classifier das kann
	if (m_regionClassifier->generatesLabel())
	{
		std::cout << timestamp << "Generating pre-labels with classifier type " << m_classifierType << std::endl;

		string label;
		typename vector<RegionPtr>::iterator region_iter = m_regions.begin();
		typename vector<RegionPtr>::iterator region_end  = m_regions.end();
		for(size_t i = 0; region_iter != region_end; ++i, ++region_iter)
		{
			// hier sind wir in den einzelnen regions und holen und die zugehörigen faces
			// darauf berechnen wir ein einziges mal das label und schreiben dann alle face ids in die map

			label = (*region_iter)->getLabel();

			vector<unsigned int> ids;
			typename vector<FacePtr>::iterator region_face_iter = (*region_iter)->m_faces.begin();
			typename vector<FacePtr>::iterator region_face_end  = (*region_iter)->m_faces.end();

			for(size_t i = 0; region_face_iter != region_face_end; ++i, ++region_face_iter)
			{
				ids.push_back((*region_face_iter)->getBufferID());
			}

			// if prelabel already exists in map, insert face id, else create new entry
			typename labeledFacesMap::iterator it = labeledFaces.find(label);

			if (it != labeledFaces.end())
			{
				it->second.insert(it->second.end(), ids.begin(), ids.end());
			}
			else
			{
				labeledFaces.insert(std::pair<std::string, std::vector<unsigned int> >(label, ids));
			}
		}
	}
}


template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::resetUsedFlags()
{
	for(size_t j=0; j<m_faces.size(); j++)
	{
		for(int k=0; k<3; k++)
		{
			(*m_faces[j])[k]->used=false;
		}
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::writeClassificationResult()
{
	m_regionClassifier->writeMetaInfo();
}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::finalizeAndRetesselate( bool genTextures, float fusionThreshold )
{
    std::cout << timestamp << "finalizeAndRetesselate mesh with classifier \"" << m_classifierType << "\"." << std::endl;

    // used Typedef's
    typedef std::vector<size_t>::iterator   intIterator;

    // default colors
    float r=0, g=200, b=0;

    map<Vertex<uchar>, unsigned int> materialMap;
	
    // Since all buffer sizes are unknown when retesselating
    // all buffers are instantiated as vectors, to avoid manual reallocation
    std::vector<float> vertexBuffer;
    std::vector<float> normalBuffer;
    std::vector<uchar> colorBuffer;
    std::vector<unsigned int> indexBuffer;
    std::vector<unsigned int> materialIndexBuffer;
    std::vector<Material*> materialBuffer;
    std::vector<float> textureCoordBuffer;

    // Reset used variables. Otherwise the getContours() function might not work quite as expected.
    resetUsedFlags();

    // Take all regions that are not in an intersection plane
    std::vector<size_t> nonPlaneRegions;
    // Take all regions that were drawn into an intersection plane
    std::vector<size_t> planeRegions;
    for( size_t i = 0; i < m_regions.size(); ++i )
    {
        if( !m_regions[i]->m_inPlane || m_regions[i]->m_regionNumber < 0)
        {
            nonPlaneRegions.push_back(i);
        }
        else
        {
            planeRegions.push_back(i);
        }
    }

    // keep track of used vertices to avoid doubles.
    map<Vertex<float>, unsigned int> vertexMap;
    Vertex<float> current;

    int globalMaterialIndex = 0;

    // Copy all regions that are non in an intersection plane directly to the buffers.
    for( intIterator nonPlane = nonPlaneRegions.begin(); nonPlane != nonPlaneRegions.end(); ++nonPlane )
    {
        size_t iRegion = *nonPlane;
        int surfaceClass = m_regions[iRegion]->m_regionNumber;

        // iterate over every face for the region number '*nonPlaneBegin'
        for( size_t i=0; i < m_regions[iRegion]->m_faces.size(); ++i )
        {
            size_t iFace=i;
            size_t pos;
            // loop over each vertex for this face
            for( int j=0; j < 3; j++ )
            {
                int iVertex = j;
                current = (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position;

                // look up the current vertex. If it was used before get the position for the indexBuffer.
                if( vertexMap.find(current) != vertexMap.end() )
                {
                    pos = vertexMap[current];
                }
                else
                {
                    pos = vertexBuffer.size() / 3;
                    vertexMap.insert(pair<Vertex<float>, unsigned int>(current, pos));
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.x );
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.y );
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.z );

                    if((*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal.length() > 0.0001)
                    {
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[0] );
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[1] );
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[2] );
                    }
                    else
                    {
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[0]);
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[1]);
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[2]);
                    }

                    //TODO: Color Vertex Traits stuff?
                    colorBuffer.push_back( r );
                    colorBuffer.push_back( g );
                    colorBuffer.push_back( b );

                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                }

                indexBuffer.push_back( pos );
            }

            if ( genTextures )
            {
                int one = 1;
                vector<VertexT> cv;
                this->m_pointCloudManager->searchTree()->kSearch((*m_regions[iRegion]->m_faces[iFace]).getCentroid(), one, cv);
                r = *((uchar*) &(cv[0][3])); /* red */
                g = *((uchar*) &(cv[0][4])); /* green */
                b = *((uchar*) &(cv[0][5])); /* blue */
            }

            // Try to find a material with the same color
            map<Vertex<uchar>, unsigned int >::iterator it = materialMap.find(Vertex<uchar>(r, g, b));
            if(it != materialMap.end())
            {
            	// If found, put material index into buffer
            	unsigned int position = it->second;
            	materialIndexBuffer.push_back(position);
            }
            else
            {
                Material* m = new Material;
                m->r = r;
                m->g = g;
                m->b = b;
                m->texture_index = -1;

                // Save material index
                materialBuffer.push_back(m);
                materialIndexBuffer.push_back(globalMaterialIndex);
                globalMaterialIndex++;
            }


        }
    }
    cout << timestamp << "Done copying non planar regions.";

    /*
         Done copying the simple stuff. Now the planes are going to be retesselated
         and the textures are generated if there are textures to generate at all.!
     */

    Texturizer<VertexT, NormalT>* texturizer = new Texturizer<VertexT, NormalT>(this->m_pointCloudManager);

    ///Tells which texture belongs to which material
    map<unsigned int, unsigned int > textureMap;

    string msg = timestamp.getElapsedTime() + "Applying textures to planes ";
    ProgressBar progress(planeRegions.size(), msg);
    for(intIterator planeNr = planeRegions.begin(); planeNr != planeRegions.end(); ++planeNr )
    {
        try
        {
            size_t iRegion = *planeNr;

            int surface_class = m_regions[iRegion]->m_regionNumber;
            //            r = (uchar)( 255 * fabs( cos( surfaceClass ) ) );
            //            g = (uchar)( 255 * fabs( sin( surfaceClass * 30 ) ) );
            //            b = (uchar)( 255 * fabs( sin( surfaceClass * 2 ) ) ) ;

            r = m_regionClassifier->r(surface_class);
            g = m_regionClassifier->g(surface_class);
            b = m_regionClassifier->b(surface_class);

            //textureBuffer.push_back( m_regions[iRegion]->m_regionNumber );

            // get the contours for this region
            vector<vector<VertexT> > contours = m_regions[iRegion]->getContours(fusionThreshold);

            // alocate a new texture
            TextureToken<VertexT, NormalT>* t = NULL;

            //retesselate these contours.
            std::vector<float> points;
            std::vector<unsigned int> indices;

            Tesselator<VertexT, NormalT>::getFinalizedTriangles(points, indices, contours);

            if( genTextures )
            {
                t = texturizer->texturizePlane( contours[0] );
                if(t)
                {
                    t->m_texture->save(t->m_textureIndex);
                }
            }

            // copy new vertex data:
            vertexBuffer.insert( vertexBuffer.end(), points.begin(), points.end() );

            // copy vertex, normal and color data.
            for(int j=0; j< points.size()/3; ++j)
            {
                normalBuffer.push_back( m_regions[iRegion]->m_normal[0] );
                normalBuffer.push_back( m_regions[iRegion]->m_normal[1] );
                normalBuffer.push_back( m_regions[iRegion]->m_normal[2] );

                colorBuffer.push_back( r );
                colorBuffer.push_back( g );
                colorBuffer.push_back( b );

                float u1 = 0;
                float u2 = 0;
                if(t) t->textureCoords( VertexT( points[j * 3 + 0], points[j * 3 + 1], points[j * 3 + 2]), u1, u2 );
                textureCoordBuffer.push_back( u1 );
                textureCoordBuffer.push_back( u2 );
                textureCoordBuffer.push_back(  0 );

            }

            // copy indices...
            // get the old end of the vertex buffer.
            size_t offset = vertexBuffer.size() - points.size();

            // calculate the index value for the old end of the vertex buffer.
            offset = ( offset / 3 );

            for(int j=0; j < indices.size(); j+=3)
            {
                /*if(indices[j] == indices[j+1] || indices[j+1] == indices[j+2] || indices[j+2] == indices[j])
                {
                    cout << "Detected degenerated face: " << indices[j] << " " << indices[j + 1] << " " << indices[j + 2] << endl;
                }*/

                // store the indices with the correct offset to the indices buffer.
                int a =  indices[j + 0] + offset;
                int b =  indices[j + 1] + offset;
                int c =  indices[j + 2] + offset;

                if(a != b && b != c && a != c)
                {
                    indexBuffer.push_back( a );
                    indexBuffer.push_back( b );
                    indexBuffer.push_back( c );
                }

            }

            // Create material and update buffer
            if(t)
            {
                map<unsigned int, unsigned int >::iterator it = textureMap.find(t->m_textureIndex);
                if(it == textureMap.end())
                {
                    //new texture -> create new material
                    Material* m = new Material;
                    m->r = r;
                    m->g = g;
                    m->b = b;
                    m->texture_index = t->m_textureIndex;
                    materialBuffer.push_back(m);
                    for( int j = 0; j < indices.size() / 3; j++ )
                    {
                        materialIndexBuffer.push_back(globalMaterialIndex);
                    }
                    textureMap[t->m_textureIndex] = globalMaterialIndex;
                    globalMaterialIndex++;
                }
                else
                {
                    //Texture already exists -> use old material
                    for( int j = 0; j < indices.size() / 3; j++ )
                    {
                        materialIndexBuffer.push_back(it->second);
                    }
                }
            }
            else
            {
                Material* m = new Material;
                m->r = r;
                m->g = g;
                m->b = b;
                m->texture_index = -1;
                materialBuffer.push_back(m);
            }


            if(t)
            {
                delete t;
            }

        }
        catch(...)
        {
            cout << timestamp << "Exception during finalization. Skipping triangle." << endl;
        };
        // Update counters
        ++progress;

    }


    // Label regions with Classifier
    LabelRegions();
    assignLBI();

    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }
    this->m_meshBuffer->setVertexArray( vertexBuffer );
    this->m_meshBuffer->setVertexColorArray( colorBuffer );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer );
    this->m_meshBuffer->setFaceArray( indexBuffer );
    this->m_meshBuffer->setVertexTextureCoordinateArray( textureCoordBuffer );
    this->m_meshBuffer->setMaterialArray( materialBuffer );
    this->m_meshBuffer->setFaceMaterialIndexArray( materialIndexBuffer );
	this->m_meshBuffer->setLabeledFacesMap( labeledFaces );
    this->m_finalized = true;
    cout << endl << timestamp << "Done retesselating." << endl;

    // Print texturizer statistics
    cout << *texturizer << endl;

    // Clean up
    delete texturizer;
    labeledFaces.clear();
    Tesselator<VertexT, NormalT>::clear();
} 

template<typename VertexT, typename NormalT>
HalfEdgeMesh<VertexT, NormalT>* HalfEdgeMesh<VertexT, NormalT>::retesselateInHalfEdge(float fusionThreshold)
{
	 std::cout << timestamp << "Retesselate mesh" << std::endl;

    // used Typedef's
    typedef std::vector<size_t>::iterator   intIterator;

    // default colors
    float r=0, g=200, b=0;

    map<Vertex<uchar>, unsigned int> materialMap;
	
    // Since all buffer sizes are unknown when retesselating
    // all buffers are instantiated as vectors, to avoid manual reallocation
    std::vector<float> vertexBuffer;
    std::vector<float> normalBuffer;
    std::vector<uchar> colorBuffer;
    std::vector<unsigned int> indexBuffer;
    std::vector<unsigned int> materialIndexBuffer;
    std::vector<Material*> materialBuffer;
    std::vector<float> textureCoordBuffer;

    // Reset used variables. Otherwise the getContours() function might not work quite as expected.
    resetUsedFlags();

    // Take all regions that are not in an intersection plane
    std::vector<size_t> nonPlaneRegions;
    // Take all regions that were drawn into an intersection plane
    std::vector<size_t> planeRegions;
    for( size_t i = 0; i < m_regions.size(); ++i )
    {
        if( !m_regions[i]->m_inPlane || m_regions[i]->m_regionNumber < 0)
        {
            nonPlaneRegions.push_back(i);
        }
        else
        {
            planeRegions.push_back(i);
        }
    }

    // keep track of used vertices to avoid doubles.
    map<Vertex<float>, unsigned int> vertexMap;
    Vertex<float> current;

    int globalMaterialIndex = 0;

    // Copy all regions that are non in an intersection plane directly to the buffers.
    for( intIterator nonPlane = nonPlaneRegions.begin(); nonPlane != nonPlaneRegions.end(); ++nonPlane )
    {
        size_t iRegion = *nonPlane;
        int surfaceClass = m_regions[iRegion]->m_regionNumber;

        // iterate over every face for the region number '*nonPlaneBegin'
        for( size_t i=0; i < m_regions[iRegion]->m_faces.size(); ++i )
        {
            size_t iFace=i;
            size_t pos;
            // loop over each vertex for this face
            for( int j=0; j < 3; j++ )
            {
                int iVertex = j;
                current = (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position;

                // look up the current vertex. If it was used before get the position for the indexBuffer.
                if( vertexMap.find(current) != vertexMap.end() )
                {
                    pos = vertexMap[current];
                }
                else
                {
                    pos = vertexBuffer.size() / 3;
                    vertexMap.insert(pair<Vertex<float>, unsigned int>(current, pos));
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.x );
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.y );
                    vertexBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.z );

                    if((*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal.length() > 0.0001)
                    {
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[0] );
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[1] );
                    	normalBuffer.push_back( (*m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[2] );
                    }
                    else
                    {
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[0]);
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[1]);
                    	normalBuffer.push_back((*m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[2]);
                    }

                    //TODO: Color Vertex Traits stuff?
                    colorBuffer.push_back( r );
                    colorBuffer.push_back( g );
                    colorBuffer.push_back( b );

                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                }

                indexBuffer.push_back( pos );
            }
        }
    }
    cout << timestamp << "Done copying non planar regions.";

    /*
         Done copying the simple stuff. Now the planes are going to be retesselated
         and the textures are generated if there are textures to generate at all.!
     */
    for(intIterator planeNr = planeRegions.begin(); planeNr != planeRegions.end(); ++planeNr )
    {
        try
        {
            size_t iRegion = *planeNr;

            int surface_class = m_regions[iRegion]->m_regionNumber;

            r = m_regionClassifier->r(surface_class);
            g = m_regionClassifier->g(surface_class);
            b = m_regionClassifier->b(surface_class);

            //textureBuffer.push_back( m_regions[iRegion]->m_regionNumber );

            // get the contours for this region
            vector<vector<VertexT> > contours = m_regions[iRegion]->getContours(fusionThreshold);

            // alocate a new texture
            TextureToken<VertexT, NormalT>* t = NULL;

            //retesselate these contours.
            std::vector<float> points;
            std::vector<unsigned int> indices;

            Tesselator<VertexT, NormalT>::getFinalizedTriangles(points, indices, contours);

            // copy new vertex data:
            vertexBuffer.insert( vertexBuffer.end(), points.begin(), points.end() );

            // copy vertex, normal and color data.
            for(int j=0; j< points.size()/3; ++j)
            {
                normalBuffer.push_back( m_regions[iRegion]->m_normal[0] );
                normalBuffer.push_back( m_regions[iRegion]->m_normal[1] );
                normalBuffer.push_back( m_regions[iRegion]->m_normal[2] );

                colorBuffer.push_back( r );
                colorBuffer.push_back( g );
                colorBuffer.push_back( b );

                float u1 = 0;
                float u2 = 0;
                //if(t) t->textureCoords( VertexT( points[j * 3 + 0], points[j * 3 + 1], points[j * 3 + 2]), u1, u2 );
                textureCoordBuffer.push_back( u1 );
                textureCoordBuffer.push_back( u2 );
                textureCoordBuffer.push_back(  0 );

            }

            // copy indices...
            // get the old end of the vertex buffer.
            size_t offset = vertexBuffer.size() - points.size();

            // calculate the index value for the old end of the vertex buffer.
            offset = ( offset / 3 );

            for(int j=0; j < indices.size(); j+=3)
            {
  
                int a =  indices[j + 0] + offset;
                int b =  indices[j + 1] + offset;
                int c =  indices[j + 2] + offset;

                if(a != b && b != c && a != c)
                {
                    indexBuffer.push_back( a );
                    indexBuffer.push_back( b );
                    indexBuffer.push_back( c );
                }

            }

        }
        catch(...)
        {
            cout << timestamp << "Exception during finalization. Skipping triangle." << endl;
        };

    }

    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }
    this->m_meshBuffer->setVertexArray( vertexBuffer );
    this->m_meshBuffer->setVertexColorArray( colorBuffer );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer );
    this->m_meshBuffer->setFaceArray( indexBuffer );
    cout << endl << timestamp << "Done retesselating." << endl;
		
	HalfEdgeMesh<VertexT, NormalT>* retased_mesh =  new HalfEdgeMesh(this->m_meshBuffer);
    Tesselator<VertexT, NormalT>::clear();
    return retased_mesh;
} 
	



template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::reduceMeshByCollapse(int n_collapses, VertexCosts<VertexT, NormalT> &c)
{
    string msg = timestamp.getElapsedTime() + "Collapsing edges...";
    ProgressBar progress(n_collapses, msg);

    // Try not to collapse more vertices than are in the mesh
    if(n_collapses >= (int)m_vertices.size())
    {
        n_collapses = (int)m_vertices.size();
    }
    //    cout << "Collapsing " << n_collapses << endl;
    //    for(int n = 0; n < n_collapses; n++)
    //    {
    //
    //    	priority_queue<vertexCost_p, vector<vertexCost_p>, cVertexCost> q;
    //    	for(size_t i = 0; i < this->m_vertices.size(); i++)
    //    	{
    //    		q.push(vertexCost_p(m_vertices[i], c(*m_vertices[i])));
    //    	}
    //
    //    	cout << q.size() << endl;
    //
    //    	// Try to collapse until a non-border-edges is found. If there are only
    //    	// borders left, collapse border edge with minimum cost
    //    	if(!q.empty())
    //    	{
    //    		// Save vertex with lowest costs
    //    		VertexPtr topVertex = q.top().first;
    //    		VertexPtr toCollapse = topVertex;
    //
    //    		// Check for border vertices
    //    		while(toCollapse->isBorderVertex() && !q.empty())
//    		{
//    			q.pop();
//    			toCollapse = q.top().first;
//    		}
//
//    		// If q is empty, no non-border vertex was found, collapse
//    		// first border vertex
//    		if(q.empty())
//    		{
//    			EdgePtr e = topVertex->getShortestEdge();
//    			cout << "E: " << e << endl;
//    			collapseEdge(e);
//    		}
//    		else
//    		{
//    			EdgePtr e = toCollapse->getShortestEdge();
//    			cout << "F: " << e << endl;
//    			collapseEdge(e);
//    		}
//    	}
//
//    	++progress;
//    }


  /*  priority_queue<vertexCost_p, vector<vertexCost_p>, cVertexCost> q;
    for(size_t i = 0; i < this->m_vertices.size(); i++)
    {
    	q.push(vertexCost_p(m_vertices[i], c(*m_vertices[i])));
    }

    for(int i = 0; i < n_collapses && !q.empty(); i++)
    {
    	VertexPtr v = q.top().first;
    	while(v->isBorderVertex() && !q.empty())
    	{
    		q.pop();
    		v = q.top().first;
    	}
    	EdgePtr e = v->getShortestEdge();
    	collapseEdge(e);
    }*/

}

template<typename VertexT, typename NormalT>
void HalfEdgeMesh<VertexT, NormalT>::getCostMap(std::map<VertexPtr, float> &costs, VertexCosts<VertexT, NormalT> &c)
{
	for(size_t i = 0; i < m_vertices.size(); i++)
	{
		costs[i] = c(*m_vertices[i]);
	}
}

} // namespace lvr
