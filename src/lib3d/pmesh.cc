// ******************************
// pmesh.cpp
//
// Progressive mesh class.
// This mesh can be simplified by
// removing edges & triangles, while
// retaining the same shape.
//
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#include <assert.h>
#include <float.h>

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#pragma warning(disable:4786) /* disable "identifier was truncated to '255' characters in the browser information" warning in Visual C++ 6*/
#endif

#include <set>
#include <map>
#include <ostream>

#include "pmesh.h"

// Used for debugging
#undef PRINT_DEBUG_INFO


// Constructor.  This will create the edge collapse list by
// calling createEdgeCollapseList
PMesh::PMesh(CubeMesh* mesh, EdgeCost ec)
{
	assert(mesh);
	assert(ec >= 0 && ec < MAX_EDGECOST);

	_mesh = mesh;
	_cost = ec;
	createEdgeCollapseList();
}

// Used for debugging
void dumpset(vertexPtrSet& ms)
{
	std::cout << "+++ Dumping set of vertices +++" << std::endl;

	vertexPtrSet::iterator iter;
	int i = 0;
	for (iter = ms.begin(); iter != ms.end(); ++iter)
	{
		std::cout << "\tvertex " << i++ << " in set: ";
		const vertexPtr v = *iter;
		std::cout << v._index;
		const vertex& vtx = v._meshptr->getVertex(v._index);
		std::cout << " cost: " << vtx.getCost();
		std::cout << " min edge vert: " << vtx.minCostEdgeVert();
		std::cout << std::endl;
	}
	std::cout << "+++ End of dumping set of vertices +++ "<< std::endl;
}

#ifdef PRINT_DEBUG_INFO

// Used for debugging
void checkConsistency(vertexPtrSet& ms,
					  vector<vertexPtrSet::iterator>& vsv,
					  CubeMesh& newmesh)
{
	int i;
	for (i = 0; i < newmesh.getNumTriangles(); ++i)
	{
		triangle& t = newmesh.getTri(i);
		if (!t.isActive()) continue;

		int v1, v2, v3;
		t.getVerts(v1, v2, v3);

		const vertex& cv = newmesh.getVertex(v1);
		assert(cv.hasTriNeighbor(t.getIndex()));
		assert(newmesh.getVertex(cv.getIndex()).isActive());

		const vertex& cv2 = newmesh.getVertex(v2);
		assert(cv2.hasTriNeighbor(t.getIndex()));
		assert(newmesh.getVertex(cv2.getIndex()).isActive());

		const vertex& cv3 = newmesh.getVertex(v3);
		assert(cv3.hasTriNeighbor(t.getIndex()));
		assert(newmesh.getVertex(cv3.getIndex()).isActive());
	}

	vertexPtrSet::iterator iter;

	for (iter = ms.begin(); iter != ms.end(); ++iter)
	{
		// check vertex ptr points to active vertex
		// const vertexPtr vp = *iter;
		assert(newmesh.getVertex((*iter)._index).isActive());
	}

	for (i = 0; i < newmesh.getNumVerts(); ++i)
	{
		if (newmesh.getVertex(i).isActive())
		{
			assert(vsv[i]->_index == i);
		}

	}
}
#endif //  PRINT_DEBUG_INFO

#ifndef NDEBUG
void PMesh::assertEveryVertActive(int nVerts, int nTri, CubeMesh &mesh)
{
	int i;
	for (i = 0; i < nVerts; ++i)
	{
		assert(mesh.getVertex(i).isActive());
	}
	for (i = 0; i < nTri; ++i)
	{
		assert(mesh.getTri(i).isActive());
	}
}
#endif

// Calculate the QEM matrices used to computer edge
// collapse costs.
void PMesh::calcQuadricMatrices(EdgeCost &cost, CubeMesh &mesh)
{
	if (QUADRICTRI == cost)
	{
		calcAllQMatrices(mesh, true);
	}
	else if (QUADRIC == cost)
	{
		calcAllQMatrices(mesh, false);
	};
}

// Calculate edge collapse costs.  Edges with low costs
// are collapsed first.
void PMesh::calcEdgeCollapseCosts(vertexPtrSet &vertSet, vector<vertexPtrSet::iterator> &vertSetVec, 
								  int nVerts, CubeMesh &mesh, EdgeCost &cost)
{
	int i;
	for (i = 0; i < nVerts; ++i)
	{
		vertex& currVert = mesh.getVertex(i);
		switch (cost)
		{
		case SHORTEST:
			shortEdgeCollapseCost(mesh, currVert);
			break;
		case MELAX:
			melaxCollapseCost(mesh, currVert);
			break;
		case QUADRIC: // deliberate fall through
		case QUADRICTRI:
			quadricCollapseCost(mesh, currVert);
			break;
		default:
			break;
		};

		vertexPtr v;
		v._index = i;
		v._meshptr = &mesh;

		vertSetVec[i] = vertSet.insert(v); // inserts a copy
	}

#ifdef PRINT_DEBUG_INFO
	int count=0; // for debug
	std::cout << "---- Initial State ----" << std::endl;
	mesh.dump();
	dumpset(vertSet);
	std::cout << "---- End Initial State ----" << std::endl;
#endif 
}

// We can't collapse Vertex1 to Vertex2 if Vertex2 is invalid.
// This can happen if Vertex2 was previously collapsed to a
// separate vertex.
void PMesh::insureEdgeCollapseValid(EdgeCollapse &ec, vertex &vc, CubeMesh &mesh, 
									const EdgeCost &cost, bool &bBadVertex)
{
	int nLoopCount = 0;
	for (;;) // at most this will loop twice -- the "to vertex" could have been removed, so it may need to be recalculated.
	{
		++nLoopCount;
		ec._vfrom = vc.getIndex(); // we'll collapse this vertex...
		ec._vto = vc.minCostEdgeVert(); // to this one

		if (-1 == vc.minCostEdgeVert())
		{
			// this point is isolated -- it's not connected to another point by an edge
			// Erase this point & get the next one
			bBadVertex = true;
			break;
		}

		if (nLoopCount > 2 || !mesh.getVertex(ec._vfrom).isActive())
		{
			bBadVertex = true;
			break;
		}

		// If not vertex active, recalc
		if (!mesh.getVertex(ec._vto).isActive())
		{
			switch (cost)
			{
			case SHORTEST:
				shortEdgeCollapseCost(mesh, vc);
				break;
			case MELAX:
				melaxCollapseCost(mesh, vc);
				break;
			case QUADRIC: // deliberate fall through!
			case QUADRICTRI:
				quadricCollapseCost(mesh, vc);
				break;
			default:
				break;
			};
		}
		else
		{
			break;
		}
	}
}

// Calculate the QEM for the "to vertex".  
// Tom Forsyth (Mucky Foot, ex-Bullfrog) says these should
// be averaged, not added(???) but we'll go with the 
// original algorithm.
void PMesh::setToVertexQuadric(vertex &to, vertex &from, const EdgeCost &cost)
{
	if (QUADRIC == cost || QUADRICTRI == cost)
	{
		int ct, ct2;

		double Qf[4][4], Qt[4][4];
		to.getQuadric(Qt);
		from.getQuadric(Qf);
		for (ct = 0; ct < 4; ++ct)
		{
			for (ct2 = 0; ct2 < 4; ++ct2)
			{
				Qt[ct][ct2] += Qf[ct][ct2];
			}
		}
		to.setQuadric(Qt);
		if (QUADRICTRI == cost)
		{
			double combinedTriArea = to.getQuadricSummedTriArea() + from.getQuadricSummedTriArea();
			to.setQuadricSummedTriArea(combinedTriArea);
		}
	}
}

// At this point, we have an edge collapse.  We're collapsing the "from vertex"
// to the "to vertex."  For all the surrounding triangles which use this edge, 
// update "from vertex" to the "to vertex".  Also keep track of the vertices
// in the surrounding triangles. 
void PMesh::updateTriangles(EdgeCollapse &ec, vertex &vc, set<int> &affectedVerts, CubeMesh &mesh)
{
	set<int>& triNeighbors = vc.getTriNeighbors();
	set<int>::iterator pos;

	for (pos = triNeighbors.begin(); pos != triNeighbors.end(); ++pos) 
	{
		// get triangle
		int triIndex = *pos;
		triangle& t = mesh.getTri(triIndex);
		if (!t.isActive()) continue;
		bool bRemoveTri = false;
		if (t.hasVertex(ec._vfrom) && t.hasVertex(ec._vto))
		{
			ec._trisRemoved.insert(triIndex);
			t.changeVertex(ec._vfrom, ec._vto); // update the vertex of this triangle

			bRemoveTri = true;

			t.setActive(false); // triangle has two vertices which are now the same, so it's just a line segment
		}
		else
		{
			t.changeVertex(ec._vfrom, ec._vto); // update the vertex of this triangle
			t.calcNormal(); // reset the normal for the triangle

			// make sure the "to" vertex knows about this triangle
			mesh.getVertex(ec._vto).addTriNeighbor(triIndex);
			
			// If the triangle has an area effectively equal to 0, remove it.
			// NOTE: should this be done?  The triangle could get bigger through 
			// another edge collapse in another direction.
			if (t.calcArea() < 1e-6) {
				t.setActive(false);
				ec._trisRemoved.insert(triIndex);
				bRemoveTri = true;
			} else {
				ec._trisAffected.insert(triIndex);
			}

		}
		// update set of affected verts
		// note that if you insert the same element twice, 
		// it will only be stored once.
		int vert1, vert2, vert3;
		t.getVerts(vert1, vert2, vert3);

		affectedVerts.insert(vert1);
		affectedVerts.insert(vert2);
		affectedVerts.insert(vert3);

		// If triangle is being removed, update each vertex which references it.
		if (bRemoveTri)
		{
			mesh.getVertex(vert1).removeTriNeighbor(triIndex);
			mesh.getVertex(vert2).removeTriNeighbor(triIndex);
			mesh.getVertex(vert3).removeTriNeighbor(triIndex);
		}
	}
}

// These vertices are not in the current collapse, but are in the triangles
// which share the collapsed edge.
void PMesh::updateAffectedVertNeighbors(vertex &vert, const EdgeCollapse &ec, 
										const set<int> &affectedVerts)
{
	if (vert.getIndex() != ec._vto)
	{
		vert.addVertNeighbor(ec._vto); // make sure vertex knows it has a new neighbor
	}
	else
	{
		set<int>::iterator mappos2;

		// Make sure the "to" vertex knows about its
		// new neighbors
		for (mappos2 = affectedVerts.begin(); mappos2 != affectedVerts.end(); ++mappos2)
		{
			if (*mappos2 != ec._vto)
			{
				vert.addVertNeighbor(*mappos2); 
			}
		}
	}

	// get rid of deleted vertex
	vert.removeVertNeighbor(ec._vfrom);
}

// Reset the edge collapse costs of vertices which were
// affected by a previous edge collapse.
void PMesh::resetAffectedVertCosts(const EdgeCost &cost, CubeMesh &mesh, vertex &vert)
{
	switch (cost)
	{
	case SHORTEST:
		shortEdgeCollapseCost(mesh, vert);
		break;
	case MELAX:
		melaxCollapseCost(mesh, vert);
		break;
	case QUADRIC: // deliberate fall through!
	case QUADRICTRI:
		// Don't calculate the quadric Collapse cost yet, because we
		// can't do that until the Q matrix is calculated for each vertes
		// that is a neighbor of this vertex.  
		// We will calculate the quadric collapse cost later for all affected
		// vertices which are still active.
		break;
	default:
		break;
	};
}


// If this vertex has no active triangles (i.e. triangles which have
// not been removed from the mesh) then set it to inactive.
void PMesh::removeVertIfNecessary(vertex &vert, vertexPtrSet &vertSet, 
								  vector<vertexPtrSet::iterator> &vertSetVec, 
								  CubeMesh &mesh, const EdgeCost &cost, 
									set<int> &affectedQuadricVerts)
{
	bool bActiveVert = false;
	set<int>& mytriNeighbors = vert.getTriNeighbors();
	set<int>::iterator pos2;
	for (pos2 = mytriNeighbors.begin(); pos2 != mytriNeighbors.end(); ++pos2) 
	{
		// get triangle
		int triIndex = *pos2;
		triangle& t = mesh.getTri(triIndex);
		if (t.isActive()) {
			bActiveVert = true;
			break;
		}
	}

	if (bActiveVert) { // if vert is active
		vertexPtr vp;
		vp._index = vert.getIndex();
		vp._meshptr = &mesh;
		vertSetVec[vp._index] = vertSet.insert(vp);

		assert(vertSetVec[vp._index]->_index == vp._index);
		mesh.getVertex(vp._index).setActive(true); 

		// If we're calculating quadric costs, keep track of
		// every active vertex which was affect by this collapse,
		// so we can recalculate collapse costs.
		if (QUADRIC == cost || QUADRICTRI == cost) {
			affectedQuadricVerts.insert(vert.getIndex());
		}
#ifdef PRINT_DEBUG_INFO
		std::cout << "\tvert affected: " << vert.getIndex() << std::endl;
#endif
	}
	else {
#ifdef PRINT_DEBUG_INFO
		std::cout << "\tvert removed: " << vert.getIndex() << std::endl;
#endif
		mesh.getVertex(vert.getIndex()).setActive(false);
	}
}

// Update the vertices affected by the most recent edge collapse
void PMesh::updateAffectedVerts(CubeMesh &mesh, vector<vertexPtrSet::iterator> &vertSetVec, 
								vertexPtrSet &vertSet, const EdgeCollapse &ec, 
								set<int> &affectedVerts, const EdgeCost &cost, 
								set<int> &affectedQuadricVerts)
{
	set<int>::iterator mappos;
	for (mappos = affectedVerts.begin(); mappos != affectedVerts.end(); ++mappos)
	{

		vertex& vert = mesh.getVertex(*mappos);
		assert(vert.getIndex() == *mappos);

		// Always erase, maybe add in.
		// Can't change in place, 'cause will screw up order of set
		vertSet.erase(vertSetVec[*mappos]);
		vertSetVec[*mappos] = vertSet.end(); // set to "invalid" value

		updateAffectedVertNeighbors(vert, ec, affectedVerts);

		// reset values for affected vertices
		resetAffectedVertCosts(cost, mesh, vert);

		// Remove vertex if it's not attached to any active triangle
		removeVertIfNecessary(vert, vertSet, vertSetVec, mesh,
								cost, affectedQuadricVerts);
	}
}

// Recalculate the QEM matrices (yeah, that's redundant) if we're
// using the Quadrics to calculate edge collapse costs.
void PMesh::recalcQuadricCollapseCosts(set<int> &affectedQuadricVerts, 
									   CubeMesh &mesh, const EdgeCost &cost)
{
	if (QUADRIC == cost || QUADRICTRI == cost)
	{
		set<int>::iterator mappos;
		for (mappos = affectedQuadricVerts.begin(); mappos != affectedQuadricVerts.end(); ++mappos)
		{			
			vertex& vert = mesh.getVertex(*mappos);
			quadricCollapseCost(mesh, vert);
		}
	}
}

// Calculate the list of edge collapses.  Each edge collapse
// consists of two vertices:  a "from vertex" and a "to vertex".
// The "from vertex" is collapsed to the "to vertex".  The
// "from vertex" is removed from the mesh.
void PMesh::buildEdgeCollapseList(CubeMesh &mesh, const EdgeCost &cost, 
								  list<EdgeCollapse> &edgeCollList,
									vertexPtrSet &vertSet, 
									vector<vertexPtrSet::iterator> &vertSetVec)
{
	for (;;)
	{
		if (0 == vertSet.size())
		{
			// we're done
			break;
		}

#ifdef PRINT_DEBUG_INFO
		// check consistency in data structures
		checkConsistency(vertSet, vertSetVec, mesh);
#endif

		const vertexPtr vp = *(vertSet.begin()); // This is a copy of the first element
		vertex vc = mesh.getVertex(vp._index);
		assert(vp._index == vc.getIndex());

		EdgeCollapse ec; // create EdgeCollapse structure

		bool bBadVertex = false;

		// Make sure this edge collapse has a valid "to vertex"
		insureEdgeCollapseValid(ec, vc, mesh, cost, bBadVertex);

		mesh.getVertex(ec._vfrom).setActive(false);
		vertSet.erase(vertSet.begin());

		if (bBadVertex) {
			continue;
		}

#ifdef PRINT_DEBUG_INFO
		std::cout << "from: " << ec._vfrom << " to: " << ec._vto << std::endl;
#endif

		vertex& to = mesh.getVertex(ec._vto);
		vertex& from = mesh.getVertex(ec._vfrom);

		setToVertexQuadric(to, from, cost);

		set<int> affectedVerts;

		// We are removing a vertex and an edge.  Look at all triangles
		// which use this vertex.  Each of these triangles is either being
		// removed or updated with a new vertex.
		updateTriangles(ec, vc, affectedVerts, mesh);
		
		set<int> affectedQuadricVerts;

		// These vertices were in triangles which either were removed or
		// were updated with new vertices.  Removed these vertices if they're
		// not connected to an active triangle.  Update these vertices if they're
		// still being displayed.
		updateAffectedVerts(mesh, vertSetVec, vertSet, ec, affectedVerts,
							cost, affectedQuadricVerts);

		// If using the quadric collapse method, 
		// recalculate the edge collapse costs for the affected vertices.
		recalcQuadricCollapseCosts(affectedQuadricVerts, mesh, cost);

#ifdef PRINT_DEBUG_INFO
		std::cout << "---- Collapse # "<< count++ << " ----" << std::endl;
		mesh.dump();
		ec.dumpEdgeCollapse();
		dumpset(vertSet);
#endif

		edgeCollList.push_back(ec); // inserts a copy
	}
}

// Where most of the work of the program is done.
// This will create a list of edge collapses.  Each edge collapse
// is a set of 2 vertices, a from vertex & a to vertex.  The from
// vertex will be collapsed to the to vertex.  No new vertices are created,
// only vertices in the original mesh are used.
void PMesh::createEdgeCollapseList()
{
	// okay, get list of verts, tris
	// for each vert, calc cost
	// add to edge collapse list

	// Copy the original mesh
	_newmesh = *_mesh;

	_edgeCollList.clear(); // empty list

	int nVerts = _newmesh.getNumVerts();
	int nTri = _newmesh.getNumTriangles();

	_nVisTriangles = nTri; // number of visible triangles
	
	// assert each vert is active -- sanity check

#ifndef NDEBUG
	assertEveryVertActive(nVerts, nTri, _newmesh);
#endif

	// calculate all 4x4 Q matrices for each vertex 
	// if using the Quadric method
	calcQuadricMatrices(_cost, _newmesh);

	// This is a set of vertex pointers, ordered by edge collapse cost.
	vertexPtrSet vertSet;
	vector<vertexPtrSet::iterator> vertSetVec(nVerts);

	// Go through, calc cost here for all vertices
	calcEdgeCollapseCosts(vertSet, vertSetVec, nVerts, _newmesh, _cost);

	// For all vertices:
	//	find lowest cost
	//	store the edge collapse structure
	//	update all verts, triangles affected by the edge collapse
	buildEdgeCollapseList(_newmesh, _cost, _edgeCollList,
							vertSet, vertSetVec);

	_newmesh = *_mesh;
	for (int i = 0; i < nTri; ++i)
	{
		_newmesh.getTri(i).setActive(true);
	}

	// set iterator to point to beginning
	_edgeCollapseIter = _edgeCollList.begin();
}


// Calculate the 4x4 Q Matrix used for the Quadric calculation
// for each vertex
void PMesh::calcAllQMatrices(CubeMesh& mesh, bool bUseTriArea)
{
	set<border> borderSet;

	int nVerts = mesh.getNumVerts();

	for (int i = 0; i < nVerts; ++i)
	{
		vertex& currVert = mesh.getVertex(i);

		currVert.calcQuadric(mesh, bUseTriArea);

		double myQ[4][4];
		currVert.getQuadric(myQ);
		double triArea = 0;
		
		if (QUADRICTRI == _cost)
		{
			triArea = currVert.getQuadricSummedTriArea();
		}

		calcQuadricError(myQ, currVert, triArea);

		// Is the current vertex on a border?  If so, get the
		// edge information
		currVert.getAllBorderEdges(borderSet, mesh);
	}

	// Keep the mesh borders from being "eaten away".
	if (!borderSet.empty())
	{
		applyBorderPenalties(borderSet, mesh);
	}
}

// Border penalties are used to prevent mesh edges from collapsing
// too soon.  This causes holes & T-junctions in the mesh to expand,
// and causes the edges of the mesh to be "eaten away".  2-manifold,
// closed meshes will not need to worry about this, and won't have
// any border penalties.
void PMesh::applyBorderPenalties(set<border> &borderSet, CubeMesh &mesh)
{
	set<border>::iterator pos;

	for (pos = borderSet.begin(); pos != borderSet.end(); ++pos) 
	{
		// First, determine the plane equation of plane perpendicular 
		// to the edge triangle.

		border edgeInfo = *pos;

		vertex& v1 = mesh.getVertex(edgeInfo.vert1);
		vertex& v2 = mesh.getVertex(edgeInfo.vert2);

		Vec3 &vec1 = v1.getXYZ();
		Vec3 &vec2 = v2.getXYZ();

		Vec3 edge = vec1 - vec2;

		triangle &tri = mesh.getTri(edgeInfo.triIndex);
		Vec3 normal = tri.getNormalVec3();

		Vec3 abc = edge.unitcross(normal);
		float &a = abc.x;
		float &b = abc.y;
		float &c = abc.z;

		float d = -(abc.dot(vec1));


		double QuadricConstraint[4][4];
		// NOTE: we could optimize this a bit by calculating values
		// like a * b and then using that twice (for Quadric[0][1] and Quadric[1][0]),
		// etc., since the matrix is symmetrical.  For now, I don't think
		// it's worth it.
		QuadricConstraint[0][0] = BOUNDARY_WEIGHT * a * a;
		QuadricConstraint[0][1] = BOUNDARY_WEIGHT * a * b;
		QuadricConstraint[0][2] = BOUNDARY_WEIGHT * a * c;
		QuadricConstraint[0][3] = BOUNDARY_WEIGHT * a * d;

		QuadricConstraint[1][0] = BOUNDARY_WEIGHT * b * a;
		QuadricConstraint[1][1] = BOUNDARY_WEIGHT * b * b;
		QuadricConstraint[1][2] = BOUNDARY_WEIGHT * b * c;
		QuadricConstraint[1][3] = BOUNDARY_WEIGHT * b * d;

		QuadricConstraint[2][0] = BOUNDARY_WEIGHT * c * a;
		QuadricConstraint[2][1] = BOUNDARY_WEIGHT * c * b;
		QuadricConstraint[2][2] = BOUNDARY_WEIGHT * c * c;
		QuadricConstraint[2][3] = BOUNDARY_WEIGHT * c * d;

		QuadricConstraint[3][0] = BOUNDARY_WEIGHT * d * a;
		QuadricConstraint[3][1] = BOUNDARY_WEIGHT * d * b;
		QuadricConstraint[3][2] = BOUNDARY_WEIGHT * d * c;
		QuadricConstraint[3][3] = BOUNDARY_WEIGHT * d * d;

		// Now add the constraint quadric to the quadrics for both of the 
		// vertices.
		double Q1[4][4], Q2[4][4];
		v1.getQuadric(Q1);
		v2.getQuadric(Q2);
		for (int ct = 0; ct < 4; ++ct)
		{
			for (int ct2 = 0; ct2 < 4; ++ct2)
			{
				Q1[ct][ct2] += QuadricConstraint[ct][ct2];
				Q2[ct][ct2] += QuadricConstraint[ct][ct2];
			}
		}
		v1.setQuadric(Q1);
		v2.setQuadric(Q2);
	}
}


// Calculate the cost of collapsing this vertex using the
// "shortest edge" method.
double PMesh::shortEdgeCollapseCost(CubeMesh& m, vertex& v)
{
	// get list of all active neighbors
	// calculate shortest edge
	// what if no neighbors??
	// return cost
	float mincost = FLT_MAX; // from float.h
	bool bNeighborFound = false;

	set<int>& neighbors = v.getVertNeighbors();
	set<int>::iterator pos;
	for (pos = neighbors.begin(); pos != neighbors.end(); ++pos) 
	{
		vertex& n = m.getVertex(*pos);
		if (!n.isActive()) continue;
		if (n == v) continue;

		// calc cost
		Vec3 s = v.getXYZ() - n.getXYZ();
		float cost = s.length();

		if (cost < mincost)
		{
			bNeighborFound = true;
			mincost = cost;
			v.setEdgeRemoveCost(cost);
			v.setMinCostEdgeVert(*pos);
			assert(v.minCostEdgeVert() >= 0 && v.minCostEdgeVert() < m.getNumVerts());
		}
	}
	
	if (bNeighborFound) {
		return mincost;
	} else {
		return FLT_MAX; // vertex not connected to an edge
	}
}


// Helper function for melaxCollapseCost().  This function
// will loop through all the triangles to which this vertex
// belongs.
void PMesh::calcMelaxMaxValue(CubeMesh &mesh, set<int> &adjfaces, 
							  vertex &v, set<int> &tneighbors,
								float &retmaxValue, 
								bool &bMaxValueFound)
{
	bool bMinValueFound  = false;
	if (adjfaces.size() > 1 && v.isBorder(mesh))
	{
		retmaxValue = 1.0f;
		bMaxValueFound = true;
	}
	else
	{
		// now go through all triangles next to vertex, 
		set<int>::iterator pos2;
		for (pos2 = tneighbors.begin(); pos2 != tneighbors.end(); ++pos2) 
		{
			float min = 1;
			int triIndex = *pos2;
			triangle& t = mesh.getTri(triIndex);
			if (!t.isActive()) continue;

			bMinValueFound = false;
			set<int>::iterator pos3;
			for (pos3 = adjfaces.begin(); pos3 != adjfaces.end(); ++pos3) 
			{
				int triIndex3 = *pos3;
				triangle& t3 = mesh.getTri(triIndex3);
				if (!t3.isActive()) continue;
				Vec3 n(t.getNormal());
				Vec3 n2(t3.getNormal());
				float dot = n.dot(n2); // cross product of face next to vertex & face along edge
				float value = (1.0f - dot) * 0.5f; // don't really need to mult. by 0.5, unless want value < 1.0
				if (value < min)
				{
					min = value;
					bMinValueFound = true;
				}
			}
			
			if (bMinValueFound && min > retmaxValue) {
				retmaxValue = min;
				bMaxValueFound = true;
			}
		}
	}
}

// Calculate the cost of collapsing this vertex using the
// "Stan Melax PolyChop" method.
double PMesh::melaxCollapseCost(CubeMesh& mesh, vertex& v)
{
	set<int>& vneighbors = v.getVertNeighbors();
	set<int>& tneighbors = v.getTriNeighbors();
	set<int>::iterator pos;
	set<int>::iterator pos2;
	float retmaxValue = -2.0;
	float mincost = 1e6;
	for (pos = vneighbors.begin(); pos != vneighbors.end(); ++pos) 
	{
		if (v.getIndex() == *pos) continue; // vertex has itself as a neighbor, by mistake //!NEW

		// get adj. faces
		set<int> adjfaces;
		// get triangle neighbors of this vertex
		for (pos2 = tneighbors.begin(); pos2 != tneighbors.end(); ++pos2) 
		{
			// get triangle
			int triIndex = *pos2;
			triangle& t = mesh.getTri(triIndex);
			if (t.isActive() && t.hasVertex(*pos)) 
			{
				adjfaces.insert(triIndex); // triangle contains both vertex & vertex neighbor
			}
		}

		bool bMaxValueFound  = false;

		// If there is only 1 face shared between the 2 vertices, then the 
		// edge is at the edge of the model.  Set it equal to 1.0, which is
		// the max value of curvature we can give it.  We do this so the 
		// edges of the model won't collapse inward.  Note that if the model 
		// has a nice manifold surface, every edge will be shared by  at least 2
		// triangles, and it won't be an issue.

		// This idea comes from Stan Melax's follup up web page to his PolyChop
		// algorithm. (http://www.melax.com/polychop/feedback/index.html)
		// or (http://www.cs.ualberta.ca/~melax/polychop/feedback)
		calcMelaxMaxValue(mesh, adjfaces, v, tneighbors,
							retmaxValue, bMaxValueFound);
		if (bMaxValueFound)
		{
			Vec3 v1 = v.getXYZ();
			Vec3 v2 = mesh.getVertex(*pos).getXYZ();
			Vec3 v3 = v1 - v2;
			retmaxValue *= v3.length();
			
			if (retmaxValue < mincost) 
			{
				mincost = retmaxValue;
				v.setEdgeRemoveCost(retmaxValue);
				v.setMinCostEdgeVert(*pos);
			}
		}
	}
	return mincost;
}

// Calculate the cost of collapsing this vertex using the
// "Garland & Heckbert Quadrics" method.
double PMesh::quadricCollapseCost(CubeMesh& m, vertex& v)
{
	// get list of all active neighbors
	// calculate quadric cost
	double mincost = FLT_MAX; // from float.h
	bool bNeighborFound = false;

	double Q1[4][4];
	v.getQuadric(Q1);

	set<int>& neighbors = v.getVertNeighbors();
	set<int>::iterator pos;
	for (pos = neighbors.begin(); pos != neighbors.end(); ++pos) 
	{

		vertex& n = m.getVertex(*pos);
		if (!n.isActive()) continue;
		if (n == v) continue;

		double Q2[4][4];
		double Qsum[4][4];

		// add two 4x4 Q matrices
		n.getQuadric(Q2);

		for(int i = 0; i < 4; ++i) {
			for ( int j = 0; j < 4; ++j) {
				Qsum[i][j] = Q1[i][j] + Q2[i][j];
			}
		}

		double triArea = 0;
		if (QUADRICTRI == _cost)
		{
			triArea = v.getQuadricSummedTriArea() + n.getQuadricSummedTriArea();
		}

		// calc cost
		double cost = calcQuadricError(Qsum, n, triArea);

		if (cost < mincost)
		{
			bNeighborFound = true;
			mincost = cost;
			v.setEdgeRemoveCost(cost);
			v.setMinCostEdgeVert(*pos);
			assert(v.minCostEdgeVert() >= 0 && v.minCostEdgeVert() < m.getNumVerts());
		}
	}
	
	if (bNeighborFound) {
		return mincost;
	} else {
		return FLT_MAX; // vertex not connected to an edge
	}
}
		
// Calculate the quadric error if using that edge collapse
// algorithm.  We're calculating
//
//  T    
// v  Q v
//

// This is the vertex multiplied by the 4x4 Q matrix, multiplied
// by the vertex again.
double PMesh::calcQuadricError(double Qsum[4][4], vertex& v, double triArea)
{
	double cost;

	// 1st, consider vertex v a 1x4 matrix: [v.x v.y v.z 1]
	// Multiply it by the Qsum 4x4 matrix, resulting in a 1x4 matrix

	double result[4];

	const Vec3 v3 = v.getXYZ();

	result[0] = v3.x * Qsum[0][0] + v3.y * Qsum[1][0] +
				v3.z * Qsum[2][0] + 1 * Qsum[3][0];
	result[1] = v3.x * Qsum[0][1] + v3.y * Qsum[1][1] +
				v3.z * Qsum[2][1] + 1 * Qsum[3][1];
	result[2] = v3.x * Qsum[0][2] + v3.y * Qsum[1][2] +
				v3.z * Qsum[2][2] + 1 * Qsum[3][2];
	result[3] = v3.x * Qsum[0][3] + v3.y * Qsum[1][3] +
				v3.z * Qsum[2][3] + 1 * Qsum[3][3];

	// Multiply this 1 x 4 matrix by the vertex v transpose (a 4 x 1 matrix).
	// This is just the dot product.

	cost =	result[0] * v3.x + result[1] * v3.y +
			result[2] * v3.z + result[3] * 1; 

	if (QUADRICTRI == _cost && triArea != 0)
	{
		cost /= triArea;
	}

	return cost;
}

// Collapse an edge (remove one vertex & edge, and possibly some triangles.)
bool PMesh::collapseEdge()
{
	// Iterator always points to next collapse to perform
	if (_edgeCollapseIter == _edgeCollList.end()) return false; // no more edge collapses in list
	EdgeCollapse& ec = *_edgeCollapseIter;

	set<int> affectedVerts; // vertices affected by this edge collapse
	int v1, v2, v3; // vertex indices

	// Remove triangles 
	set<int>::iterator tripos;
	for (tripos = ec._trisRemoved.begin(); tripos != ec._trisRemoved.end(); ++tripos) 
	{
		// get triangle
		int triIndex = *tripos;
		triangle& t = _newmesh.getTri(triIndex);
		t.getVerts(v1, v2, v3); // get triangle vertices
		t.setActive(false);
		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// Adjust vertices of triangles
	for (tripos = ec._trisAffected.begin(); tripos != ec._trisAffected.end(); ++tripos) 
	{
		// get triangle
		int triIndex = *tripos;
		triangle& t = _newmesh.getTri(triIndex);
		t.changeVertex(ec._vfrom, ec._vto); // update the vertex of this triangle
		t.calcNormal(); // reset the normal for the triangle
		t.getVerts(v1, v2, v3); // get triangle vertices
		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// redo the vertex normal for the vertices affected.  these are
	// vertices of triangles which were shifted around as a result
	// of this edge collapse.
	set<int>::iterator affectedVertsIter;
	for (affectedVertsIter = affectedVerts.begin(); affectedVertsIter != affectedVerts.end(); ++affectedVertsIter) 
	{
		if (ec._vfrom == *affectedVertsIter) continue; // skip the from vertex -- it's no longer active

		// We have the affected vertex index, so redo the its normal (for Gouraud shading);
		_newmesh.calcOneVertNormal(*affectedVertsIter);
	}

	// Since iterator always points to next collapse to perform, go to the next
	// collapse in list.
	++_edgeCollapseIter;

	_nVisTriangles -=  ec._trisRemoved.size();

	return true;
}

// Split a vertex (add one vertex & edge, and possibly some triangles.)
bool PMesh::splitVertex()
{
	// Iterator always points to next collapse to perform.
	// But we don't want to collapse, we want to undo the previous
	// collapse.  Go to that edge collapse, unless we're at the front of
	// the list, in which case there are no collapses to undo (the mesh
	// is fully displayed w/o any collapses).
	if (_edgeCollapseIter == _edgeCollList.begin()) return false;
	--_edgeCollapseIter; // go to previous edge collapse, so we can undo it
	EdgeCollapse& ec = *_edgeCollapseIter;

	set<int> affectedVerts; // vertices affected by this edge collapse
	int v1, v2, v3; // vertex indices

	// Add triangles which were removed
	set<int>::iterator tripos;
	for (tripos = ec._trisRemoved.begin(); tripos != ec._trisRemoved.end(); ++tripos) 
	{
		// get triangle
		int triIndex = *tripos;
		triangle& t = _newmesh.getTri(triIndex);
		t.setActive(true);
		t.getVerts(v1, v2, v3); // get triangle vertices
		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// Adjust vertices of triangles
	for (tripos = ec._trisAffected.begin(); tripos != ec._trisAffected.end(); ++tripos) 
	{
		// get triangle
		int triIndex = *tripos;
		triangle& t = _newmesh.getTri(triIndex);
		t.changeVertex(ec._vto, ec._vfrom); // update the vertex of this triangle
		t.calcNormal(); // reset the normal for the triangle
		t.getVerts(v1, v2, v3); // get triangle vertices
		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// redo the vertex normal for the vertices affected.  these are
	// vertices of triangles which were shifted around as a result
	// of this edge split.
	set<int>::iterator affectedVertsIter;
	for (affectedVertsIter = affectedVerts.begin(); affectedVertsIter != affectedVerts.end(); ++affectedVertsIter) 
	{
		// We have the affected vertex index, so redo the its normal (for Gouraud shading);
		_newmesh.calcOneVertNormal(*affectedVertsIter);
	}

	_nVisTriangles +=  ec._trisRemoved.size();
	
	// Since iterator always points to next collapse to perform, leave it here.
	return true;
}

// Return a short text description of the current Edge Cost method
char* PMesh::getEdgeCostDesc()
{
	switch(_cost)
	{
	case PMesh::SHORTEST:
		{
			return "Shortest Edge";
			break;
		};
	case PMesh::MELAX:
		{
			return "Melax";
			break;
		};
	case PMesh::QUADRIC:
		{
			return "Quadric";
			break;
		};
	case PMesh::QUADRICTRI:
		{
			return "Quadric Weighted by Triangle Area";
			break;
		};
	default:
		assert(false);
		return "Error";
		// error
	};
}


// bool PMesh::toDXF(char *filename)
// {
//   //return _newmesh.toDXF(filename);
  
// }
