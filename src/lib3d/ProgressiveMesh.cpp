/*
 * ProgressiveMesh.cpp
 *
 *  Created on: 17.12.2008
 *      Author: twiemann
 */

#include "ProgressiveMesh.h"


ProgressiveMesh::ProgressiveMesh(LinkedTriangleMesh* m, EdgeCost ec)
{
	assert(m);
	assert(ec >= 0 && ec < MAX_EDGECOST);

	original_mesh = m;
	edge_cost = ec;
	m->calcVertexNormals();
	createEdgeCollapseList();
}


void ProgressiveMesh::calcQuadricMatrices(EdgeCost &cost, LinkedTriangleMesh &mesh)
{
	if (QUADRICTRI == cost)
	{
		calcAllQuadricMatrices(mesh, true);
	}
	else if (QUADRIC == cost)
	{
		calcAllQuadricMatrices(mesh, false);
	};
}

void ProgressiveMesh::calcEdgeCollapseCosts(
		VertexPointerSet &vertSet, vector<VertexPointerSet::iterator> &vertSetVec,
		int nVerts, LinkedTriangleMesh &mesh, EdgeCost &cost)
{
	static int c = 0;

	int i;
	for (i = 0; i < nVerts; i++)
	{
		TriangleVertex& currVert = mesh.getVertex(i);
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

		VertexPointer v;
		v.index = i;
		v.mesh = &mesh;

		vertSetVec[i] = vertSet.insert(v); // inserts a copy
		c++;
	}

	cout << c << endl;
//#ifdef PRINT_DEBUG_INFO
//	int count=0; // for debug
//	std::cout << "---- Initial State ----" << std::endl;
//	mesh.dump();
//	dumpset(vertSet);
//	std::cout << "---- End Initial State ----" << std::endl;
//#endif
}

void ProgressiveMesh::ensureEdgeCollapseIsValid(
		EdgeCollapse &ec, TriangleVertex &vc,
		LinkedTriangleMesh &mesh, const EdgeCost &cost, bool &bBadVertex)
{
	int nLoopCount = 0;
	for (;;) // at most this will loop twice -- the "to vertex" could have been removed, so it may need to be recalculated.
	{
		++nLoopCount;
		ec.from = vc.getIndex();       // we'll collapse this vertex...
		ec.to = vc.getMinCostNeighbor();  // to this one

		if (-1 == vc.getMinCostNeighbor())
		{
			// this point is isolated -- it's not connected to another point by an edge
			// Erase this point & get the next one
			bBadVertex = true;
			break;
		}

		if (nLoopCount > 2 || !mesh.getVertex(ec.from).isActive())
		{
			bBadVertex = true;
			break;
		}

		// If not vertex active, recalc
		if (!mesh.getVertex(ec.to).isActive())
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

void ProgressiveMesh::setToVertexQuadric(
		TriangleVertex &to, TriangleVertex &from, const EdgeCost &cost)
{
	if (QUADRIC == cost || QUADRICTRI == cost)
	{
		int ct, ct2;

		float Qf[4][4], Qt[4][4];
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
			float combinedTriArea = to.getSummedTriangleArea() + from.getSummedTriangleArea();
			to.setSummedTriangleArea(combinedTriArea);
		}
	}
}



void ProgressiveMesh::updateTriangles(
		EdgeCollapse &ec, TriangleVertex &vc, set<int> &affectedVerts,
		LinkedTriangleMesh &mesh)
{
	set<int>& triNeighbors = vc.getTriangleNeighbors();
	set<int>::iterator pos;

	for (pos = triNeighbors.begin(); pos != triNeighbors.end(); ++pos)
	{
		// get triangle
		int triIndex = *pos;
		LinkedTriangle& t = mesh.getTriangle(triIndex);
		if (!t.isActive()) continue;
		bool bRemoveTri = false;
		if (t.hasVertex(ec.from) && t.hasVertex(ec.to))
		{
			ec.removed_triangles.insert(triIndex);
			t.changeVertex(ec.from, ec.to); // update the vertex of this triangle

			bRemoveTri = true;

			t.setActive(false); // triangle has two vertices which are now the same, so it's just a line segment
		}
		else
		{
			t.changeVertex(ec.from, ec.to); // update the vertex of this triangle
			t.calculateNormal(); // reset the normal for the triangle

			// make sure the "to" vertex knows about this triangle
			mesh.getVertex(ec.to).addTriangleNeighbor(triIndex);

			// If the triangle has an area effectively equal to 0, remove it.
			// NOTE: should this be done?  The triangle could get bigger through
			// another edge collapse in another direction.
			if (t.calculateArea() < 1e-6) {
				t.setActive(false);
				ec.removed_triangles.insert(triIndex);
				bRemoveTri = true;
			} else {
				ec.affected_triangles.insert(triIndex);
			}

		}
		// update set of affected verts
		// note that if you insert the same element twice,
		// it will only be stored once.
		int vert1, vert2, vert3;
		//t.getVertices(vert1, vert2, vert3);
		vert1 = t.getIndex(0);
		vert2 = t.getIndex(1);
		vert3 = t.getIndex(2);

		affectedVerts.insert(vert1);
		affectedVerts.insert(vert2);
		affectedVerts.insert(vert3);

		// If triangle is being removed, update each vertex which references it.
		if (bRemoveTri)
		{
			mesh.getVertex(vert1).removeTriangleNeighbor(triIndex);
			mesh.getVertex(vert2).removeTriangleNeighbor(triIndex);
			mesh.getVertex(vert3).removeTriangleNeighbor(triIndex);
		}
	}
}

void ProgressiveMesh::updateAffectedVertexNeighbors(
		TriangleVertex &vert, const EdgeCollapse &ec,
		const set<int> &affectedVerts)
{
	if (vert.getIndex() != ec.to)
	{
		vert.addVertexNeighbor(ec.to); // make sure vertex knows it has a new neighbor
	}
	else
	{
		set<int>::iterator mappos2;

		// Make sure the "to" vertex knows about its
		// new neighbors
		for (mappos2 = affectedVerts.begin(); mappos2 != affectedVerts.end(); ++mappos2)
		{
			if (*mappos2 != ec.to)
			{
				vert.addVertexNeighbor(*mappos2);
			}
		}
	}

	// get rid of deleted vertex
	vert.removeVertexNeighbor(ec.from);
}

void ProgressiveMesh::resetAffectedVertexCosts(
		const EdgeCost &cost,
		LinkedTriangleMesh &mesh, TriangleVertex &vert)
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

void ProgressiveMesh::removeVertexIfNecessary(
		TriangleVertex &vert, VertexPointerSet &vertSet,
		vector<VertexPointerSet::iterator> &vertSetVec,
		LinkedTriangleMesh &mesh, const EdgeCost &cost,
		set<int> &affectedQuadricVerts)
{
	bool bActiveVert = false;
	set<int>& mytriNeighbors = vert.getTriangleNeighbors();
	set<int>::iterator pos2;
	for (pos2 = mytriNeighbors.begin(); pos2 != mytriNeighbors.end(); ++pos2)
	{
		// get triangle
		int triIndex = *pos2;
		LinkedTriangle& t = mesh.getTriangle(triIndex);
		if (t.isActive()) {
			bActiveVert = true;
			break;
		}
	}

	if (bActiveVert) { // if vert is active
		VertexPointer vp;
		vp.index = vert.getIndex();
		vp.mesh = &mesh;
		vertSetVec[vp.index] = vertSet.insert(vp);

		assert(vertSetVec[vp.index]->index == vp.index);
		mesh.getVertex(vp.index).setActive(true);

		// If we're calculating quadric costs, keep track of
		// every active vertex which was affect by this collapse,
		// so we can recalculate collapse costs.
		if (QUADRIC == cost || QUADRICTRI == cost) {
			affectedQuadricVerts.insert(vert.getIndex());
		}
//#ifdef PRINT_DEBUG_INFO
//		std::cout << "\tvert affected: " << vert.getIndex() << std::endl;
//#endif
//	}
//	else {
//#ifdef PRINT_DEBUG_INFO
//		std::cout << "\tvert removed: " << vert.getIndex() << std::endl;
//#endif
		mesh.getVertex(vert.getIndex()).setActive(false);
	}
}

void ProgressiveMesh::updateAffectedVertices(
		LinkedTriangleMesh &mesh, vector<VertexPointerSet::iterator> &vertSetVec,
		VertexPointerSet &vertSet, const EdgeCollapse &ec,
		set<int> &affectedVerts, const EdgeCost &cost,
		set<int> &affectedQuadricVerts)
{
	VertexPointerSet::iterator del;
	set<int>::iterator mappos;

	for (mappos = affectedVerts.begin(); mappos != affectedVerts.end(); ++mappos)
	{
		//cout << "MAPPOS" << endl << flush;
		TriangleVertex& vert = mesh.getVertex(*mappos);
		assert(vert.getIndex() == *mappos);

		del = vertSetVec[*mappos];

//		if(del == vertSet.end()) cout << "ERRROR!" << endl << flush;

//		cout << "TEST" << endl << flush;
//		cout << (*del).index << " " << *mappos << endl << flush;
//		cout << (*del).mesh << " " << original_mesh << " " << &reduced_mesh << endl << flush;
//		cout << ((*del).mesh == 0) << endl << flush;
//		cout << "END TEST" << endl << flush;

		if((*del).index >= original_mesh->getNumberOfVertices() ||
				(*del).index < 0 || (*del).mesh != &reduced_mesh) continue;
		// Always erase, maybe add in.
		// Can't change in place, 'cause will screw up order of se
		vertSet.erase(*del);

		//cout << "invalidate" << endl << flush;
		vertSetVec[*mappos] = vertSet.end(); // set to "invalid" value

		//cout << "updateAffectedVertexNeighbors" << endl  << flush;
		updateAffectedVertexNeighbors(vert, ec, affectedVerts);

		// reset values for affected vertices
		//cout << "resetAffectedVertexCosts" << endl  << flush;
		resetAffectedVertexCosts(cost, mesh, vert);

		// Remove vertex if it's not attached to any active triangle
		//cout << "removeVertexIfNecessary" << endl << flush;
		removeVertexIfNecessary(vert, vertSet, vertSetVec, mesh,
								cost, affectedQuadricVerts);
	}
}

void ProgressiveMesh::recalcQuadricCollapseCosts(
		set<int> &affectedQuadricVerts,
		LinkedTriangleMesh &mesh, const EdgeCost &cost)
{
	if (QUADRIC == cost || QUADRICTRI == cost)
	{
		set<int>::iterator mappos;
		for (mappos = affectedQuadricVerts.begin(); mappos != affectedQuadricVerts.end(); ++mappos)
		{
			TriangleVertex& vert = mesh.getVertex(*mappos);
			quadricCollapseCost(mesh, vert);
		}
	}
}

void ProgressiveMesh::buildEdgeCollapseList(
		LinkedTriangleMesh &mesh, const EdgeCost &cost,
		list<EdgeCollapse> &edgeCollList,
		VertexPointerSet &vertSet,
		vector<VertexPointerSet::iterator> &vertSetVec)
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

		const VertexPointer vp = *(vertSet.begin()); // This is a copy of the first element
		TriangleVertex vc = mesh.getVertex(vp.index);
		assert(vp.index == vc.getIndex());

		EdgeCollapse ec; // create EdgeCollapse structure

		bool bBadVertex = false;

		// Make sure this edge collapse has a valid "to vertex"
		ensureEdgeCollapseIsValid(ec, vc, mesh, cost, bBadVertex);

		mesh.getVertex(ec.from).setActive(false);
		vertSet.erase(vertSet.begin());

		if (bBadVertex) {
			continue;
		}

#ifdef PRINT_DEBUG_INFO
		std::cout << "from: " << ec._vfrom << " to: " << ec._vto << std::endl;
#endif

		TriangleVertex& to = mesh.getVertex(ec.to);
		TriangleVertex& from = mesh.getVertex(ec.from);

		//cout << "setToVertexQuadric" << endl << flush;
		setToVertexQuadric(to, from, cost);

		set<int> affectedVerts;

		// We are removing a vertex and an edge.  Look at all triangles
		// which use this vertex.  Each of these triangles is either being
		// removed or updated with a new vertex.
		//cout << "updateTriangles" << endl << flush;
		updateTriangles(ec, vc, affectedVerts, mesh);

		set<int> affectedQuadricVerts;

		// These vertices were in triangles which either were removed or
		// were updated with new vertices.  Removed these vertices if they're
		// not connected to an active triangle.  Update these vertices if they're
		// still being displayed.
		//cout << "updateAffectedVertices" << endl << flush;
		updateAffectedVertices(mesh, vertSetVec, vertSet, ec, affectedVerts,
							   cost, affectedQuadricVerts);

		// If using the quadric collapse method,
		// recalculate the edge collapse costs for the affected vertices.
		// cout << "recalcQuadricCollapseCosts" << endl << flush;
		recalcQuadricCollapseCosts(affectedQuadricVerts, mesh, cost);

#ifdef PRINT_DEBUG_INFO
		std::cout << "---- Collapse # "<< count++ << " ----" << std::endl;
		mesh.dump();
		ec.dumpEdgeCollapse();
		dumpset(vertSet);
#endif
		cout << "PUSH BACK" << endl << flush;
		edgeCollList.push_back(ec); // inserts a copy

	}
}


void ProgressiveMesh::createEdgeCollapseList()
{
	// okay, get list of verts, tris
	// for each vert, calc cost
	// add to edge collapse list

	// Copy the original mesh
	reduced_mesh = *original_mesh;

	edge_collapse_list.clear(); // empty list

	int nVerts = reduced_mesh.getNumberOfVertices();
	int nTri = reduced_mesh.getNumberOfTriangles();

	//cout << "**************************** nVerts: " << nVerts << endl;

	number_of_visited_triangles = nTri; // number of visible triangles

	// calculate all 4x4 Q matrices for each vertex
	// if using the Quadric method
	calcQuadricMatrices(edge_cost, reduced_mesh);


	// This is a set of vertex pointers, ordered by edge collapse cost.
	VertexPointerSet vertSet;
	vertSet.clear();
	vector<VertexPointerSet::iterator> vertSetVec(nVerts);

	// Go through, calc cost here for all vertices
	calcEdgeCollapseCosts(vertSet, vertSetVec, nVerts, reduced_mesh, edge_cost);

	// For all vertices:
	//	find lowest cost
	//	store the edge collapse structure
	//	update all verts, triangles affected by the edge collapse
	buildEdgeCollapseList(reduced_mesh, edge_cost, edge_collapse_list,
							vertSet, vertSetVec);


	reduced_mesh = *original_mesh;
	for (int i = 0; i < nTri; ++i)
	{
		reduced_mesh.getTriangle(i).setActive(true);
	}

	// set iterator to point to beginning
	edge_collapse_it = edge_collapse_list.begin();
}


void ProgressiveMesh::calcAllQuadricMatrices(LinkedTriangleMesh& mesh, bool bUseTriArea)
{
	set<Border> borderSet;

	int nVerts = mesh.getNumberOfVertices();

	for (int i = 0; i < nVerts; ++i)
	{
		TriangleVertex& currVert = mesh.getVertex(i);

		currVert.calcQuadric(mesh, bUseTriArea);

		float myQ[4][4];
		currVert.getQuadric(myQ);
		float triArea = 0;

		if (QUADRICTRI == edge_cost)
		{
			triArea = currVert.getSummedTriangleArea();
		}

		calcQuadricError(myQ, currVert, triArea);

		// Is the current vertex on a border?  If so, get the
		// edge information
		currVert.getBorderEdges(borderSet, mesh);
	}

	// Keep the mesh borders from being "eaten away".
	if (!borderSet.empty())
	{
		applyBorderPenalties(borderSet, mesh);
	}
}


void ProgressiveMesh::applyBorderPenalties(
		set<Border> &borderSet, LinkedTriangleMesh &mesh)
{
	set<Border>::iterator pos;

	for (pos = borderSet.begin(); pos != borderSet.end(); ++pos)
	{
		// First, determine the plane equation of plane perpendicular
		// to the edge triangle.

		Border edgeInfo = *pos;

		TriangleVertex& v1 = mesh.getVertex(edgeInfo.vert1);
		TriangleVertex& v2 = mesh.getVertex(edgeInfo.vert2);

		//Vec3 &vec1 = v1.getXYZ();
		//Vec3 &vec2 = v2.getXYZ();

		Vertex vec1 = v1.getPosition();
		Vertex vec2 = v2.getPosition();
		Vertex edge = vec1 - vec2;

		LinkedTriangle &triangle = mesh.getTriangle(edgeInfo.triIndex);
		Normal normal = triangle.getNormal();

		Vertex abc = edge.cross(normal);

		float &a = abc.x;
		float &b = abc.y;
		float &c = abc.z;

		float d = -(abc * vec1);


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
		float Q1[4][4], Q2[4][4];
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

float ProgressiveMesh::shortEdgeCollapseCost(LinkedTriangleMesh& m, TriangleVertex& v)
{
	// get list of all active neighbors
	// calculate shortest edge
	// what if no neighbors??
	// return cost
	float mincost = FLT_MAX; // from float.h
	bool bNeighborFound = false;

	set<int>& neighbors = v.getVertexNeighbors();
	set<int>::iterator pos;
	for (pos = neighbors.begin(); pos != neighbors.end(); ++pos)
	{
		TriangleVertex& n = m.getVertex(*pos);
		if (!n.isActive()) continue;
		if (n == v) continue;

		// calc cost
		Vertex s = v.position - n.position;
		float cost = s.length();

		if (cost < mincost)
		{
			bNeighborFound = true;
			mincost = cost;
			v.setEdgeCost(cost);
			v.setMinCostNeighbor(*pos);
			//assert(v.minCostEdgeVert() >= 0 && v.minCostEdgeVert() < m.getNumVerts());
		}
	}

	if (bNeighborFound) {
		return mincost;
	} else {
		return FLT_MAX; // vertex not connected to an edge
	}
}


void ProgressiveMesh::calcMelaxMaxValue(
		LinkedTriangleMesh &mesh, set<int> &adjfaces,
		TriangleVertex &v, set<int> &tneighbors,
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
			LinkedTriangle& t = mesh.getTriangle(triIndex);
			if (!t.isActive()) continue;

			bMinValueFound = false;
			set<int>::iterator pos3;
			for (pos3 = adjfaces.begin(); pos3 != adjfaces.end(); ++pos3)
			{
				int triIndex3 = *pos3;
				LinkedTriangle& t3 = mesh.getTriangle(triIndex3);
				if (!t3.isActive()) continue;

				Normal n  = t.getNormal();
				Normal n2 = t3.getNormal();

				float dot = n * n2;

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

float ProgressiveMesh::melaxCollapseCost(
		LinkedTriangleMesh& mesh, TriangleVertex& v)
{
	set<int>& vneighbors = v.getVertexNeighbors();
	set<int>& tneighbors = v.getTriangleNeighbors();
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
			LinkedTriangle& t = mesh.getTriangle(triIndex);
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
			Vertex v1 = v.position;
			Vertex v2 = mesh.getVertex(*pos).position;
			Vertex v3 = v1 - v2;

			retmaxValue *= v3.length();

			if (retmaxValue < mincost)
			{
				mincost = retmaxValue;
				v.setEdgeCost(retmaxValue);
				v.setMinCostNeighbor(*pos);
			}
		}
	}
	return mincost;
}

float ProgressiveMesh::quadricCollapseCost(LinkedTriangleMesh& m, TriangleVertex& v)
{
	// get list of all active neighbors
	// calculate quadric cost
	float mincost = FLT_MAX; // from float.h
	bool bNeighborFound = false;

	float Q1[4][4];
	v.getQuadric(Q1);

	set<int>& neighbors = v.getVertexNeighbors();
	set<int>::iterator pos;

	for (pos = neighbors.begin(); pos != neighbors.end(); ++pos)
	{
		TriangleVertex& n = m.getVertex(*pos);
		if (!n.isActive()) continue;
		if (n == v) continue;

		float Q2[4][4];
		float Qsum[4][4];

		// add two 4x4 Q matrices
		n.getQuadric(Q2);

		for(int i = 0; i < 4; ++i) {
			for ( int j = 0; j < 4; ++j) {
				Qsum[i][j] = Q1[i][j] + Q2[i][j];
			}
		}

		double triArea = 0;
		if (QUADRICTRI == edge_cost)
		{
			triArea = v.getSummedTriangleArea() + n.getSummedTriangleArea();
		}

		// calc cost
		double cost = calcQuadricError(Qsum, n, triArea);

		if (cost < mincost)
		{
			bNeighborFound = true;
			mincost = cost;
			v.setEdgeCost(cost);
			v.setMinCostNeighbor(*pos);
			//assert(v.minEdge() >= 0 && v.minCostEdgeVert() < m.getNumVerts());
		}
	}

	if (bNeighborFound) {
		return mincost;
	} else {
		return FLT_MAX; // vertex not connected to an edge
	}
}

float ProgressiveMesh::calcQuadricError(
		float Qsum[4][4],
		TriangleVertex& v, float triArea)
{
	float cost;

	// 1st, consider vertex v a 1x4 matrix: [v.x v.y v.z 1]
	// Multiply it by the Qsum 4x4 matrix, resulting in a 1x4 matrix

	float result[4];

	//const Vec3 v3 = v.getXYZ();
	Vertex v3 = v.position;

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

	if (QUADRICTRI == edge_cost && triArea != 0)
	{
		cost /= triArea;
	}

	return cost;
}


void ProgressiveMesh::collapseEdge()
{
	// Iterator always points to next collapse to perform
	if (edge_collapse_it == edge_collapse_list.end()) return; // no more edge collapses in list
	EdgeCollapse& ec = *edge_collapse_it;

	set<int> affectedVerts; // vertices affected by this edge collapse
	int v1, v2, v3; // vertex indices

	// Remove triangles
	set<int>::iterator tripos;
	for (tripos = ec.removed_triangles.begin(); tripos != ec.removed_triangles.end(); ++tripos)
	{
		// get triangle
		int triIndex = *tripos;
		LinkedTriangle & t = reduced_mesh.getTriangle(triIndex);
		//t.getVerts(v1, v2, v3); // get triangle vertices

		v1 = t.getIndex(0);
		v2 = t.getIndex(1);
		v3 = t.getIndex(2);

		t.setActive(false);
		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// Adjust vertices of triangles
	for (tripos = ec.affected_triangles.begin(); tripos != ec.affected_triangles.end(); ++tripos)
	{
		// get triangle
		int triIndex = *tripos;
		LinkedTriangle& t = reduced_mesh.getTriangle(triIndex);
		t.changeVertex(ec.from, ec.to); // update the vertex of this triangle
		t.calculateNormal(); // reset the normal for the triangle

		v1 = t.getIndex(0);
		v2 = t.getIndex(1);
		v3 = t.getIndex(2);

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
		if (ec.from == *affectedVertsIter) continue; // skip the from vertex -- it's no longer active

		// We have the affected vertex index, so redo the its normal (for Gouraud shading);
		reduced_mesh.calcOneNormal(*affectedVertsIter);
	}

	// Since iterator always points to next collapse to perform, go to the next
	// collapse in list.
	++edge_collapse_it;

	number_of_visited_triangles -=  ec.removed_triangles.size();

}

void ProgressiveMesh::splitVertex()
{
	// Iterator always points to next collapse to perform.
	// But we don't want to collapse, we want to undo the previous
	// collapse.  Go to that edge collapse, unless we're at the front of
	// the list, in which case there are no collapses to undo (the mesh
	// is fully displayed w/o any collapses).
	if (edge_collapse_it == edge_collapse_list.begin()) return;
	--edge_collapse_it; // go to previous edge collapse, so we can undo it
	EdgeCollapse& ec = *edge_collapse_it;

	set<int> affectedVerts; // vertices affected by this edge collapse
	int v1, v2, v3; // vertex indices

	// Add triangles which were removed
	set<int>::iterator tripos;
	for (tripos = ec.removed_triangles.begin(); tripos != ec.removed_triangles.end(); ++tripos)
	{
		// get triangle
		int triIndex = *tripos;
		LinkedTriangle& t = reduced_mesh.getTriangle(triIndex);
		t.setActive(true);
		//t.getVerts(v1, v2, v3); // get triangle vertices

		v1 = t.getIndex(0);
		v2 = t.getIndex(1);
		v3 = t.getIndex(2);

		affectedVerts.insert(v1); // add vertices to list
		affectedVerts.insert(v2); // of vertices affected
		affectedVerts.insert(v3); // by this collapse
	}

	// Adjust vertices of triangles
	for (tripos = ec.affected_triangles.begin(); tripos != ec.affected_triangles.end(); ++tripos)
	{
		// get triangle
		int triIndex = *tripos;
		LinkedTriangle& t = reduced_mesh.getTriangle(triIndex);
		t.changeVertex(ec.to, ec.from); // update the vertex of this triangle
		t.calculateNormal(); // reset the normal for the triangle

		v1 = t.getIndex(0);
		v2 = t.getIndex(1);
		v3 = t.getIndex(2);

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
		reduced_mesh.calcOneNormal(*affectedVertsIter);
	}

	number_of_visited_triangles +=  ec.removed_triangles.size();

}

void ProgressiveMesh::simplify(float reduction){

	int number_of_collapses = edge_collapse_list.size() - reduction * edge_collapse_list.size();

	cout << "NUMBER OF COLLAPSES: " << number_of_collapses << " " << edge_collapse_list.size() << endl;

	for(int i = 0; i < number_of_collapses; i++){
		collapseEdge();
	}

}



