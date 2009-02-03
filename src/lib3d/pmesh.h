// ******************************
// pmesh.h
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

#ifndef __PMesh_h
#define __PMesh_h

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#pragma warning(disable:4786) /* disable "identifier was truncated to '255' characters in the browser information" warning in Visual C++ 6*/
#endif

//#define WIN32_LEAN_AND_MEAN
//#include <windows.h>

using namespace std;

#include <vector>
#include <list>
#include "vertex.h"
#include "triangle.h"
#include "cubemesh.h"


// The edge collapse structure.  The "from vertex" will
// be collapsed to the "to vertex."  This may flatten some
// triangles, which will be removed, and will affect those
// triangles which contain the "from vertex".  Those triangles
// will be updated with the new vertex.
struct EdgeCollapse
{
	int _vfrom;
	int _vto;
	set<int> _trisRemoved;
	set<int> _trisAffected;


	// Used for debugging
	void dumpEdgeCollapse()
	{
		std::cout << "**** Edge Collapse Dump ****" << std::endl;

		std::cout << "\tFrom Vert# " << _vfrom << " to Vert# " << _vto << std::endl;
		cout << "\tTris removed:";
		set<int>::iterator pos;
		for (pos = _trisRemoved.begin(); pos != _trisRemoved.end(); ++pos) 
		{
			std::cout << " " << *pos;
		}
		cout << std::endl << "\tTris affected:";
		for (pos = _trisAffected.begin(); pos != _trisAffected.end(); ++pos) 
		{
			std::cout << " " << *pos;
		}
		std::cout  << std::endl << "**** End of Edge Collapse Dump ****" << std::endl;
	}
};

// This is a "pointer" to a vertex in a given mesh
struct vertexPtr
{
	CubeMesh* _meshptr;
	int _index; // ptr to vertex position in mesh

	bool operator<(const vertexPtr& vp) const 
	{
		return (_meshptr->getVertex(_index) < vp._meshptr->getVertex(vp._index));
	}
};


typedef multiset<vertexPtr, less<vertexPtr> > vertexPtrSet;


// Progressive Mesh class.  This class will calculate and keep track
// of which vertices and triangles should be removed from/added to the
// mesh as it's simplified (or restored).
class PMesh
{
public:
	// Type of progress mesh algorithm
	enum EdgeCost {SHORTEST, MELAX, QUADRIC, QUADRICTRI, MAX_EDGECOST};

	PMesh(CubeMesh* mesh, EdgeCost ec);

	// Collapse one vertex to another.
	bool collapseEdge();

	// One vertex will be split into two vertices -- this
	// is the opposite of a collapse
	bool splitVertex();

	// number of edge collapses
	int numCollapses() {return _edgeCollList.size();}
	int numEdgeCollapses() {return _edgeCollList.size();}

	// number of triangles, and visible triangles in mesh
	int numTris() {return _newmesh.getNumTriangles();}
	int numVisTris() {return _nVisTriangles;}
  //bool toDXF(char* filename);
	bool getTri(int i, triangle& t) {
		t = _newmesh.getTri(i);
		return true;
	}
 
	// Return a short text description of the current Edge Cost method
	char* getEdgeCostDesc();

private:

	CubeMesh* _mesh; // original mesh - not changed
	CubeMesh _newmesh; // we change this one

	EdgeCost _cost; // Type of progressive mesh algorithm

	list<EdgeCollapse> _edgeCollList; // list of edge collapses
	list<EdgeCollapse>::iterator _edgeCollapseIter;

	// functions used to calculate edge collapse costs.  Different
	// methods can be used, depending on user preference.
	double shortEdgeCollapseCost(CubeMesh& m, vertex& v);
	double melaxCollapseCost(CubeMesh& m, vertex& v);
	double quadricCollapseCost(CubeMesh& m, vertex& v);

	int _nVisTriangles; // # of triangles, after we collapse edges

	// Create the list of the edge collapses used
	// to simplify the mesh.
	void createEdgeCollapseList();

	// Used in the QEM edge collapse methods.
	void calcAllQMatrices(CubeMesh& mesh, bool bUseTriArea); // used for quadric method
	double calcQuadricError(double Qsum[4][4], vertex& v, double triArea); // used for quadric method

	enum {BOUNDARY_WEIGHT = 1000}; // used to weight border edges so they don't collapse
	void applyBorderPenalties(set<border> &borderSet, CubeMesh &mesh);

	PMesh(const PMesh&); // don't allow copy ctor -- too expensive
	PMesh& operator=(const PMesh&); // don't allow assignment op.
	bool operator==(const PMesh&); // don't allow op==

#ifndef NDEBUG
	// used in debugging
	void assertEveryVertActive(int nVerts, int nTri, CubeMesh &mesh);
#endif
	// helper function for edge collapse costs
	void calcEdgeCollapseCosts(vertexPtrSet &vertSet, vector<vertexPtrSet::iterator> &vertSetVec, 
								  int nVerts, CubeMesh &mesh, EdgeCost &cost);

	// Calculate the QEM matrices used to computer edge
	// collapse costs.
	void calcQuadricMatrices(EdgeCost &cost, CubeMesh &mesh);

	// We can't collapse Vertex1 to Vertex2 if Vertex2 is invalid.
	// This can happen if Vertex2 was previously collapsed to a
	// separate vertex.
	void insureEdgeCollapseValid(EdgeCollapse &ec, vertex &vc, CubeMesh &mesh, 
									const EdgeCost &cost, bool &bBadVertex);

	// Calculate the QEM for the "to vertex" in the edge collapse.
	void setToVertexQuadric(vertex &to, vertex &from, const EdgeCost &cost);

	// At this point, we have an edge collapse.  We're collapsing the "from vertex"
	// to the "to vertex."  For all the surrounding triangles which use this edge, 
	// update "from vertex" to the "to vertex".  Also keep track of the vertices
	// in the surrounding triangles. 
	void updateTriangles(EdgeCollapse &ec, vertex &vc, set<int> &affectedVerts, CubeMesh &mesh);


	// These affected vertices are not in the current collapse, 
	// but are in the triangles which share the collapsed edge.
	void updateAffectedVertNeighbors(vertex &vert, const EdgeCollapse &ec, 
		const set<int> &affectedVerts);

	// Reset the edge collapse costs of vertices which were
	// affected by a previous edge collapse.
	void resetAffectedVertCosts(const EdgeCost &cost, CubeMesh &newmesh, vertex &vert);

	// If this vertex has no active triangles (i.e. triangles which have
	// not been removed from the mesh) then set it to inactive.
	void removeVertIfNecessary(vertex &vert, vertexPtrSet &vertSet, 
								  vector<vertexPtrSet::iterator> &vertSetVec, 
								  CubeMesh &mesh, const EdgeCost &cost, 
									set<int> &affectedQuadricVerts);

	// Update the vertices affected by the most recent edge collapse
	void updateAffectedVerts(CubeMesh &_newmesh, vector<vertexPtrSet::iterator> &vertSetVec, 
							vertexPtrSet &vertSet, const EdgeCollapse &ec, 
							set<int> &affectedVerts, const EdgeCost &cost, 
							set<int> &affectedQuadricVerts);

	// Recalculate the QEM matrices (yeah, that's redundant) if we're
	// using the Quadrics to calculate edge collapse costs.
	void recalcQuadricCollapseCosts(set<int> &affectedQuadricVerts, 
								   CubeMesh &mesh, const EdgeCost &cost);

	// Calculate the list of edge collapses.  Each edge collapse
	// consists of two vertices:  a "from vertex" and a "to vertex".
	// The "from vertex" is collapsed to the "to vertex".  The
	// "from vertex" is removed from the mesh.
	void buildEdgeCollapseList(CubeMesh &mesh, const EdgeCost &cost, 
							  list<EdgeCollapse> &_edgeCollList,
								vertexPtrSet &vertSet, 
								vector<vertexPtrSet::iterator> &vertSetVec);

	// Helper function for melaxCollapseCost().  This function
	// will loop through all the triangles to which this vertex
	// belongs.
	void calcMelaxMaxValue(CubeMesh &mesh, set<int> &adjfaces, 
							  vertex &v, set<int> &tneighbors,
								float &retmaxValue, 
								bool &bMaxValueFound);
};

#endif // __PMesh_h
