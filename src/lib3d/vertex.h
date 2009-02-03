// ******************************
// vertex.h
//
// Mesh vertex has both a position and a normal
//
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#ifndef __vertex_h
#define __vertex_h

#if defined (_MSC_VER) && (_MSC_VER >= 1020)

#pragma once
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#endif

using namespace std;

#include <vector>
#include <set>

#include "vec3.h"
#include "triangle.h"


class CubeMesh;

// Used to store an edge -- two vertices which have only one
// triangle in common form an edge of the mesh.
struct border
{
	int vert1;
	int vert2;
	int triIndex;

	// We need operator< because it's used by the STL's set<> to determine equality
	// (if (not a<b) and (not b>a) then a is equal to b)
	bool operator<(const border& b) const 
	{
		int v1, v2, b1, b2;
		// make sure the smaller vert index is always first.
		if (vert1 < vert2) {
			v1 = vert1; v2 = vert2;
		} else {
			v1 = vert2; v2 = vert1;
		}
		if (b.vert1 < b.vert2) {
			b1 = b.vert1; b2 = b.vert2;
		} else {
			b1 = b.vert2; b2 = b.vert1;
		}
		if (v1 < b1) return true;
		if (v1 > b1) return false;
		return (v2 < b2); // v1 == b1
	}
};

// A vertex has both a position and a normal
class vertex
{
public:
	// Constructors and Destructors
	vertex() : 
		_myVertex(0.0, 0.0, 0.0), _vertexNormal(0.0, 0.0, 0.0), 
		_bActive(false), _cost(0), _minCostNeighbor(-1),
		_index(-1), _QTriArea(0)
	{
		initQuadric();
	};

	vertex(float x1, float y1, float z1) : 
		_myVertex(x1, y1, z1), _vertexNormal(0.0, 0.0, 0.0), 
		_bActive(true), _cost(0), _minCostNeighbor(-1),
		_index(-1), _QTriArea(0)
	{
		initQuadric();
	};

	vertex(float av[3]): 
		_myVertex(av), _vertexNormal(0.0, 0.0, 0.0), 
		_bActive(true), _cost(0), _minCostNeighbor(-1),
		_index(-1), _QTriArea(0)
	{
		initQuadric();
	};

	vertex(float av[3], float vn[3]): 
		_myVertex(av), _vertexNormal(vn), 
		_bActive(true),_cost(0), _minCostNeighbor(-1),
		_index(-1), _QTriArea(0)
	{
		initQuadric();
	};

	// copy ctor
	vertex(const vertex& v) : _myVertex(v._myVertex), _vertexNormal(v._vertexNormal),
							_vertNeighbors(v._vertNeighbors), _triNeighbors(v._triNeighbors),
							_bActive(v._bActive), _cost(v._cost), 
							_minCostNeighbor(v._minCostNeighbor),
							_index(v._index), _QTriArea(v._QTriArea)
	{
		// copy quadric
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				_Q[i][j] = v._Q[i][j];
			}
		}		
	};

	// destructor
	~vertex() {_vertNeighbors.erase(_vertNeighbors.begin(), _vertNeighbors.end());
				_triNeighbors.erase(_triNeighbors.begin(), _triNeighbors.end());};

	// Assignment operator
	vertex& operator=(const vertex& v) 
	{
		if (this == &v) return *this; // check for assignment to self
		_myVertex =v._myVertex;
		_vertexNormal = v._vertexNormal; 
		_vertNeighbors = v._vertNeighbors;
		_triNeighbors = v._triNeighbors;
		_bActive = v._bActive;
		_cost = v._cost;
		_minCostNeighbor = v._minCostNeighbor;
		_index = v._index;
		_QTriArea = v._QTriArea;
		// copy quadric
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				_Q[i][j] = v._Q[i][j];
			}
		}
		return *this;
	};
	// Assignment operator
	vertex& operator=(const float av[3])
	{
		_myVertex.x=av[0];_myVertex.y=av[1];_myVertex.z=av[2];
		// erase the list of neighboring vertices, faces
		// since we're copying from an array of floats
		_vertNeighbors.erase(_vertNeighbors.begin(), _vertNeighbors.end());
		_triNeighbors.erase(_triNeighbors.begin(), _triNeighbors.end());
		_cost = 0;
		_minCostNeighbor = -1;
		_index = -1;
		_QTriArea = 0;
		initQuadric();
		return *this;
	};

	// Comparision operators
	bool operator==(const vertex& v) {return (_myVertex == v._myVertex && _vertexNormal == v._vertexNormal);};
	bool operator!=(const vertex& v) {return (_myVertex != v._myVertex || _vertexNormal != v._vertexNormal);};

	// Input and Output
	friend std::ostream&	operator<<(std::ostream& , const vertex& );

//	friend istream&
//	operator>>(istream& is, vertex& vi);

	// NOTE: a better solution would be to return a reference
	const float* getArrayVerts() const {
		_v[0]=_myVertex.x;
		_v[1]=_myVertex.y;
		_v[2]=_myVertex.z;
		return _v;}

	const float* getArrayVertNorms() const {
		_vn[0]=_vertexNormal.x;
		_vn[1]=_vertexNormal.y;
		_vn[2]=_vertexNormal.z;
		return _vn;}

	// a vertex neighbor is connected by an edge
	void addVertNeighbor(int v)
	{
		_vertNeighbors.insert(v);
	}

	// remove a vertex which is no longer connected by an edge
	unsigned removeVertNeighbor(int v) {return _vertNeighbors.erase(v);}

	// a triangle neighbor is a triangle which uses this vertex
	void addTriNeighbor(int t)
	{
		_triNeighbors.insert(t);
	}

	// remove triangle if it no longer uses this vertex
	unsigned  removeTriNeighbor(int t) {return _triNeighbors.erase(t);}

	Vec3& getXYZ() {return _myVertex;};
	const Vec3& getXYZ() const {return _myVertex;};

	// if a vertex is removed, we set a flag
	bool isActive() const {return _bActive;};
	void setActive(bool b) {_bActive = b;};

	const set<int>& getVertNeighbors() const {return _vertNeighbors;}
	set<int>& getVertNeighbors() {return _vertNeighbors;}
	const set<int>& getTriNeighbors() const {return _triNeighbors;}
	set<int>& getTriNeighbors() {return _triNeighbors;}

	bool hasVertNeighbor(int v) const {return (_vertNeighbors.find(v) != _vertNeighbors.end());}
	bool hasTriNeighbor(int t) const {return (_triNeighbors.find(t) != _triNeighbors.end());}

	// edge remove costs are used in mesh simplification
	double edgeRemoveCost() {return _cost;};
	void setEdgeRemoveCost(double f) {_cost = f;};

	int minCostEdgeVert() const {return _minCostNeighbor;};
	void setMinCostEdgeVert(int i) {_minCostNeighbor = i;}

	double getCost() const {return _cost;}

	// operator< & operator> are used to order vertices by edge removal costs
	bool operator<(const vertex& v) const {return (_cost < v._cost);}
	bool operator>(const vertex& v) const {return (_cost > v._cost);}

	int getIndex() const {return _index;}
	void setIndex(int i) {_index = i;}

	// Used for Garland & Heckbert's quadric edge collapse cost (used for mesh simplifications/progressive meshes)
	void calcQuadric(CubeMesh& m, bool bUseTriArea); // calculate the 4x4 Quadric matrix

	void getQuadric(double Qret[4][4]) 
	{
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				Qret[i][j] = _Q[i][j];
			}
		}
	}

	void setQuadric(double Qnew[4][4]) 
	{
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				 _Q[i][j] = Qnew[i][j];
			}
		}
	}


	bool isBorder(CubeMesh& m); // is this vertex on the border 
							// (i.e. is there an edge which uses this vertex which 
							// is only used for one triangle?)

	// Used for Gouraud shading
	void setVertNomal(const Vec3& vn) {_vertexNormal = vn;};

	double getQuadricSummedTriArea() {return _QTriArea;};
	void setQuadricSummedTriArea(double newArea) {_QTriArea = newArea;};

	// Is the current vertex on an edge?  If so, get edge information.
	// This is used to put constraints on the border so that the mesh
	// edges aren't "eaten away" by the mesh simplification.
	void getAllBorderEdges(set<border> &borderSet, CubeMesh& m);

	Vec3 _myVertex; // X, Y, Z position of this vertex
	Vec3 _vertexNormal; // vertex normal, used for Gouraud shading
     bool closed;  //used for closing holes
private:
	set<int> _vertNeighbors; // connected to this vertex via an edge
	set<int> _triNeighbors; // triangles of which this vertex is a part

	bool _bActive; // false if vertex has been removed

	double _cost; // cost of removing this vertex from Progressive Mesh
	int _minCostNeighbor; // index of vertex at other end of the min. cost edge

	int _index;

	mutable float _v[3];
	mutable float _vn[3];

	double _Q[4][4]; // Used for Quadric error cost.

	double _QTriArea; // summed area of triangles used to computer quadrics

	// Used for Garland & Heckbert's quadric edge collapse cost (used for mesh simplifications/progressive meshes)
	void initQuadric()
	{
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				_Q[i][j] = -1;
			}
		}
	}
};

#endif // #ifndef __vertex_h
