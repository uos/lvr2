// ******************************
// vertex.cpp
//
// Mesh vertex has both a position and a normal
//
// Jeff Somers
// Copyright (c) 2002
//
// jsomers@alumni.williams.edu
// March 27, 2002
// ******************************

#if defined (_MSC_VER) && (_MSC_VER >= 1020)
#pragma warning(disable:4710) // function not inlined
#pragma warning(disable:4702) // unreachable code
#pragma warning(disable:4514) // unreferenced inline function has been removed
#endif

#include "vertex.h"
#include "cubemesh.h"

// Friend functions for input and output

std::ostream&
operator<<(std::ostream& os, const vertex& vo)

{
//	return (os << "<" << vo._myVertex.x << ", " << vo._myVertex.y << ", " << vo._myVertex.z << ">");
	os << " Index: " << vo.getIndex() << " ";
	os << vo.getXYZ(); // for some reason this isn't working as a friend function, not sure why
							// it is pulling ostream from the STL typedef, not the regular ostream, though.
	set<int>::iterator pos;
	os << " Vert Neighbors:";
	for (pos = vo.getVertNeighbors().begin(); pos != vo.getVertNeighbors().end(); ++pos)
	{
		os << " " << *pos;
	}
	os << " Tri Neighbors:";
	for (pos = vo.getTriNeighbors().begin(); pos != vo.getTriNeighbors().end(); ++pos)
	{
		os << " " << *pos;
	}

	os << " Is Active: " << vo.isActive();
	os << " Cost: " << vo.getCost();
	os << " Min Vert: " << vo.minCostEdgeVert();
	return os;
}

// Calculate the Quadric 4x4 matrix
void 
vertex::calcQuadric(CubeMesh& m, bool bUseTriArea)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			_Q[i][j] = 0;
		}
	}

	set<int>::iterator pos;
	for (pos = _triNeighbors.begin(); pos != _triNeighbors.end(); ++pos)
	{
		int triIndex = *pos;
		triangle& t = m.getTri(triIndex);
		if (t.isActive()) 
		{
			float triArea = 1;
			if (bUseTriArea)
			{
				triArea = t.calcArea();
				_QTriArea += triArea;
			}

			const Vec3 normal = t.getNormalVec3();
			const float a = normal.x;
			const float b = normal.y;
			const float c = normal.z;
			const float d = t.getD();

			// NOTE: we could optimize this a bit by calculating values
			// like a * b and then using that twice (for _Q[0][1] and _Q[1][0]),
			// etc., since the matrix is symmetrical.  For now, I don't think
			// it's worth it.
			_Q[0][0] += triArea * a * a;
			_Q[0][1] += triArea * a * b;
			_Q[0][2] += triArea * a * c;
			_Q[0][3] += triArea * a * d;

			_Q[1][0] += triArea * b * a;
			_Q[1][1] += triArea * b * b;
			_Q[1][2] += triArea * b * c;
			_Q[1][3] += triArea * b * d;

			_Q[2][0] += triArea * c * a;
			_Q[2][1] += triArea * c * b;
			_Q[2][2] += triArea * c * c;
			_Q[2][3] += triArea * c * d;

			_Q[3][0] += triArea * d * a;
			_Q[3][1] += triArea * d * b;
			_Q[3][2] += triArea * d * c;
			_Q[3][3] += triArea * d * d;
		}
	}
}

// So go through the list of all neighboring vertices, and see how many
// triangles this vertex has in common w/ each neighboring vertex.  Normally
// there will be two triangles in common, but if there is only one, then this 
// vertex is on an edge.
bool vertex::isBorder(CubeMesh& m)
{
	set<int>::iterator pos, pos2;
	for (pos = getVertNeighbors().begin(); pos != getVertNeighbors().end(); ++pos)
	{
		int triCount = 0;

		vertex& v = m.getVertex(*pos);

		for (pos2 = v.getTriNeighbors().begin(); pos2 != v.getTriNeighbors().end(); ++pos2)
		{
			if (m.getTri(*pos2).hasVertex(_index) )
			{
				++triCount;
			}
		}

		if (1 == triCount)
		{
			return true;
		}
	}

	return false;
}

// Return all border info if the vertex is on an edge of the mesh.
void vertex::getAllBorderEdges(set<border> &borderSet, CubeMesh& m)
{
	// So go through the list of all neighboring vertices, and see how many
	// triangles this vertex has in common w/ each neighboring vertex.  Normally
	// there will be two triangles in common, but if there is only one, then this 
	// vertex is on an edge.
	set<int>::iterator pos, pos2;

	for (pos = getVertNeighbors().begin(); pos != getVertNeighbors().end(); ++pos)
	{
		int triCount = 0;
		int triIndex = -1;
		vertex& v = m.getVertex(*pos);
		for (pos2 = v.getTriNeighbors().begin(); pos2 != v.getTriNeighbors().end(); ++pos2)
		{
			if (m.getTri(*pos2).hasVertex(_index) )
			{
				++triCount;
				triIndex = m.getTri(*pos2).getIndex();
			}
		}

		if (1 == triCount) // if only one triangle in common, it's an edge
		{
			// store the smaller index first
			border b;
			b.triIndex = triIndex;
			if (_index < v.getIndex())
			{
				b.vert1 = _index;
				b.vert2 = v.getIndex();
			}
			else
			{
				b.vert1 = v.getIndex();
				b.vert2 = _index;
			}
			borderSet.insert(b);
		}
	}
}

/**** Not implemented
istream& operator>>(istream &io, vertex &vi)
{
	char inBuf[Vec3::MAX_INPUT_STRING]; //!FIX this
	io >> inBuf; // operator>>(ostream&, char*); -- or is it istream?
	//!FIX need to convert string to vector here
//	vi = inBuf;	// String::operator=(const char*)
	return io;
}

*/

