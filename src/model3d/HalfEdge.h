#ifndef __HALF_EDGE_H__
#define __HALF_EDGE_H__

class HalfEdgeVertex;
class HalfEdgeFace;

class HalfEdge{
public:
	HalfEdge();
	~HalfEdge();

	HalfEdge* next;
	HalfEdge* pair;

	HalfEdgeVertex* start;
	HalfEdgeVertex* end;

	HalfEdgeFace* face;

	bool used;
};

#endif
