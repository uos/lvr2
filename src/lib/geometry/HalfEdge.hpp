#ifndef __HALF_EDGE_H__
#define __HALF_EDGE_H__

namespace lssr
{

template<typename VertexT, typename FaceT>
class HalfEdge{
public:
	HalfEdge() : start(0), end(0), pair(0), next(0), face(0), used(false) {};

	~HalfEdge()
	{
		delete next;
		delete pair;
	}

	HalfEdge<VertexT, FaceT> next;
	HalfEdge<VertexT, FaceT>* pair;

	VertexT* start;
	VertexT* end;

	FaceT* face;

	bool used;
};

} // namespace lssr

#endif
