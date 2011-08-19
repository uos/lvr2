#include "Region.hpp"

namespace lssr
{

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::Region(vector<HalfEdgeFace<VertexT, NormalT>*>    faces)
{
	this->m_faces = faces;
}

template<typename VertexT, typename NormalT>
vector<stack<HVertex*> > Region<VertexT, NormalT>::getContours(float epsilon)
{
	vector<stack<HVertex*> > result;
	//TODO: implement
	return result;
}

template<typename VertexT, typename NormalT>
NormalT Region<VertexT, NormalT>::getNormal()
{
	NormalT result;
	//TODO: implement
	return result;
}


}
