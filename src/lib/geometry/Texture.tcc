/*
 * Texture.tcc
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

namespace lssr {

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::Texture(PointCloudManager<VertexT, NormalT>* pm, Region<VertexT, NormalT>* region)
{
	vector<HVertex*> HOuter_contour = region->getContours(0.01)[0];



}

template<typename VertexT, typename NormalT>
void Texture<VertexT, NormalT>::save()
{

}

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::~Texture() {
}

}
