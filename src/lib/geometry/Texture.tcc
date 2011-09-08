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
	this->m_region = region;

	if(this->m_region->m_inPlane)
		vector<HVertex*> HOuter_contour = this->m_region->getContours(0.01)[0];

	//Just for testing...
	this->m_sizeX = 640;
	this->m_sizeY = 480;
	m_data = new ColorT*[this->m_sizeY];
	for (int y = 0; y<this->m_sizeY; y++)
	{
		m_data[y] = new ColorT[this->m_sizeX];
		memset(m_data[y],200,this->m_sizeX*sizeof(ColorT));
	}
	//End testing

}

template<typename VertexT, typename NormalT>
void Texture<VertexT, NormalT>::save()
{
	PPMIO<ColorT> ppm;
	ppm.setDataArray(this->m_data,this->m_sizeX, this->m_sizeY);
	ppm.write("texture_"+boost::lexical_cast<std::string>(this->m_region->m_region_number)+".ppm");
}

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::~Texture() {
	for(int y = 0; y<m_sizeY; y++)
		delete m_data[y];
	delete m_data;
}

}
