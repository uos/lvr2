/*
 * Texture.tcc
 *
 *  Created on: 08.09.2011
 *      Author: pg2011
 */

namespace lssr {

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::Texture(PointCloudManager<VertexT, NormalT>* pm, Region<VertexT, NormalT>* region, vector<vector<HVertex*> > contours)
{
	this->m_region = region;
	this->m_data = 0;

	if(this->m_region->m_inPlane)
	{
		if(contours.size() > 0)
		{
			vector<HVertex*> HOuter_contour = contours[0];
			NormalT n = m_region->m_normal;
			p = HOuter_contour[0]->m_position;
			v1 = HOuter_contour[1]->m_position - HOuter_contour[0]->m_position;
			v2 = v1.cross(n);

			a_min = FLT_MAX; a_max = FLT_MIN; b_min = FLT_MAX; b_max = FLT_MIN;

			for(int c = 0; c < HOuter_contour.size(); c++)
			{
				float denom = v1[0] * v2[1] - v1[1] * v2[0];
				float a = ((HOuter_contour[c]->m_position[0] - p[0]) * v2[1] - (HOuter_contour[c]->m_position[1] - p[1]) * v2[0]) / denom;
				float b = ((HOuter_contour[c]->m_position[1] - p[1]) * v1[0] - (HOuter_contour[c]->m_position[0] - p[0]) * v1[1]) / denom;
				if (a > a_max) a_max = a;
				if (a < a_min) a_min = a;
				if (b > b_max) b_max = b;
				if (b < b_min) b_min = b;
			}

			float pixelSize = 1;
			this->m_sizeX = (a_max - a_min) / pixelSize;
			this->m_sizeY = (b_max - b_min) / pixelSize;

//			cout<<"sizeX: "<<m_sizeX<<endl;
//			cout<<"sizeY: "<<m_sizeY<<endl;

			m_data = new ColorT*[this->m_sizeY];

			for(int y = 0; y < this->m_sizeY; y++)
			{
				m_data[y] = new ColorT[this->m_sizeX];
				for(int x = 0; x < this->m_sizeX; x++)
				{
					vector<VertexT> cv;

					VertexT current_position = p + v1 * (x * pixelSize + a_min - pixelSize/2.0) + v2 * (y * pixelSize + b_min - pixelSize/2.0);

					int one = 1;
					pm->getkClosestVertices(current_position, one, cv);

					ColorT currCol;
					currCol.r = cv[0].r;
					currCol.g = cv[0].g;
					currCol.b = cv[0].b;
					m_data[y][x] = currCol;
				}
			}
		}
	}

	//default texture
	if(m_data == 0)
	{
		this->m_sizeX = 1;
		this->m_sizeY = 1;
		m_data = new ColorT*[this->m_sizeY];
		for (int y = 0; y<this->m_sizeY; y++)
		{
			m_data[y] = new ColorT[this->m_sizeX];
			memset(m_data[y],200,this->m_sizeX*sizeof(ColorT));
		}
	}
}

template<typename VertexT, typename NormalT>
void Texture<VertexT, NormalT>::textureCoords(VertexT v, float &x, float &y)
{
	 VertexT t =  v - ((v1 * a_min) + (v2 * b_min) + p);
	 x = (v1 * (t * v1)).length() / m_sizeX;
	 y = (v2 * (t * v2)).length() / m_sizeY;

	 x = x > 1 ? 1 : x;
	 x = x < 0 ? 0 : x;
	 y = y > 1 ? 1 : y;
	 y = y < 0 ? 0 : y;
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
