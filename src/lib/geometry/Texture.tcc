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
		if(contours.size() > 0 && contours[0].size() > 2)
		{
			vector<HVertex*> HOuter_contour = contours[0];
			NormalT n = m_region->m_normal;
			p = HOuter_contour[0]->m_position;
			v1 = HOuter_contour[1]->m_position - HOuter_contour[0]->m_position;
			v2 = v1.cross(n);
			a_min = FLT_MAX; a_max = FLT_MIN; b_min = FLT_MAX; b_max = FLT_MIN;

			for(size_t c = 0; c < HOuter_contour.size(); c++)
			{
				int r = 0;
				int s = 1;
				if(n[0] == 0 && n[2] == 0)
					s = 2;
				if(n[1] == 0 && n[2] == 0)
				{
					r = 1;
					s = 2;
				}
				float denom = v1[r] * v2[s] - v1[s] * v2[r];
				float a = ((HOuter_contour[c]->m_position[r] - p[r]) * v2[s] - (HOuter_contour[c]->m_position[s] - p[s]) * v2[r]) / denom;
				float b = ((HOuter_contour[c]->m_position[s] - p[s]) * v1[r] - (HOuter_contour[c]->m_position[r] - p[r]) * v1[s]) / denom;


				if (a > a_max) a_max = a;
				if (a < a_min) a_min = a;
				if (b > b_max) b_max = b;
				if (b < b_min) b_min = b;
			}

			m_pixelSize = 1;
			this->m_sizeX = (a_max - a_min) / m_pixelSize;
			this->m_sizeX = pow(2, ceil(log(this->m_sizeX)/log(2)));
			this->m_sizeY = (b_max - b_min) / m_pixelSize;
			this->m_sizeY = pow(2, ceil(log(this->m_sizeY)/log(2)));

			m_data = new ColorT*[this->m_sizeY];

			for(int y = 0; y < this->m_sizeY; y++)
			{
				m_data[m_sizeY-y-1] = new ColorT[this->m_sizeX];
				for(int x = 0; x < this->m_sizeX; x++)
				{
					if (y <= (b_max - b_min) / m_pixelSize  && x <= (a_max - a_min) / m_pixelSize)
					{
						vector<VertexT> cv;

						VertexT current_position = p + v1 * (x * m_pixelSize + a_min - m_pixelSize/2.0) + v2 * (y * m_pixelSize + b_min - m_pixelSize/2.0);

						int one = 1;
						pm->getkClosestVertices(current_position, one, cv);

						ColorT currCol;
						currCol.r = cv[0].r;
						currCol.g = cv[0].g;
						currCol.b = cv[0].b;
						m_data[m_sizeY-y-1][x] = currCol;
					}
					else
					{
						ColorT currCol;
						currCol.r =0;
						currCol.g = 0;
						currCol.b = 255;
						m_data[m_sizeY-y-1][x] = currCol;
					}

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
	 x = (v1 * (t * v1)).length()/m_pixelSize / m_sizeX;
	 y = (v2 * (t * v2)).length()/m_pixelSize / m_sizeY;

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
	ppm.write("texture_"+boost::lexical_cast<std::string>(this->m_region->m_regionNumber)+".ppm");
}

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::~Texture() {
	for(int y = 0; y<m_sizeY; y++)
		delete m_data[y];
	delete m_data;
}

}
