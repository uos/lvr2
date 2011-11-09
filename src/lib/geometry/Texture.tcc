/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /*
 * Texture.tcc
 *
 *  @date 08.09.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

namespace lssr {

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::Texture(PointCloudManager<VertexT, NormalT>* pm, Region<VertexT, NormalT>* region, vector<vector<HVertex*> > contours)
{
	this->m_region = region;
	this->m_data   = 0;

	//determines the texture resolution
	m_pixelSize = 1;

	int minArea = INT_MAX;

	if(this->m_region->m_inPlane)
	{
		if(contours.size() > 0 && contours[0].size() > 2)
		{
			vector<HVertex*> HOuter_contour = contours[0];
			NormalT n = m_region->m_normal;

			//store a stuetzvector for the bounding box
			p = HOuter_contour[0]->m_position;

			//calculate a vector in the plane of the bounding box
			NormalT v1 = HOuter_contour[1]->m_position - HOuter_contour[0]->m_position, v2;

			//determines the resolution of iterative improvement steps
			float delta = M_PI / 2 / 90;

			for(float theta = 0; theta < M_PI / 2; theta += delta)
			{
				//rotate the bounding box
				v1 = v1 * cos(theta) + v2 * sin(theta);
				v2 = v1.cross(n);

				//calculate the bounding box
				float a_min = FLT_MAX, a_max = FLT_MIN, b_min = FLT_MAX, b_max = FLT_MIN;
				for(size_t c = 0; c < HOuter_contour.size(); c++)
				{
					int r = 0;
					int s = 0;
					float denom = 0.01;
					for(int t = 0; t < 3; t++)
					{
						for(int u = 0; u < 3; u++)
						{
							if(fabs(v1[t] * v2[u] - v1[u] * v2[t]) > fabs(denom))
							{
								denom = v1[t] * v2[u] - v1[u] * v2[t];
								r = t;
								s = u;
							}
						}
					}
					float a = ((HOuter_contour[c]->m_position[r] - p[r]) * v2[s] - (HOuter_contour[c]->m_position[s] - p[s]) * v2[r]) / denom;
					float b = ((HOuter_contour[c]->m_position[s] - p[s]) * v1[r] - (HOuter_contour[c]->m_position[r] - p[r]) * v1[s]) / denom;
					if (a > a_max) a_max = a;
					if (a < a_min) a_min = a;
					if (b > b_max) b_max = b;
					if (b < b_min) b_min = b;
				}
				int x = ceil((a_max - a_min) / m_pixelSize);
				int y = ceil((b_max - b_min) / m_pixelSize);

				//iterative improvement of the area
				if(x * y < minArea)
				{
					minArea = x * y;
					best_a_min = a_min;
					best_a_max = a_max;
					best_b_min = b_min;
					best_b_max = b_max;
					best_v1 = v1;
					best_v2 = v2;
				}
			}

			//calculate the texture size and round up to a size to base 2
			this->m_sizeX = ceil((best_a_max - best_a_min) / m_pixelSize);
			this->m_sizeX = pow(2, ceil(log(this->m_sizeX) / log(2)));
			this->m_sizeY = ceil((best_b_max - best_b_min) / m_pixelSize);
			this->m_sizeY = pow(2, ceil(log(this->m_sizeY) / log(2)));

			m_data = new ColorT*[this->m_sizeY];

			//walk through the bounding box and collect color information for each texel
            #pragma omp parallel for
			for(int y = 0; y < this->m_sizeY; y++)
			{
				m_data[m_sizeY-y-1] = new ColorT[this->m_sizeX];
				for(int x = 0; x < this->m_sizeX; x++)
				{
					if (y <= (best_b_max - best_b_min) / m_pixelSize  && x <= (best_a_max - best_a_min) / m_pixelSize)
					{
						vector<VertexT> cv;

						VertexT current_position = p + best_v1 * (x * m_pixelSize + best_a_min - m_pixelSize / 2.0) + best_v2 * (y * m_pixelSize + best_b_min - m_pixelSize / 2.0);

						int one = 1;
						pm->getkClosestVertices(current_position, one, cv);

						ColorT currCol;
						currCol.r = cv[0].r;
						currCol.g = cv[0].g;
						currCol.b = cv[0].b;
						m_data[m_sizeY - y - 1][x] = currCol;
					}
					else
					{
						ColorT currCol;
						currCol.r = 0;
						currCol.g = 0;
						currCol.b = 255;
						m_data[m_sizeY - y - 1][x] = currCol;
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
		for (int y = 0; y < this->m_sizeY; y++)
		{
			m_data[y] = new ColorT[this->m_sizeX];
			memset(m_data[y], 200, this->m_sizeX * sizeof(ColorT));
		}
	}
}

template<typename VertexT, typename NormalT>
void Texture<VertexT, NormalT>::textureCoords(VertexT v, float &x, float &y)
{
	 VertexT t =  v - ((best_v1 * best_a_min) + (best_v2 * best_b_min) + p);
	 x = (best_v1 * (t * best_v1)).length() / m_pixelSize / m_sizeX;
	 y = (best_v2 * (t * best_v2)).length() / m_pixelSize / m_sizeY;

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
	for(int y = 0; y < m_sizeY; y++)
	{
		delete m_data[y];
	}
	delete m_data;
}

}
