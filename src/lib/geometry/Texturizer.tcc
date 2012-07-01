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
 * Texturizer.tcc
 *
 *  @date 03.05.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

namespace lssr {

template<typename VertexT, typename NormalT>
Texturizer<VertexT, NormalT>::Texturizer(typename PointsetSurface<VertexT>::Ptr pm, string filename)
{
	//Load texture package
	this->m_tio = new TextureIO(filename);
	
	this->m_pm = pm;
}

template<typename VertexT, typename NormalT>
TextureToken<VertexT, NormalT>* Texturizer<VertexT, NormalT>::createInitialTexture(vector<VertexT> contour)
{
	int minArea = INT_MAX;

	float best_a_min, best_a_max, best_b_min, best_b_max;
	VertexT best_v1, best_v2;

	NormalT n = (contour[1] - contour[0]).cross(contour[2] - contour[0]);

	//store a stuetzvector for the bounding box
	VertexT p = contour[0];

	//calculate a vector in the plane of the bounding box
	NormalT v1 = contour[1] - contour[0], v2;

	//determines the resolution of iterative improvement steps
	float delta = M_PI / 2 / 90;

	for(float theta = 0; theta < M_PI / 2; theta += delta)
	{
		//rotate the bounding box
		v1 = v1 * cos(theta) + v2 * sin(theta);
		v2 = v1.cross(n);

		//calculate the bounding box
		float a_min = FLT_MAX, a_max = FLT_MIN, b_min = FLT_MAX, b_max = FLT_MIN;
		for(size_t c = 0; c < contour.size(); c++)
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
			float a = ((contour[c][r] - p[r]) * v2[s] - (contour[c][s] - p[s]) * v2[r]) / denom;
			float b = ((contour[c][s] - p[s]) * v1[r] - (contour[c][r] - p[r]) * v1[s]) / denom;
			if (a > a_max) a_max = a;
			if (a < a_min) a_min = a;
			if (b > b_max) b_max = b;
			if (b < b_min) b_min = b;
		}
		int x = ceil((a_max - a_min) / Texture::m_texelSize);
		int y = ceil((b_max - b_min) / Texture::m_texelSize);

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
	unsigned short int sizeX = ceil((best_a_max - best_a_min) / Texture::m_texelSize);
	sizeX = max(8.0, pow(2, ceil(log(sizeX) / log(2))));
	unsigned short int sizeY = ceil((best_b_max - best_b_min) / Texture::m_texelSize);
	sizeY = max(8.0, pow(2, ceil(log(sizeY) / log(2))));

	//create the texture
	Texture* texture = new Texture(sizeX, sizeY, 3, 1, 0, 0, 0, 0);

	//create TextureToken
	TextureToken<VertexT, NormalT>* result = new TextureToken<VertexT, NormalT>(best_v1, best_v2, p, best_a_min, best_b_min, texture);
 

	//walk through the bounding box and collect color information for each texel
	#pragma omp parallel for
	for(int y = 0; y < sizeY; y++)
	{
		for(int x = 0; x < sizeX; x++)
		{
			if (y <= (best_b_max - best_b_min) / Texture::m_texelSize  && x <= (best_a_max - best_a_min) / Texture::m_texelSize)
			{
				vector<VertexT> cv;

				VertexT current_position = p + best_v1
					* (x * Texture::m_texelSize + best_a_min - Texture::m_texelSize / 2.0)
					+ best_v2
					* (y * Texture::m_texelSize + best_b_min - Texture::m_texelSize / 2.0);

				int one = 1;
				m_pm->searchTree()->kSearch(current_position, one, cv);

				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = cv[0].r;
				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = cv[0].g;
				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = cv[0].b;
			}
			else
			{
				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = 0;
				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = 0;
				texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = 255;
			}

		}
	}

	//calculate SURF features of  texture
	ImageProcessor::calcSURF(texture);

	return result;
}

template<typename VertexT, typename NormalT>
TextureToken<VertexT, NormalT>* Texturizer<VertexT, NormalT>::texturizePlane(vector<VertexT> contour)
{
	//create an initial texture from the point cloud
	TextureToken<VertexT, NormalT>* initialTexture = createInitialTexture(contour);
	std::cout<<"==================================================================="<<std::endl;
	float minSurfDist = FLT_MAX;
	int minIndex = -1;
	//Check all textures of the texture package 
	for(int i = 0; i < this->m_tio->m_textures.size(); i++)
	{
		//TODO: other methods for texture matching
		//SURF
		float surfDistance = ImageProcessor::compareTexturesSURF(initialTexture->m_texture, this->m_tio->m_textures[i]);
		if(surfDistance < minSurfDist)
		{
			minSurfDist = surfDistance;
			minIndex = i;
		}
	}
	std::cout<<minSurfDist<<std::endl;
	if (minSurfDist < 0.02) //TODO: Param
	{
		cout<<"Using Texture from texture package!!!"<<endl;
		//Found a matching texture
		return new TextureToken<VertexT, NormalT>(	initialTexture->v1, initialTexture->v2,
								initialTexture->p, 
								initialTexture->a_min, initialTexture->b_min,
								this->m_tio->m_textures[minIndex]);
	}

/*
	//No texture found -> try to extract a pattern
	Texture* pattern = 0;
	if (ImageProcessor::extractPattern(initialTexture->m_texture, &pattern) > 0.95) //TODO: Param
	{
		//calculate surf features for pattern
		ImageProcessor::calcSURF(pattern);

		//Add pattern to texture package
		this->m_tio->add(pattern);
		this->m_tio->write();

		//return a texture token
		return new TextureToken<VertexT, NormalT>(	initialTexture->v1, initialTexture->v2,
								initialTexture->p, 
								initialTexture->a_min, initialTexture->b_min,
								initialTexture->a_max, initialTexture->b_max,
								pattern);
	}
	else
	{
		delete pattern;
	}
	//TODO: other methods for pattern extraction
*/	

	//Pattern extraction failed -> use initial texture
	//Add initial texture to texture pack
	this->m_tio->add(initialTexture->m_texture);
	this->m_tio->write();
	return initialTexture;
}

}
