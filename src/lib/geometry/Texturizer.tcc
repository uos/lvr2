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

        ///File name of texture pack
	template<typename VertexT, typename NormalT>
        string Texturizer<VertexT, NormalT>::m_filename = "";

        ///Number of colors to use for stats calculations
	template<typename VertexT, typename NormalT>
        unsigned int Texturizer<VertexT, NormalT>::m_numStatsColors = 0;

        ///Number of colors to use for CCV calculation
	template<typename VertexT, typename NormalT>
        unsigned int Texturizer<VertexT, NormalT>::m_numCCVColors = 0;

        ///coherence threshold for CCV calculation
	template<typename VertexT, typename NormalT>
        unsigned int Texturizer<VertexT, NormalT>::m_coherenceThreshold = 0;

        ///Threshold for color based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_colorThreshold = 0;

        ///Threshold for cross correlation based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_crossCorrThreshold = 0;

        ///Threshold for statistics based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_statsThreshold = 0;

        ///Threshold for feature based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_featureThreshold = 0;

        ///Threshold for pattern extraction
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_patternThreshold = 0;


template<typename VertexT, typename NormalT>
Texturizer<VertexT, NormalT>::Texturizer(typename PointsetSurface<VertexT>::Ptr pm)
{
	//Load texture package
	this->m_tio = new TextureIO(Texturizer::m_filename);
	
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


	//calculate the texture size
	unsigned short int sizeX = ceil((best_a_max - best_a_min) / Texture::m_texelSize);
	unsigned short int sizeY = ceil((best_b_max - best_b_min) / Texture::m_texelSize);

	//create the texture
	Texture* texture = new Texture(sizeX, sizeY, 3, 1, 0, 0, 0, 0, 0, false, 0, 0);

	//create TextureToken
	TextureToken<VertexT, NormalT>* result = new TextureToken<VertexT, NormalT>(best_v1, best_v2, p, best_a_min, best_b_min, texture);
 

	//walk through the bounding box and collect color information for each texel
	#pragma omp parallel for
	for(int y = 0; y < sizeY; y++)
	{
		for(int x = 0; x < sizeX; x++)
		{
			vector<VertexT> cv;

			VertexT current_position = p + best_v1
				* (x * Texture::m_texelSize + best_a_min - Texture::m_texelSize / 2.0)
				+ best_v2
				* (y * Texture::m_texelSize + best_b_min - Texture::m_texelSize / 2.0);

			int one = 1;
			m_pm->searchTree()->kSearch(current_position, one, cv);

			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = cv[0].b;
			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = cv[0].g;
			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = cv[0].r;
		}
	}

	//calculate SURF features of  texture
	ImageProcessor::calcSURF(texture);

	//calculate statistics
	ImageProcessor::calcStats(texture, Texturizer<VertexT, NormalT>::m_numStatsColors); 

	//calculate CCV
	ImageProcessor::calcCCV(texture, Texturizer<VertexT, NormalT>::m_numCCVColors, Texturizer<VertexT, NormalT>::m_coherenceThreshold);

	return result;
}

template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByColor(vector<Texture*> &textures, Texture* refTexture, float threshold)
{
	vector<Texture*> toDelete;

	//Filter by histogram
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesHist(textures[i], refTexture);
		textures[i]->m_distance += dist;
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
	}	
	//filter by CCV
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesCCV(textures[i], refTexture);
		textures[i]->m_distance += dist;
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
	}	
	
}
template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByCrossCorr(vector<Texture*> &textures, Texture* refTexture, float threshold)
{
	vector<Texture*> toDelete;

	//filter by CC
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = 0;//TODO
		textures[i]->m_distance += dist;
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
	}	
}
template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByStats(vector<Texture*> &textures, Texture* refTexture, float threshold)
{
	vector<Texture*> toDelete;

	//filter by stats
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesStats(textures[i], refTexture);
		textures[i]->m_distance += dist;
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
	}	
}
template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByFeatures(vector<Texture*> &textures, Texture* refTexture, float threshold)
{
	vector<Texture*> toDelete;

	//filter by features
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesSURF(textures[i], refTexture);
		textures[i]->m_distance += dist;
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
	}	
}

template<typename VertexT, typename NormalT>
TextureToken<VertexT, NormalT>* Texturizer<VertexT, NormalT>::texturizePlane(vector<VertexT> contour)
{
//	std::cout<<"==================================================================="<<std::endl;
	TextureToken<VertexT, NormalT>* initialTexture = 0;

	float colorThreshold 		= Texturizer<VertexT, NormalT>::m_colorThreshold;
	float crossCorrThreshold 	= Texturizer<VertexT, NormalT>::m_crossCorrThreshold;
	float statsThreshold 		= Texturizer<VertexT, NormalT>::m_statsThreshold;
	float featureThreshold 		= Texturizer<VertexT, NormalT>::m_featureThreshold;
	float patternThreshold 		= FLT_MAX;//Texturizer<VertexT, NormalT>::m_patternThreshold; //TODO: uncomment


	if(contour.size() >= 3)
	{
		//create an initial texture from the point cloud
		initialTexture = createInitialTexture(contour);

		//reset distance values
		for (int i = 0; i < this->m_tio->m_textures.size(); i++)
		{
			this->m_tio->m_textures[i]->m_distance = 0;
		}
		//reduce number of matching textures from the texture pack step by step
		std::vector<Texture*> textures = this->m_tio->m_textures;
		filterByColor		(textures, initialTexture->m_texture, colorThreshold);
		filterByStats		(textures, initialTexture->m_texture, statsThreshold);
		filterByFeatures	(textures, initialTexture->m_texture, featureThreshold);
//		filterByCrossCorr	(textures, initialTexture->m_texture, crossCorrThreshold); //TODO
		sort(textures.begin(), textures.end(), Texture::cmpTextures);		

		if (textures.size() > 0)
		{
			cout<<"Using Texture from texture package!!!"<<endl;

			//TODO: Transform parameters for texture coordinate calculation
			Transform* tans = new Transform(initialTexture->m_texture, textures[0]);
			//Found matching textures in texture package -> use best match
			return new TextureToken<VertexT, NormalT>(	initialTexture->v1, initialTexture->v2,
									initialTexture->p, 
									initialTexture->a_min, initialTexture->b_min,
									textures[0], find(this->m_tio->m_textures.begin(), this->m_tio->m_textures.end(), textures[0]) - this->m_tio->m_textures.begin());
		}
		else
		{
			//Try to extract pattern
			Texture* pattern = 0;
			if (ImageProcessor::extractPattern(initialTexture->m_texture, &pattern) > patternThreshold)
			{
				cout<<"Using pattern texture!!!"<<endl;
				//calculate surf features for pattern
				ImageProcessor::calcSURF(pattern);
				//calculate statistics for pattern
				ImageProcessor::calcStats(pattern, Texturizer<VertexT, NormalT>::m_numStatsColors); 
				//calculate CCV for pattern
				ImageProcessor::calcCCV(pattern, Texturizer<VertexT, NormalT>::m_numCCVColors, Texturizer<VertexT, NormalT>::m_coherenceThreshold);

				//Add pattern to texture package
				int index = this->m_tio->add(pattern);
				this->m_tio->write();

				//return a texture token
				return new TextureToken<VertexT, NormalT>(	initialTexture->v1, initialTexture->v2,
										initialTexture->p, 
										initialTexture->a_min, initialTexture->b_min,
										pattern, index);
			}
			else
			{
				cout<<"Using initial texture"<<endl;
				//Pattern extraction failed -> use initial texture
				delete pattern; 
				//Add initial texture to texture pack */
				initialTexture->m_textureIndex = this->m_tio->add(initialTexture->m_texture);
				this->m_tio->write();
			}
		}
	} 
	return initialTexture;
}

}
