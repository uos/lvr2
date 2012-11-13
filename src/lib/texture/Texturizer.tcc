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
        float Texturizer<VertexT, NormalT>::m_colorThreshold = FLT_MAX;

        ///Threshold for cross correlation based texture filtering
	template<typename VertexT, typename NormalT>
        bool Texturizer<VertexT, NormalT>::m_useCrossCorr = false;

        ///Threshold for statistics based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_statsThreshold = FLT_MAX;

        ///Threshold for feature based texture filtering
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_featureThreshold = FLT_MAX;

        ///Threshold for pattern extraction
	template<typename VertexT, typename NormalT>
        float Texturizer<VertexT, NormalT>::m_patternThreshold = 0;


template<typename VertexT, typename NormalT>
Texturizer<VertexT, NormalT>::Texturizer(typename PointsetSurface<VertexT>::Ptr pm)
{
	//Load texture package
	this->m_tio = new TextureIO(Texturizer::m_filename);
	
	this->m_pm = pm;

        m_stats_texturizedPlanes = 0;
        m_stats_matchedIndTextures = 0;
        m_stats_matchedPatTextures = 0;
        m_stats_extractedPatterns = 0;
}

template<typename VertexT, typename NormalT>
TextureToken<VertexT, NormalT>* Texturizer<VertexT, NormalT>::createInitialTexture(vector<VertexT> contour)
{

	int minArea = INT_MAX;

	float best_a_min, best_a_max, best_b_min, best_b_max;
	VertexT best_v1, best_v2;

	NormalT n = (contour[1] - contour[0]).cross(contour[2] - contour[0]);
	if (n.x < 0)
	{
		n *= -1;
	}

	//store a stuetzvector for the bounding box
	VertexT p = contour[0];

	//calculate a vector in the plane of the bounding box
	NormalT v1 = contour[1] - contour[0], v2;
	if (v1.x < 0)
	{
		v1 *= -1;
	}

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
	Texture* texture = new Texture(sizeX, sizeY, 3, 1, Texturizer<VertexT, NormalT>::classifyNormal(n), 0, 0, 0, 0, 0, false, 0, 0);

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

			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 0] = cv[0].r;
			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 1] = cv[0].g;
			texture->m_data[(sizeY - y - 1) * (sizeX * 3) + 3 * x + 2] = cv[0].b;
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
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
		textures[i]->m_distance += dist;
	}	
	//filter by CCV
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesCCV(textures[i], refTexture);
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
		textures[i]->m_distance += dist;
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		if(find(textures.begin(), textures.end(), toDelete[d]) != textures.end())
		{
			textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
		}
	}	
	
}
template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByCrossCorr(vector<Texture*> &textures, Texture* refTexture)
{
	//filter by CC
	for (int i = 0; i < textures.size(); i++)
	{
		float dist = ImageProcessor::compareTexturesCrossCorr(textures[i], refTexture);
		textures[i]->m_distance += dist;
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
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
		textures[i]->m_distance += dist;
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
		if(dist > threshold)
		{
			toDelete.push_back(textures[i]);
		}
		textures[i]->m_distance += dist;
	}	
	
	//delete bad matches
	for (int d = 0; d < toDelete.size(); d++)
	{
		textures.erase(find(textures.begin(), textures.end(), toDelete[d]));
	}	
}
template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::filterByNormal(vector<Texture*> &textures, vector<VertexT> contour)
{
	vector<Texture*> toDelete;

	//calculate normal of plane
	NormalT n = (contour[1] - contour[0]).cross(contour[2]-contour[0]);
	
	//filter by normal
	for (int i = 0; i < textures.size(); i++)
	{
		if(Texturizer<VertexT, NormalT>::classifyNormal(n) != textures[i]->m_textureClass)
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
	TextureToken<VertexT, NormalT>* initialTexture = 0;

	float colorThreshold 		= Texturizer<VertexT, NormalT>::m_colorThreshold;
	bool  useCrossCorr 		= Texturizer<VertexT, NormalT>::m_useCrossCorr;
	float statsThreshold 		= Texturizer<VertexT, NormalT>::m_statsThreshold;
	float featureThreshold 		= Texturizer<VertexT, NormalT>::m_featureThreshold;
	float patternThreshold 		= Texturizer<VertexT, NormalT>::m_patternThreshold;


	if(contour.size() >= 3)
	{
		m_stats_texturizedPlanes++;
		//create an initial texture from the point cloud
		initialTexture = createInitialTexture(contour);

		//reset distance values
		for (int i = 0; i < this->m_tio->m_textures.size(); i++)
		{
			this->m_tio->m_textures[i]->m_distance = 0;
		}
		//reduce number of matching textures from the texture pack step by step
		std::vector<Texture*> textures = this->m_tio->m_textures;
		filterByNormal		(textures, contour);

		if (colorThreshold != FLT_MAX)
		{
			filterByColor		(textures, initialTexture->m_texture, colorThreshold);
		}
		if (statsThreshold != FLT_MAX)
		{
			filterByStats		(textures, initialTexture->m_texture, statsThreshold);
		}
		if (featureThreshold != FLT_MAX)
		{
			filterByFeatures	(textures, initialTexture->m_texture, featureThreshold);
		}
		if (useCrossCorr != FLT_MAX)
		{
			filterByCrossCorr	(textures, initialTexture->m_texture); 
		}

		sort(textures.begin(), textures.end(), Texture::cmpTextures);		

		if (textures.size() > 0)
		{
			//Found matching textures in texture package -> use best match
			TextureToken<VertexT, NormalT>* result = new TextureToken<VertexT, NormalT>(
									initialTexture->v1, initialTexture->v2, initialTexture->p,
									initialTexture->a_min, initialTexture->b_min, textures[0],
									find(this->m_tio->m_textures.begin(), this->m_tio->m_textures.end(), textures[0])
									- this->m_tio->m_textures.begin());
			if(textures[0]->m_isPattern)
			{
				m_stats_matchedPatTextures++;
			//	cout<<"Using Pattern Texture from texture package!!!"<<endl;
			}
			else
			{
				m_stats_matchedIndTextures++;
			//	cout<<"Using Texture from texture package!!!"<<endl;
			//	cerr<<"Distance: "<<textures[0]->m_distance <<endl;
				//Calculate transformation for texture coordinate calculation
				Transform* trans = new Transform(initialTexture->m_texture, textures[0]);
				double* mat = trans->getTransArr();
				for (int i = 0; i < 6; i++)
				{	
					result->m_transformationMatrix[i] = mat[i];
				}
				result->m_mirrored = trans->m_mirrored;
				delete mat;
				delete trans;
			}
			return result;
		}
		else
		{
			//Try to extract pattern
			Texture* pattern = 0;
			float pattern_quality = ImageProcessor::extractPattern(initialTexture->m_texture, &pattern);
		//	cout<<pattern_quality<<" ";
			if (pattern_quality > patternThreshold)
			{
			//	cout<<"Using pattern texture!!! "<<pattern_quality<<endl;
				m_stats_extractedPatterns++;
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
				TextureToken<VertexT, NormalT>* result = new TextureToken<VertexT, NormalT>(	initialTexture->v1, initialTexture->v2,
										initialTexture->p, 
										initialTexture->a_min, initialTexture->b_min,
										pattern, index);
				return result;
			}
			else 
			{
			//	cout<<"Using initial texture";
				//Pattern extraction failed -> use initial texture
				delete pattern; 
				//Add initial texture to texture pack
				initialTexture->m_textureIndex = this->m_tio->add(initialTexture->m_texture);
				this->m_tio->write();
			//	cout<<initialTexture->m_textureIndex<<endl;
			}
		}
	} 
	return initialTexture;
}

template<typename VertexT, typename NormalT>
unsigned short int Texturizer<VertexT, NormalT>::classifyNormal(NormalT n)
{
	float epsilon = 0.1;
	
	//wall
	if (fabs(n * NormalT(0,0,1)) < epsilon)
	{
		return 1;
	}

	//ceiling or floor
	if (fabs(fabs(n * NormalT(0,0,1)) - 1) < epsilon)
	{
		return 2;
	}

	//other
	return 0;
}


template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::markTexture(TextureToken<VertexT, NormalT>* tt, char color)
{
	Texture* t = tt->m_texture;
	cv::Mat img(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);
	switch (color)
	{
		case 'r':
				cv::putText(img, "#########", cv::Point2f(0,img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 2);
				break;
		case 'g':	
				cv::putText(img, "#########", cv::Point2f(0,img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 2);
				break;
	}
}

template<typename VertexT, typename NormalT>
void Texturizer<VertexT, NormalT>::showTexture(TextureToken<VertexT, NormalT>* tt, string caption)
{
	Texture* t = tt->m_texture;

	for (int i = 0; i < 6; i++)
	{
		std::cout<<std::setw(10)<<tt->m_transformationMatrix[i];
	}
	std::cout<<std::endl;
	cv::Mat img(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);

	cv::putText(img, caption, cv::Point2f(0,img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 2);

	cv::startWindowThread();
	
	//show the reference image
	cv::namedWindow("Window", CV_WINDOW_AUTOSIZE);
	cv::imshow("Window", img);
	cv::waitKey();

	cv::destroyAllWindows();
}

}
