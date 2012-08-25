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
 * ImageProcessor.cpp
 *
 *  @date 12.06.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "ImageProcessor.hpp"
#include <opencv/cv.h>

namespace lssr {

void ImageProcessor::reduceColors(cv::Mat input, cv::Mat &output, int numColors)
{
	//allocate output
	output = cv::Mat(input.size(), CV_8U);
	//3 channel pointer to input image
	cv::Mat_<cv::Vec3b>& ptrInput = (cv::Mat_<cv::Vec3b>&)input; 
	//1 channel pointer to output image
	cv::Mat_<uchar>& ptrOutput = (cv::Mat_<uchar>&)output;

	for (int y = 0; y < input.size().height; y++)
	{
		for(int x = 0; x < input.size().width; x++)
		{
			unsigned long int currCol = 0;
			currCol |= (ptrInput(y, x)[0]) << 16;
			currCol |= (ptrInput(y, x)[1]) <<  8;
			currCol |= (ptrInput(y, x)[2]) <<  0;
			ptrOutput(y,x) = currCol / (pow(2, 24) / numColors);
		}
	}
}


void ImageProcessor::reduceColorsG(cv::Mat input, cv::Mat &output, int numColors)
{
	//allocate output
	output = cv::Mat(input.size(), CV_8U);

	for (int y = 0; y < input.size().height; y++)
	{
		for(int x = 0; x < input.size().width; x++)
		{
			output.at<uchar>(y,x) = input.at<uchar>(y,x) / (256.0f / numColors);
		}
	}
}

unsigned long int ImageProcessor::find(unsigned long int x, unsigned long int parent[])
{
	while(parent[x] != x)
	{
		parent[x] = parent[parent[x]]; //path halving
		x = parent[x];
	}
	return x;
}

void ImageProcessor::unite(unsigned long int x, unsigned long int y, unsigned long int parent[])
{
	parent[ImageProcessor::find(x, parent)] = ImageProcessor::find(y, parent);
}

void ImageProcessor::connectedCompLabeling(cv::Mat input, cv::Mat &output)
{
	//Allocate output and set it to zero
	output = cv::Mat(input.size(), CV_16U);
	output.setTo(cv::Scalar(0));
	
	//1 channel pointer to input image
	cv::Mat_<uchar>& ptrInput = (cv::Mat_<uchar>&)input;
	//1 channel pointer to output image
	cv::Mat_<ushort>& ptrOutput = (cv::Mat_<ushort>&)output; 

	//disjoint set data structure to manage the labels 
	unsigned long int* parent = new unsigned long int[output.size().height * output.size().width];
	for(unsigned long int i = 0; i < output.size().height * output.size().width; i++) parent[i] = i;

	std::vector<int>  rank (output.size().height * output.size().width);
//	std::vector<int>  parent (output.size().height * output.size().width);
//	boost::disjoint_sets<int*,int*> ds(&rank[0], &parent[0]);
//	for(unsigned long int i = 0; i < output.size().height * output.size().width; i++) {ds.make_set(i);}

	//first pass: Initial labeling
	unsigned short int currentLabel = 0;
	for (int y = 0; y < input.size().height; y++)
	{
		for(int x = 0; x < input.size().width; x++)
		{
			if (y == 0)
			{
				if(x == 0)
				{
					//First pixel. Create first label.
					ptrOutput(y,x) = ++currentLabel;
				}
				else
				{
					//First row. Only check left pixel	
					if (ptrInput(y,x) == ptrInput(y, x - 1))
					{
						//same region as left pixel -> assign same label
						ptrOutput(y,x) = ptrOutput(y, x - 1);
					}
					else
					{
						//different region -> create new label
						ptrOutput(y,x) = ++currentLabel;
					}
				}
			}
			else
			{
				if(x == 0)
				{
					//First column. Only check top pixel	
					if (ptrInput(y,x) == ptrInput(y - 1, x))
					{
						//same region as top pixel -> assign same label
						ptrOutput(y,x) = ptrOutput(y - 1, x);
					}
					else
					{
						//different region -> create new label
						ptrOutput(y,x) = ++currentLabel;
					}
				}
				else
				{
					//Regular column. Check top and left pixel
					if (ptrInput(y,x) == ptrInput(y, x - 1) && ptrInput(y,x) == ptrInput(y - 1, x))
					{
						//same region as left and top pixel -> assign minimum label of both
						ptrOutput(y,x) = std::min(ptrOutput(y, x - 1), ptrOutput(y - 1, x));
						if (ptrOutput(y, x - 1) != ptrOutput(y - 1, x))
						{
							//mark labels as equivalent
							//we are using the union/find algorithm for disjoint sets
							ImageProcessor::unite(ImageProcessor::find(ptrOutput(y, x - 1), parent),
									      ImageProcessor::find(ptrOutput(y - 1, x), parent), parent);
//							ds.union_set(ptrOutput(y, x - 1), ptrOutput(y - 1, x));
						}
					}
					else
					if (ptrInput(y,x) == ptrInput(y, x - 1))
					{
						//same region as left pixel -> assign same label
						ptrOutput(y,x) = ptrOutput(y, x - 1);
					}
					else
					if (ptrInput(y,x) == ptrInput(y - 1, x))
					{
						//same region as top pixel -> assign same label
						ptrOutput(y,x) = ptrOutput(y - 1, x);
					}
					else
					{
						//different region -> create new label
						ptrOutput(y,x) = ++currentLabel;
					}
				}
			}
		}
	}

	//second pass: Merge equivalent labels
	for (int y = 0; y < output.size().height; y++)
	{
		for(int x = 0; x < output.size().width; x++)
		{
			//we are using the union/find algorithm for disjoint sets
			ptrOutput(y,x) = (unsigned short int) ImageProcessor::find(ptrOutput(y, x), parent);
//			ptrOutput(y,x) = ds.find_set(ptrOutput(y, x));
		}
	}

	delete[] parent;
}


void ImageProcessor::calcSURF(Texture* tex)
{
	if (tex->m_width >= 8 && tex->m_height >= 8)
	{
		//convert texture to cv::Mat
		cv::Mat img1(cv::Size(tex->m_width, tex->m_height), CV_MAKETYPE(tex->m_numBytesPerChan * 8, tex->m_numChannels), tex->m_data);
		//convert image to gray scale
		cv::cvtColor(img1, img1, CV_RGB2GRAY);
		
		//initialize SURF objects
//		cv::SurfFeatureDetector detector(100);
		cv::Ptr<cv::FeatureDetector> detector(new cv::DynamicAdaptedFeatureDetector (new cv::SurfAdjuster(), 100, 110, 30));
		cv::SurfDescriptorExtractor extractor;

		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptors;

		//calculate SURF features for the image
		detector->detect( img1, keyPoints );
		extractor.compute( img1, keyPoints, descriptors );

		//return the results
		tex->m_numFeatures 		= descriptors.rows;
		tex->m_numFeatureComponents	= descriptors.cols;
		tex->m_featureDescriptors = new float[descriptors.rows * descriptors.cols];
		tex->m_keyPoints 		= new float[tex->m_numFeatures * 2];
		for (int r = 0; r < descriptors.rows; r++)
		{
			for (int c = 0; c < descriptors.cols; c++)
			{
				tex->m_featureDescriptors[r * descriptors.cols + c] = descriptors.at<float>(r, c);
			}
			tex->m_keyPoints[r * 2 + 0] 	= 	keyPoints[r].pt.x;
			tex->m_keyPoints[r * 2 + 1] 	= 	keyPoints[r].pt.y;
		}
	}
	else
	{
		tex->m_numFeatures = 0;
	}
}

void ImageProcessor::floatArrToSURF(Texture* t, std::vector<cv::KeyPoint> &kp, cv::Mat &desc)
{
	desc = cv::Mat(t->m_numFeatures, t->m_numFeatureComponents, CV_32FC1);
	for (int r = 0; r < desc.rows; r++)
	{
		for (int c = 0; c < desc.cols; c++)
		{
			desc.at<float>(r, c) = t->m_featureDescriptors[r * desc.cols + c];
		}
		kp.push_back( *(new cv::KeyPoint(t->m_keyPoints[r * 2 + 0], t->m_keyPoints[r * 2 + 1], 0)));
	}
}

float ImageProcessor::compareTexturesSURF(Texture* tex1, Texture* tex2)
{
	float result = FLT_MAX;

	//convert float arrays to cv::Mat
	cv::Mat descriptors1(tex1->m_numFeatures, tex1->m_numFeatureComponents, CV_32FC1);
	for (int r = 0; r < descriptors1.rows; r++)
	{
		for (int c = 0; c < descriptors1.cols; c++)
		{
			descriptors1.at<float>(r, c) = tex1->m_featureDescriptors[r * descriptors1.cols + c];
		}
	}
	cv::Mat descriptors2(tex2->m_numFeatures, tex2->m_numFeatureComponents, CV_32FC1);
	for (int r = 0; r < descriptors2.rows; r++)
	{
		for (int c = 0; c < descriptors2.cols; c++)
		{
			descriptors2.at<float>(r, c) = tex2->m_featureDescriptors[r * descriptors2.cols + c];
		}
	}
	
	if(tex1->m_numFeatures != 0 && tex2->m_numFeatures != 0)
	{
		
		result = 0;

		//calculate matching
		cv::BruteForceMatcher<cv::L2<float> > matcher;
		//cv::FlannBasedMatcher matcher;
		std::vector< cv::DMatch > matches;
		matcher.match( descriptors1, descriptors2, matches);

		//search best match
		double minDist = 100;
		for (int i = 0; i < matches.size(); i++)
		{ 
			if(matches[i].distance < minDist) minDist = matches[i].distance;
		}

		//Calculate result. Only good matches are considered.
		int numGoodMatches = 0;
		for( int i = 0; i < matches.size(); i++ )
		{ 
			if(matches[i].distance <= 2 * minDist)
			{
				result += matches[i].distance * matches[i].distance;
				numGoodMatches++;
			}
		}
		result /= numGoodMatches;
	}
	return result;

}

float ImageProcessor::extractPattern(Texture* tex, Texture** dst)
{
	//convert texture to cv::Mat
	cv::Mat src(cv::Size(tex->m_width, tex->m_height), CV_MAKETYPE(tex->m_numBytesPerChan * 8, tex->m_numChannels), tex->m_data);

	//try to extract pattern
	unsigned int sizeX, sizeY, sX, sY;
	AutoCorr* ac = new AutoCorr(src);
	float result = ac->getMinimalPattern(sX, sY, sizeX, sizeY);
	
	//save the pattern
	cv::Mat pattern = cv::Mat(src, cv::Rect(sX, sY, sizeX, sizeY));

	//convert the pattern to Texture
	*dst = new Texture(sizeX, sizeY, tex->m_numChannels, tex->m_numBytesPerChan, tex->m_textureClass, 0, 0 ,0, 0, 0, true, 0, 0);
	for (int x = 0; x < sizeX * 3; x+= 3)
	{
		for (int y = 0; y < sizeY; y++)
		{
			(*dst)->m_data[y * sizeX * 3 + x + 0] = pattern.at<cv::Vec3b>(y,x/3)[0];
			(*dst)->m_data[y * sizeX * 3 + x + 1] = pattern.at<cv::Vec3b>(y,x/3)[1];
			(*dst)->m_data[y * sizeX * 3 + x + 2] = pattern.at<cv::Vec3b>(y,x/3)[2];
		}
	}	

	//ImageProcessor::showTexture(*dst, "POIMMES");

	return result;
}

void ImageProcessor::calcStats(Texture* t, int numColors)
{
	Statistics* stat = new Statistics(t, numColors);
	t->m_stats = new float[14];
	t->m_stats[0]  = stat->calcASM();
	t->m_stats[1]  = stat->calcContrast();
	t->m_stats[2]  = stat->calcCorrelation();
	t->m_stats[3]  = stat->calcSumOfSquares();
	t->m_stats[4]  = stat->calcInverseDifference();
	t->m_stats[5]  = stat->calcSumAvg();
	t->m_stats[6]  = stat->calcSumEntropy();
	t->m_stats[7]  = stat->calcSumVariance();
	t->m_stats[8]  = stat->calcEntropy();
	t->m_stats[9]  = stat->calcDifferenceVariance();
	t->m_stats[10] = stat->calcDifferenceEntropy();
	t->m_stats[11] = stat->calcInformationMeasures1();
	t->m_stats[12] = stat->calcInformationMeasures2();
	t->m_stats[13] = stat->calcMaxCorrelationCoefficient();
	delete stat;
}

void ImageProcessor::calcCCV(Texture* t, int numColors, int coherenceThreshold)
{
	if (t->m_width >= 8 && t->m_height >= 8)
	{
		CCV* ccv = new CCV(t, numColors, coherenceThreshold);
		t->m_numCCVColors = numColors;
		t->m_CCV = new unsigned long [numColors * 3 * 2];
		ccv->toArray_r(t->m_CCV);
		ccv->toArray_g(&(t->m_CCV[numColors * 2]));
		ccv->toArray_b(&(t->m_CCV[numColors * 2 * 2]));
		delete ccv;
	}
}

float ImageProcessor::compareTexturesHist(Texture* tex1, Texture* tex2)
{
	if (tex1->m_numCCVColors == tex2->m_numCCVColors)
	{
		float result = 0;

		//r, g and b
		for (int i = 0; i < tex1->m_numCCVColors * 2 * 3; i += 2)
		{
			//calculate histogram entries from the CCVs
			int col1 = tex1->m_CCV[i + 0] + tex1->m_CCV[i + 1];
			int col2 = tex2->m_CCV[i + 0] + tex2->m_CCV[i + 1];
			
			//distance between relative values
			result += fabs( col1 * 1.0f / (tex1->m_width*tex1->m_height) - col2 * 1.0f / (tex2->m_width*tex2->m_height)); 
		}

		return result;
	}
	else
	{
		return FLT_MAX;	
	}
	
}

float ImageProcessor::compareTexturesCCV(Texture* tex1, Texture* tex2)
{
	CCV* ccv1 = new CCV(tex1);
	CCV* ccv2 = new CCV(tex2);
	return ccv1->compareTo(ccv2);
}

float ImageProcessor::compareTexturesStats(Texture* tex1, Texture* tex2)
{
	return Statistics::textureVectorDistance(tex1->m_stats, tex2->m_stats);
}


void ImageProcessor::showTexture(Texture* t, string caption)
{
	cv::Mat img(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);

	//cv::putText(img, caption, cv::Point2f(0,img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 2);

	cv::startWindowThread();
	
	//show the reference image
	cv::namedWindow(caption, CV_WINDOW_AUTOSIZE);
	cv::imshow(caption, img);
	cv::waitKey();

	cv::destroyAllWindows();
}

void ImageProcessor::showTexture(cv::Mat img, string caption)
{

	//cv::putText(img, caption, cv::Point2f(0,img.rows/2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,0,0), 2);

	cv::startWindowThread();
	
	//show the reference image
	cv::namedWindow(caption, CV_WINDOW_AUTOSIZE);
	cv::imshow(caption, img);
	cv::waitKey();

	cv::destroyAllWindows();
}
}
