/*******************************************************************************
 * Copyright © 2012 Universität Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA  02111-1307, USA
 ******************************************************************************/


/**
 * @file       statstrain.cpp
 * @brief      Program to determine coefficients for statistical texture matching
 * @details    
 * @author     Kim Oliver Rinnewitz (krinnewitz), krinnewitz@uos.de
 * @version    120108
 * @date       Created:       2012-07-21 02:49:26
 * @date       Last modified: 2012-07-21 02:49:30
 */

#include <iostream>
#include <io/Timestamp.hpp>
#include <io/TextureIO.hpp>
#include <geometry/Texture.hpp>
#include <geometry/ImageProcessor.hpp>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

/**
 * \brief Main entry point of the program.
**/
int main( int argc, char ** argv )
{

	if (argc != 2)
	{
		cout<<"Usage: "<<argv[0]<<" <filename>"<<endl;
		return EXIT_FAILURE;
	}

	cout<<"Welcome to statstrain - matching textures with a passion!"<<endl;
	cout<<"------------------------------------------------"<<endl;
	lssr::TextureIO* tio = new lssr::TextureIO(argv[1]);

	//Generate training data randomly
	vector<lssr::Texture*> tData;
	for (int i = 0; i < tio->m_textures.size(); i++)
	{
		//choose random size for subrect
		int width  = rand() % tio->m_textures[i]->m_width  + 20;
		int height = rand() % tio->m_textures[i]->m_height + 20;

		//choose a random center for the subrect
		int cx = rand() % (tio->m_textures[i]->m_width  - 10) + 10;
		int cy = rand() % (tio->m_textures[i]->m_height - 10) + 10;

		//extract the subrect
		cv::Mat img(cv::Size(tio->m_textures[i]->m_width, tio->m_textures[i]->m_height), CV_MAKETYPE(tio->m_textures[i]->m_numBytesPerChan * 8, tio->m_textures[i]->m_numChannels), tio->m_textures[i]->m_data);
		cv::Mat dst;
		cv::Size size(width, height);
		cv::Point2f center(cx, cy);
		cv::getRectSubPix(img, size, center, dst);
		unsigned char depth = dst.depth() == CV_8U ? 1 : 2;
		lssr::Texture* t = new lssr::Texture(dst.size().width, dst.size().height, dst.channels(), depth, 0, 0, 0 ,0, 0, true, 0, 0);
		memcpy(t->m_data, dst.data, dst.size().width * dst.size().height * dst.channels() * depth);
		//calculate stats for training data
		lssr::ImageProcessor::calcStats(t, 16); //TODO: Param oder member
		tData.push_back(t);
/*
		cv::startWindowThread();
		cv::Mat img2(cv::Size(t->m_width, t->m_height),
			    CV_MAKETYPE(t->m_numBytesPerChan * 8,
			    t->m_numChannels), t->m_data);
		cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
		cv::imshow("MyWindow", img2);
		cv::waitKey();
		cv::destroyAllWindows();
*/
	}	

	//calculate stat diffs
	float*** x = new float**[tio->m_textures.size()];
	for (int i = 0; i < tio->m_textures.size(); i++)
	{
		x[i] = new float*[tio->m_textures.size()]; 
		for (int j = 0; j < tio->m_textures.size(); j++)
		{
			x[i][j] = new float[14];
			for (int k = 0; k < 14; k++)
			{
				x[i][j][k] = fabs(tio->m_textures[i]->m_stats[k] - tData[j]->m_stats[k]);
			}
		}
	}


	float coeffs[14];
	//Write zimpl file
//TODO
	//Solve LP
//TODO
	//write sc.co
	ofstream out("sc.co");
	for (int i = 0; i < 14; i++)
	{
		out<<coeffs[i];
		if (i < 13) 
		{
			out<<" ";

		}
	}
	out.close();

	delete tio;
	return EXIT_SUCCESS;

}
