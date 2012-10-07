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
#include <texture/Texture.hpp>
#include <texture/ImageProcessor.hpp>
#include <texture/Statistics.hpp>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ext/soplex/src/soplex.h>

using namespace std;
using namespace soplex;

/**
 * \brief Main entry point of the program.
**/
int main( int argc, char ** argv )
{

	float S1 = 15000; 
	float S2 = 15;

	if (argc != 5)
	{
		cout<<"Usage: "<<argv[0]<<" <filename> <S1> <S2> <number of colors>"<<endl;
		return EXIT_FAILURE;
	}
	S1 = atof(argv[2]);
	S2 = atof(argv[3]);
	int numStatsColors = atoi(argv[4]);

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
		lssr::Texture* t = new lssr::Texture(dst.size().width, dst.size().height, dst.channels(), depth, 0, 0, 0 ,0, 0, 0, true, 0, 0);
		memcpy(t->m_data, dst.data, dst.size().width * dst.size().height * dst.channels() * depth);
		//calculate stats for training data
		lssr::ImageProcessor::calcStats(t, numStatsColors);
		tData.push_back(t);
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


	

	//Formulate the LP
	SoPlex mysoplex;

	//set the objective sense
	mysoplex.changeSense(SPxLP::MAXIMIZE);

	// we first add the 14 variables
	DSVector dummycol(0);
	for (int k = 0; k < 14; k++)
	{
		float v = 0;
		for (int i = 0; i < tio->m_textures.size(); i++)
		{
			for (int j = 0; j < tio->m_textures.size(); j++)
			{
				v += x[i][j][k];
			}
		}
		//Add variable
		mysoplex.addCol(LPCol(v, dummycol, S2, 0));
	}

	/* then constraints one by one */
	DSVector row1(14);
	for (int k = 0; k < 14; k++)
	{
		float v = 0;
		for (int i = 0; i < tio->m_textures.size(); i++)
		{
			v += x[i][i][k];
		}
		row1.add(k, v);
	}
	mysoplex.addRow(LPRow(0, row1, S1));

	//Solve LP
	DVector prim(14);
	SPxSolver::Status stat = mysoplex.solve();

	//get solution
	float coeffs[14];
	if( stat == SPxSolver::OPTIMAL )
	{
		mysoplex.getPrimal(prim);
		cout << "LP solved to optimality."<<endl;
		cout << "Primal solution is [";
		for (int k = 0; k < 14; k++)
		{	coeffs[k] = prim[k];
			cout << prim[k] << ", ";
		}
		cout<<"]"<<endl;
	}

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


	//calculate accuracy 
	int hits = 0;
	for (int i = 0; i < tData.size(); i++)
	{
		float min = FLT_MAX;
		int min_index = -1;
		for (int j = 0; j < tio->m_textures.size(); j++)
		{
			float curr = lssr::Statistics::textureVectorDistance(tData[i]->m_stats, tio->m_textures[j]->m_stats, coeffs);
			if (curr < min)
			{
				min = curr;
				min_index = j;
			}
		}
		if (min_index == i)
		{
			hits++;
		}
	}
	cout<<"Accuracy: "<<hits *1.0f /tData.size() * 100 <<"%"<<"     "<<hits<<endl;

	delete tio;

	return EXIT_SUCCESS;

}
