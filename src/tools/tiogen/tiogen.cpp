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
 * @file       texman.cpp
 * @brief      Program to manage texture packages.
 * @details    
 * @author     Kim Oliver Rinnewitz (krinnewitz), krinnewitz@uos.de
 * @version    120108
 * @date       Created:       2012-05-02 02:49:26
 * @date       Last modified: 2012-05-02 02:49:30
 */


#include <io/Timestamp.hpp>
#include <io/TextureIO.hpp>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <texture/ImageProcessor.hpp>

#include <iostream>
#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
using namespace std;
using namespace boost::filesystem;


/**
 * \brief Create and add a new texture
 *
 * \param tio	A TextureIO object
**/
void a(string fn, lvr::TextureIO* tio, int numStatsColors, int numCCVColors, int coherenceThreshold)
{

	cv::Mat img = cv::imread(fn.c_str());
	if (img.data)
	{	
		/*cout<<"\t(a)dd: Enter texture class: ";
		unsigned short int tc = 0;
		cin>>tc;
		cout<<"\t(a)dd: Is this texture a pattern texture? (y/n): ";
		char in;
		cin>>in;*/

	    int tc = 0;

		bool isPattern = false;
		unsigned char depth = img.depth() == CV_8U ? 1 : 2;

		//create Texture
		lvr::Texture* t = new lvr::Texture(img.size().width, img.size().height, img.channels(), depth, tc, 0, 0 ,0, 0, 0, isPattern, 0, 0);
		memcpy(t->m_data, img.data, img.size().width * img.size().height * img.channels() * depth);

		// calculate features
		lvr::ImageProcessor::calcSURF(t);

		// calculate stats
		lvr::ImageProcessor::calcStats(t, numStatsColors);

		//calculate CCV
		lvr::ImageProcessor::calcCCV(t, numCCVColors, coherenceThreshold);
		

		cout<<"\t(a)dded new texture."<<endl;
		tio->add(t);
	}
	else
	{
		cout<<"\t(a)dd failed: Could not load texture."<<endl;
	}
}

/**
 * \brief Configure number of colors for statistics and CCV and coherence threshold
 *
 * \param numStatsColors 	The variable to hold the number of colors to use in statistics
 * \param numCCVColors 		The variable to hold the number of colors to use in CCV calculation
 * \param coherenceThreshold 	The variable to hold the coherence threshold to use in CCV calculation
**/
void c(int &numStatsColors, int &numCCVColors, int &coherenceThreshold)
{
	int i= -1;
	cout<<"\tEnter number of colors for statistics (current value: "<<numStatsColors<<"): ";
	cin>>i;
	if (i < 1 || i > 255)
	{
		cout<<"\tInvalid value. Did not change anything."<<endl;
	}
	else
	{
		numStatsColors = i;
	}

	cout<<"\tEnter number of colors for CCVs  (current value: "<<numCCVColors<<"): ";
	cin>>i;
	if (i < 1 || i > 255)
	{
		cout<<"\tInvalid value. Did not change anything."<<endl;
	}
	else
	{
		numCCVColors = i;
	}

	cout<<"\tEnter coherence Threshold for CCV (current value: "<<coherenceThreshold<<"): ";
	cin>>i;
	if (i < 1)
	{
		cout<<"\tInvalid value. Did not change anything."<<endl;
	}
	else
	{
		coherenceThreshold = i;
	}
}

/**
 * \brief delete the texture with the given index
 *
 * \param tio	A TextureIO object
 *
 * \param sel 	The index of the selected texture
**/
void d(lvr::TextureIO* tio, int &sel)
{
	if(sel != -1)
	{
		tio->remove(sel);
		cout<<"\t(d)eleted texture #"<<sel<<"."<<endl; 
		sel = -1;
	}
	else
	{
		cout<<"\t(d)elete failed: Nothing selected."<<endl;
	}
}

/**
 * \brief Show help
**/
void h()
{
	cout<<"\ta: Add a new texture to the file"<<endl;
	cout<<"\tc: Configure parameters for statistics and CCVs"<<endl;
	cout<<"\td: Delete the selected texture"<<endl;
	cout<<"\th: Show this help"<<endl;
	cout<<"\ti: Show file information"<<endl;
	cout<<"\tl: List all textures in the file"<<endl;
	cout<<"\ts: Select a texture" <<endl;
	cout<<"\tu: Update the selected texture"<<endl;
	cout<<"\tv: View the selected texture"<<endl;			
	cout<<"\tw: Write changes to disk"<<endl;		
	cout<<"\tx: Exit"<<endl;
}

/**
 * \brief Show file info
 *
 * \param tio	A TextureIO object
**/
void i(lvr::TextureIO* tio)
{	
	cout<<"\t(i)nfo: "<<tio->m_filename<<" containing "<<tio->m_textures.size()<<" textures."<<endl;
}


/**
 * \brief List all textures
 *
 * \param tio	A TextureIO object
 *
 * \param sel 	The index of the selected texture
**/
void l(lvr::TextureIO* tio, int sel)
{
	cout<<"\t(l)ist of textures:"<<endl;
	cout<<"\t"<<setw(8)<<"index"<<setw(16)<<"WxH"<<setw(10)<<"channels"<<setw(8)<<"depth"<<setw(8)<<"class"<<setw(10)<<"features"<<setw(11)<<"CCVColors";
	cout<<setw(10)<<"selected"<<endl;

	for(int i = 0; i<tio->m_textures.size(); i++)
	{
		lvr::Texture* t = tio->m_textures[i];
		ostringstream wxh; wxh << t->m_width<<"x"<<t->m_height;
		cout<<"\t"<<setw(8)<<i<<setw(16)<< wxh.str();
		cout<<setw(10)<<(unsigned short)t->m_numChannels<<setw(8)<<(unsigned short)t->m_numBytesPerChan;
		cout<<setw(8)<<t->m_textureClass;
		cout<<setw(10)<<t->m_numFeatures;
		cout<<setw(11)<<(unsigned int)t->m_numCCVColors;
		if (sel == i) cout<<setw(10)<<"*";
		cout<<endl;
	}
}

/**
 * \brief View stats for all textures
 *
 * \param tio	A TextureIO object
**/
void e1(lvr::TextureIO* tio)
{
	cout<<"\t(l)ist of textures:"<<endl;
	cout<<"\t"<<setw(8)<<"index";
	for (int i = 0; i < 14; i++)
	{
		if (i < 10)
		{
			cout<<setw(9)<<"S"<<i;
		}
		else 
		{
			cout<<setw(8)<<"S"<<i;
		}
	}
	cout<<endl;

	for(int i = 0; i<tio->m_textures.size(); i++)
	{
		lvr::Texture* t = tio->m_textures[i];
		cout<<"\t"<<setw(8)<<i;
		for (int j = 0; j < 14; j++)
		{
			cout<<setw(10)<<t->m_stats[j];
		}
		cout<<endl;
	}
}

/**
 * \brief View CCVs for all textures
 *
 * \param tio	A TextureIO object
**/
void e2(lvr::TextureIO* tio)
{
	cout<<"\t(l)ist of textures:"<<endl;

	for(int i = 0; i<tio->m_textures.size(); i++)
	{
		cout<<"==============================================="<<"Texture "<<i<<"==============================================="<<endl;
		lvr::Texture* t = tio->m_textures[i];

		cout<<"r:"<<endl;	
		for (int j = 0; j < t->m_numCCVColors; j++)
		{
			cout<<j<<":<"<<t->m_CCV[j * 2 + 0]<<","<<t->m_CCV[j * 2 + 1]<<">,   ";
		}
		cout<<endl;
		cout<<"g:"<<endl;	
		for (int j = t->m_numCCVColors; j < 2 * t->m_numCCVColors; j++)
		{
			cout<<j - t->m_numCCVColors<<":<"<<t->m_CCV[j * 2 + 0]<<","<<t->m_CCV[j * 2 + 1]<<">,   ";
		}
		cout<<endl;
		cout<<"b:"<<endl;	
		for (int j = 2 * t->m_numCCVColors; j < 3 * t->m_numCCVColors; j++)
		{
			cout<<j - 2 * t->m_numCCVColors<<":<"<<t->m_CCV[j * 2 + 0]<<","<<t->m_CCV[j * 2 + 1]<<">,   ";
		}
		cout<<endl;
	}
}

/**
 * \brief select a texture
 *
 * \param tio	A TextureIO object
 *
 * \param sel 	The variable to hold the index of the selected texture
**/
void s(lvr::TextureIO* tio, int &sel)
{
	cout<<"\t(s)elect an index: ";
	int ind = -1;
	cin>>ind;
	if (ind < 0 || ind > tio->m_textures.size()-1)
	{
		cout<<"\tIndex out of bounds. Did not select anything."<<endl;
		sel = -1;
	}
	else
	{
		sel = ind;
		cout<<"\tSelected texture #"<<sel<<"."<<endl;
	}
}

/**
 * \brief Update the texture with the given index
 *
 * \param tio	A TextureIO object
 *
 * \param sel 	The index of the selected texture
**/
void u(lvr::TextureIO* tio, int &sel, int numStatsColors, int numCCVColors, int coherenceThreshold)
{
	if(sel != -1)
	{
		cout<<"\t(u)pdate: Enter path to texture image (<Return> to skip): ";
		char fn[256] = "";
		cin.getline(fn, 256);
		cin.getline(fn, 256);
		if (strlen(fn))
		{
			cv::Mat img = cv::imread(fn);
			if (img.data)
			{
				cout<<"\t(u)pdate: Enter texture class (old: "<<tio->m_textures[sel]->m_textureClass<<"):";
				unsigned short int tc = 0;
				cin>>tc;
				unsigned char depth = img.depth() == CV_8U ? 1 : 2;

				cout<<"\t(a)dd: Is this texture a pattern texture? (y/n): ";
				char in;
				cin>>in;
				bool isPattern = in == 'y' || in == 'Y';

				//create Texture
				lvr::Texture* t = new lvr::Texture(img.size().width, img.size().height, img.channels(), depth, tc, 0, 0 ,0, 0, 0, isPattern, 0, 0);
				memcpy(t->m_data, img.data, img.size().width * img.size().height * img.channels() * depth);

				// calculate features
				lvr::ImageProcessor::calcSURF(t);

				//calculate stats
				lvr::ImageProcessor::calcStats(t, numStatsColors);

				//calculate CCV
				lvr::ImageProcessor::calcCCV(t, numCCVColors, coherenceThreshold);
		
				tio->update(sel, t);
				cout<<"\t(u)dated texture #"<<sel<<"."<<endl; 
			}
			else
			{
				cout<<"\t(u)pdate failed: Could not load new texture."<<endl;
			}
		}
		else
		{
				cout<<"\t(u)pdate: Enter texture class (old: "<<tio->m_textures[sel]->m_textureClass<<"):";
				unsigned short int tc = 0;
				cin>>tc;
				lvr::Texture* t = new lvr::Texture(*(tio->m_textures[sel]));
				t->m_textureClass = tc;
				tio->update(sel, t);
				cout<<"\t(u)dated texture #"<<sel<<"."<<endl; 
		}
	

	}
	else
	{
		cout<<"\t(u)pdate failed: Nothing selected."<<endl;
	}
}
/**
 * \brief View selected texture
 *
 * \param tio	A TextureIO object
 *
 * \param sel 	The index of the selected texture
**/
void v(lvr::TextureIO* tio, int sel)
{
	if (sel != -1)
	{
		cv::startWindowThread();
		cv::Mat img(cv::Size(tio->m_textures[sel]->m_width, tio->m_textures[sel]->m_height),
			    CV_MAKETYPE(tio->m_textures[sel]->m_numBytesPerChan * 8,
			    tio->m_textures[sel]->m_numChannels), tio->m_textures[sel]->m_data);
		cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
		cv::imshow("MyWindow", img);
		cv::waitKey();
		cv::destroyAllWindows();
	}
	else
	{
		cout<<"\t(v)iewing the texture failed: Nothing selected."<<endl;
	}
}

/**
 * \brief Write changes to disk
 *
 * \param tio	A TextureIO object
**/
void w(lvr::TextureIO* tio)
{
	tio->write();
	cout<<"\t(w)rote file."<<endl;
}


/**
 * \brief Main entry point of the program.
**/
int main( int argc, char ** argv )
{

	if (argc != 3)
	{
		cout<<"Usage: "<<argv[0]<<" <filename> <path>"<<endl;
		return EXIT_FAILURE ;
	}

    int numStatsColors = 16;
    int numCCVColors   = 64;
    int coherenceThreshold = 50;

	lvr::TextureIO* tio = new lvr::TextureIO(argv[1]);
	path p (argv[2]);
	try
	{
	    if(exists(p))
	    {
	        if(is_directory(p))
	        {
	            directory_iterator end_itr;
	            for(directory_iterator itr(p); itr != end_itr; itr++)
	            {
	                path file = *itr;
	                if(is_regular_file(file))
	                {
	                    cout << "Adding " << file.string() << "to texture packet" << endl;
	                    a(file.string(), tio, numStatsColors, numCCVColors, coherenceThreshold);
	                }
	                else
	                {
	                    cout << "Error: " << file.string() << " is not a regular file" << endl;
	                }
	            }
	        }
	        else
	        {
	            cout << "Error: " << argv[2] << " is not a adirectory." << endl;
	        }
	    }
	    else
	    {
	        cout << "Error: " << argv[2] << " does not exist." << endl;
	    }
	}
	catch(...)
	{

	}

	w(tio);

	cout<<"\tExiting. Good bye."<<endl;

	delete tio;
	return EXIT_SUCCESS;

}
