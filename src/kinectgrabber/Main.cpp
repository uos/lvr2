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
#include <iostream>

#include "io/ModelFactory.hpp"
#include "io/KinectIO.hpp"
#include "io/CoordinateTransform.hpp"

#include "reconstruction/SearchTreeFlann.hpp"

#include "Options.hpp"

#include <vector>
#include <string.h>

using namespace lssr;

#define MEDIAN_COUNT 5

int main(int argc, char** argv)
{

	// Try to connect

	KinectIO* io;
	try
	{
		cout << 1 << endl;
		io = KinectIO::instance();
	}
	catch(...)
	{
		cout << "Kinect connection failed. Try again..." << endl;
		return -1;
	}

	int c = 0;

	vector<PointBufferPtr> scans;
	while(true && c < 20)
	{
		PointBufferPtr buffer = io->getBuffer();
		if(buffer == 0)
		{
			cout << "No data yet..." << endl;
			// Give it some time
			usleep(100000);
		}
		else
		{

			convert(OPENGL_METERS, SLAM6D, buffer);
			usleep(100000);

		}

		c++;
		scans.push_back(buffer);
	}



	for(size_t i = 0; i < scans.size(); i++)
	{
		char fout[256];
		sprintf(fout, "scan%03d.3d", (int)i);
		ModelFactory::saveModel(ModelPtr(new Model(scans[i])), string(fout));

		cout << "Saving " << string(fout) << " with " << scans[i]->getNumPoints() << " points." << endl;
		char pout[256];

		sprintf(pout, "scan%03d.pose", (int)i);
		ofstream out(pout);
		out << "0 0 0 0 0 0" << endl;
		out.close();
	}


	return 0;
}
