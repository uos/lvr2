/**
 * Copyright (C) 2013 Universität Osnabrück
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


// Program options for this tool
#include "Options.hpp"
#include "registration/ICPPointAlign.hpp"
#include "io/DataStruct.hpp"
#include "io/ModelFactory.hpp"

#include <iostream>


using namespace lvr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
	try
	{

	    float euler[6];
		// Parse command line arguments
		registration::Options options(argc, argv);
		cout << options;

		// Load model and data point cloud
		string modelName = options.getModelName();
		string dataName = options.getDataName();

		ModelPtr modelModel = ModelFactory::readModel(modelName);
		ModelPtr dataModel = ModelFactory::readModel(dataName);

		Vertexf position(options.getTx(), options.getTy(), options.getTz());
		Vertexf angle(options.getRx(), options.getRy(), options.getRz());
		Matrix4f transformation(position, angle);

		ICPPointAlign align(modelModel->m_pointCloud, dataModel->m_pointCloud, transformation);
		align.setMaxIterations(options.getMaxIterations());
		align.setMaxMatchDistance(options.getMaxDistance());
		Matrix4f correction = align.match();


		correction.toPostionAngle(euler);

		cout << "Pose: " << euler[0] << " " << euler[1] << " " << euler[2] << " " << euler[3] << " " << euler[4] << " " << euler[5] << endl;

	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'lvr_registration --help' for more information." << std::endl;
	}
	return 0;
}

