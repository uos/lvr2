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


#include "Options.hpp"

#include "io/Timestamp.hpp"
#include "io/ModelFactory.hpp"
#include "reconstruction/PCLFiltering.hpp"

#include <iostream>

using namespace lvr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
	try
	{
		filter::Options options(argc, argv);


		// Exit if options had to generate a usage message
		// (this means required parameters are missing)
		if (options.printUsage()) return 0;

		::std::cout<< options<<::std::endl;

		ModelPtr m = ModelFactory::readModel(options.inputFile());

		// Check if Reading was successful
		if(m)
		{
			if(m->m_pointCloud)
			{
				PCLFiltering filter(m->m_pointCloud);

				// Apply filters
				if(options.removeOutliers())
				{
					filter.applyOutlierRemoval(options. sorMeanK(), options.sorDevThreshold());
				}


				if(options.mlsDistance() > 0)
				{
					filter.applyMLSProjection(options.mlsDistance());
				}

				PointBufferPtr pb( filter.getPointBuffer() );
				ModelPtr out_model( new Model( pb ) );

				ModelFactory::saveModel(out_model, options.outputFile());
			}
		}
		else
		{
			cout << timestamp << "Failed to read " << options.inputFile() << endl;
		}
	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'filter --help' for more information." << std::endl;
	}
	return 0;
}

