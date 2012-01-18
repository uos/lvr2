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
#include "io/AsciiIO.hpp"
#include "io/DataStruct.hpp"
#include "io/ModelFactory.hpp"
#include "io/Progress.hpp"

#include <iostream>
#include <fstream>

using namespace lssr;


/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    ascii_convert::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage()) return 0;

    std::cout << options << std::endl;

    // Get input and putput files
    string inputFile = options.inputFile();
    string outputFile = options.outputFile();

    // Count entries
    int numEntries = AsciiIO::getEntriesInLine(inputFile);
    int numLines = AsciiIO::countLines(inputFile);
    int numPoints = numLines - 1;

    if(numPoints <= 0)
    {
        std::cout << timestamp << "File contains no points. Exiting." << std::endl;
    }

    // Check color and intensity options
    bool readColor = true;
    if( (options.r() < 0) || (options.g() < 0) || (options.b() < 0) )
    {
        readColor = false;
    }

    bool readIntensity = options.i() >= 0;
    bool convert = options.convertRemission();

    // Print stats
    std::cout << timestamp << "Read colors\t\t: " << readColor << std::endl;
    std::cout << timestamp << "Read intensities\t\t: " << readColor << std::endl;
    std::cout << timestamp << "Convert intensities\t: " << readColor << std::endl;

    // Alloc buffers
    floatArr points(new float[3 * numPoints]);
    ucharArr colors;
    floatArr intensities;

    if(readColor || (readIntensity && convert))
    {
        colors = ucharArr(new uchar[3 * numPoints]);
    }

    if(readIntensity)
    {
        intensities = floatArr(new float[numPoints]);
    }

    // Open input file and skip first line

    string comment = timestamp.getElapsedTime() + "Reading file " + inputFile;
    ProgressBar progress(numPoints, comment);

    std::ifstream in(inputFile.c_str());
    char buffer[2048];
    if(in.good())
    {
        // Data to store the input data
        float data[numEntries];
        in.getline(buffer, 2048);
        int c = 0;
        while(in.good() && c <= numPoints)
        {
            // Read a new line
            for(int i = 0; i < numEntries; i++)
            {
                in >> data[i];
            }

            // Fill data arrays
            int posPtr = 3 * c;

            points[posPtr    ] = data[options.x()];
            points[posPtr + 1] = data[options.y()];
            points[posPtr + 2] = data[options.z()];

            points[posPtr    ] *= options.sx();
            points[posPtr + 1] *= options.sy();
            points[posPtr + 2] *= options.sz();

            if(convert)
            {
                colors[posPtr    ] = (uchar)data[options.i()];
                colors[posPtr + 1] = (uchar)data[options.i()];
                colors[posPtr + 2] = (uchar)data[options.i()];
            }
            else if (readColor)
            {
                colors[posPtr    ] = (uchar)data[options.r()];
                colors[posPtr + 1] = (uchar)data[options.g()];
                colors[posPtr + 2] = (uchar)data[options.b()];
            }

            if(readIntensity)
            {
                intensities[c] = data[options.i()];
            }
            c++;
            ++progress;
        }

        // Create model and save data
        PointBufferPtr pointBuffer(new PointBuffer);
        pointBuffer->setPointArray(points, numPoints);
        pointBuffer->setPointColorArray(colors, numPoints);
        pointBuffer->setPointIntensityArray(intensities, numPoints);

        ModelPtr model( new Model(pointBuffer));
        ModelFactory::saveModel(model, outputFile);
    }
    else
    {
        std::cout << "Unable to open file for output: " << inputFile << std::endl;
    }

	return 0;
}

