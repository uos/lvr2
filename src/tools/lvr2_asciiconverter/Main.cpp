/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Options.hpp"

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/AsciiIO.hpp"
#include "lvr2/io/DataStruct.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Progress.hpp"

#include <iostream>
#include <fstream>

using namespace lvr2;


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
    std::cout << timestamp << "Read intensities\t\t: " << readIntensity << std::endl;
    std::cout << timestamp << "Convert intensities\t: " << convert << std::endl;

    // Alloc buffers
    floatArr points(new float[3 * numPoints]);
    ucharArr colors;
    floatArr intensities;

    if(readColor || (readIntensity && convert))
    {
        colors = ucharArr(new unsigned char[3 * numPoints]);
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
        float* data = new float[numEntries];
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
                colors[posPtr    ] = (unsigned char)data[options.i()];
                colors[posPtr + 1] = (unsigned char)data[options.i()];
                colors[posPtr + 2] = (unsigned char)data[options.i()];
            }
            else if (readColor)
            {
                colors[posPtr    ] = (unsigned char)data[options.r()];
                colors[posPtr + 1] = (unsigned char)data[options.g()];
                colors[posPtr + 2] = (unsigned char)data[options.b()];
            }

            if(readIntensity)
            {
                intensities[c] = data[options.i()];
            }
            c++;
            ++progress;
        }
		delete[] data;

        // Create model and save data
        PointBufferPtr pointBuffer(new PointBuffer );
        pointBuffer->setPointArray(points, numPoints);
        pointBuffer->setColorArray(colors, numPoints);
        pointBuffer->addFloatChannel(intensities, "intensities", numPoints, 1);

        ModelPtr model( new Model(pointBuffer));
        ModelFactory::saveModel(model, outputFile);
    }
    else
    {
        std::cout << "Unable to open file for output: " << inputFile << std::endl;
    }

    std::cout << std::endl;

	return 0;
}

