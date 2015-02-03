/*******************************************************************************
 * Copyright © 2011 Universität Osnabrück
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
 * @file       convert.cpp
 * @brief      Converts meshes and point clouds from one file format to another.
 * @details    
 * @author     Thomas Wiemann, lkiesow@uos.de
 */

#include <iostream>
#include "io/ModelFactory.hpp"
#include "io/Timestamp.hpp"
#include "display/ColorMap.hpp"
#include "Options.hpp"

using namespace lvr;
using namespace std;

int main( int argc, char ** argv )
{

    // Parse command line arguments
    convert::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if ( options.printUsage() )
    {
        return 0;
    }

    ::std::cout << options << ::std::endl;

    // Read input file
    ModelPtr model = ModelFactory::readModel(options.getInputFile());

    if(model)
    {

        if(model->m_pointCloud)
        {
            // Get parameters
            bool convert           = options.convertIntensity();
            bool filterIntensity   = options.getMinIntensity() > -1e10 || options.getMaxIntensity() < 1e10;

            // Check if we have to modify some data
            if(convert || filterIntensity)
            {
                PointBufferPtr points = model->m_pointCloud;
                size_t numPoints = points->getNumPoints();
                if(points)
                {
                    size_t n;
                    if(convert && !points->getPointColorArray(n))
                    {
                        cout << timestamp << "Allocating new point color array." << endl;
                        ucharArr colors(new unsigned char[numPoints * 3]);
                        points->setPointColorArray(colors, numPoints);
                    }

                    float max_i = -1e10;
                    float min_i = +1e10;
                    float maxCutOff = options.getMaxIntensity();
                    float minCutOff = options.getMinIntensity();

                    floatArr intensities = points->getPointIntensityArray(n);
                    if(intensities)
                    {
                        // Cutoff intensity values
                        for(size_t i = 0; i < numPoints; i++)
                        {
                            //cout << intensities[i] << endl;
                            if(intensities[i] < minCutOff) intensities[i] = minCutOff;
                            if(intensities[i] > maxCutOff) intensities[i] = maxCutOff;
                            if(intensities[i] < min_i) min_i = intensities[i];
                            if(intensities[i] > max_i) max_i = intensities[i];
                        }

                        // Map intensities to 0 .. 255
                        cout << max_i << " " << min_i << endl;
                        float r_diff = max_i - min_i;
                        if(r_diff > 0)
                        {
                            float b_size = r_diff / 255.0;
                            for(int a = 0; a < numPoints; a++)
                            {
                                //cout << intensities[a] << endl;
                                float value = intensities[a];
                                value -= min_i;
                                value /= b_size;
                                intensities[a] = value;
                                //cout << intensities[a] << endl << endl;
                            }
                        }

                        // Convert
                        if(convert)
                        {
                            ucharArr colors = points->getPointColorArray(n);

                            GradientType gradType = GREY;
                            string gradientName = options.getColorMap();

                            if(gradientName == "HOT")   gradType = HOT;
                            if(gradientName == "HSV")   gradType = HSV;
                            if(gradientName == "SHSV")  gradType = SHSV;
                            if(gradientName == "JET")   gradType = JET;

                            ColorMap colorMap(255);

                            for(size_t i = 0; i < numPoints; i++)
                            {
                                float color[3];

                                colorMap.getColor(color, (size_t)intensities[i], gradType );

                                colors[3 * i    ] = (unsigned char)(color[0] * 255);
                                colors[3 * i + 1] = (unsigned char)(color[1] * 255);
                                colors[3 * i + 2] = (unsigned char)(color[2] * 255);
                                //cout << intensities[i] << endl;
                                //cout << (int) colors[3 * i    ] << " " << (int)colors[3 * i + 1] << " " << (int)colors[3 * i + 2] << endl;

                            }

                            points->setPointColorArray(colors, numPoints);
                        }
                        model->m_pointCloud = points;
                    }
                    else
                    {
                        cout << timestamp << "Model contains no point intensities to convert." << endl;
                    }

                }
            }
        }
        ModelFactory::saveModel(model, options.getOutputFile());

    }
    else
    {
        cout << timestamp << "Error reading file " << options.getInputFile() << endl;
    }
}
