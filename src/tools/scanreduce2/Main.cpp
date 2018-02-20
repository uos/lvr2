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

#include <lvr/io/ModelFactory.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>      
#include <boost/shared_ptr.hpp>
using namespace lvr;

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    // Parse command line arguments
    reduce::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if (options.printUsage()) return 0;

    std::cout << options << std::endl;

    std::string path = options.inputFile()[0];

    ModelPtr m = lvr::ModelFactory::readModel(path);
    PointBufferPtr pBuffer = m->m_pointCloud;

    size_t numPoints, numNormals, numColors;

    coord3fArr points = pBuffer->getIndexedPointArray(numPoints);
    coord3fArr normals = pBuffer->getIndexedPointNormalArray(numNormals);
    color3bArr colors = pBuffer->getIndexedPointColorArray(numColors);
    size_t newSize;
    if(options.reduction() > 1)
    {
        newSize = numPoints/options.reduction();

    }
    else
    {
        newSize = options.points();
    }

    vector<size_t> oldToNew(numPoints);
    std::iota(std::begin(oldToNew), std::end(oldToNew), 0);
    if(options.reduction() > 1)
    {
        
    }
    else
    {
        std::random_shuffle(oldToNew.begin(), oldToNew.end());
    }




    for(size_t i = 0 ; i< newSize; i++)
    {
        std::swap(points[i],points[oldToNew[i]]);
        if(numNormals == numPoints)
        {
            std::swap(normals[i],normals[oldToNew[i]]);
        }
        if(numColors == numPoints)
        {
            std::swap(colors[i],colors[oldToNew[i]]);
        }
    }

    ModelPtr newModel(new Model);
    PointBufferPtr newPBuffer(new PointBuffer);
    newPBuffer->setIndexedPointArray(points,newSize);
    if(numNormals == numPoints)
    {
        newPBuffer->setIndexedPointNormalArray(normals,newSize);
    }
    if(numColors == numPoints)
    {
        newPBuffer->setIndexedPointColorArray(colors,newSize);
    }
    newModel->m_pointCloud = newPBuffer;

    ModelFactory::saveModel(newModel, options.outputFile());

	return 0;
}

