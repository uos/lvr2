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

/**
 * Main.cpp
 *
 *  Created on: Aug 9, 2013
 *      Author: Thomas Wiemann
 */


#include "Options.hpp"
#include "lvr2/io/ScanDirectoryParser.hpp"

using namespace lvr2;

const kaboom::Options* options;

int main(int argc, char** argv) {

    // Parse command line arguments
    kaboom::Options options(argc, argv);

    ScanDirectoryParser parser(options.getInputDir());
    parser.setStart(options.getStart());
    parser.setEnd(options.getEnd());
    parser.setPointCloudPrefix(options.getScanPrefix());
    parser.setPosePrefix(options.getPosePrefix());
    parser.setPointCloudExtension(options.getScanExtension());
    parser.setPoseExtension(options.getPoseExtension());
    parser.parseDirectory();

    if(options.getTargetSize() && options.getVoxelSize())
    {
        std::cout << timestamp << "Warning: Octree reduction and random reduction requested." << std::endl;
        std::cout << timestamp << "Please chose set either octree voxel size with -v or target " << std::endl;
        std::cout << timestamp << "size with random reduction using --targetSize." << std::endl;
        return 0;
    }

    if(options.getTargetSize())
    {
        PointBufferPtr result = parser.randomSubSample(options.getTargetSize());
    }
    else
    {
        PointBufferPtr result = parser.octreeSubSample(options.getVoxelSize(), options.getMinPointsPerVoxel());
    }

    return 0;
}
