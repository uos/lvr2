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

// Program options for this tool


#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/reconstruction/ModelToImage.hpp"
#include "lvr2/reconstruction/PanoramaNormals.hpp"
#include "Options.hpp"
using namespace lvr2;


/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
    image_normals::Options opt(argc, argv);
    cout << opt << endl;

    ModelPtr model = ModelFactory::readModel(opt.inputFile());

    // Determine coordinate system
    ModelToImage::CoordinateSystem system = ModelToImage::NATIVE;

    if(opt.coordinateSystem() == "SLAM6D")
    {
        system = ModelToImage::SLAM6D;
    }
    else if(opt.coordinateSystem() == "UOS")
    {
        system = ModelToImage::UOS;
    }

    ModelToImage mti(
                model->m_pointCloud,
                ModelToImage::CYLINDRICAL,
                opt.imageWidth(), opt.imageHeight(),
                opt.minZ(), opt.maxZ(),
                opt.minH(), opt.maxH(),
                opt.minV(), opt.maxV(),
                opt.optimize(), system);

    mti.writePGM(opt.imageFile(), 3000);

    PanoramaNormals normals(&mti);
    PointBufferPtr buffer = normals.computeNormals(opt.regionWidth(), opt.regionHeight(), false);

    ModelPtr out_model(new Model(buffer));

    ModelFactory::saveModel(out_model, "normals.ply");
}

