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
 * calcNormalsCuda.h
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#include <boost/filesystem.hpp>

#include "lvr2/reconstruction/opencl/ClSurface.hpp"

#include "lvr2/geometry/HalfEdgeMesh.hpp"

#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/reconstruction/PointsetGrid.hpp"
#include "lvr2/reconstruction/AdaptiveKSearchSurface.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/reconstruction/BilinearFastBox.hpp"

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "Options.hpp"


using namespace lvr2;

void computeNormals(string filename, cl_normals::Options& opt, PointBufferPtr& buffer)
{
    ModelPtr model = ModelFactory::readModel(filename);
    size_t num_points;

    floatArr points;
    if (model && model->m_pointCloud )
    {
        num_points = model->m_pointCloud->numPoints();
        points = model->m_pointCloud->getPointArray();
        cout << timestamp << "Read " << num_points << " points from " << filename << endl;
    }
    else
    {
        cout << timestamp << "Warning: No point cloud data found in " << filename << endl;
        return;
    }

    floatArr normals = floatArr(new float[ num_points * 3 ]);

    cout << timestamp << "Constructing kd-tree..." << endl;
    ClSurface gpu_surface(points, num_points);
    cout << timestamp << "Finished kd-tree construction." << endl;

    gpu_surface.setKn(opt.kn());
    gpu_surface.setKi(opt.ki());

    if(opt.useRansac())
    {
        gpu_surface.setMethod("RANSAC");
    } else
    {
        gpu_surface.setMethod("PCA");
    }
    gpu_surface.setFlippoint(opt.flipx(), opt.flipy(), opt.flipz());

    cout << timestamp << "Start Normal Calculation..." << endl;
    gpu_surface.calculateNormals();

    gpu_surface.getNormals(normals);
    cout << timestamp << "Finished Normal Calculation. " << endl;

    buffer->setPointArray(points, num_points);
    buffer->setNormalArray(normals, num_points);

    gpu_surface.freeGPU();
}

void reconstructAndSave(PointBufferPtr& buffer, cl_normals::Options& opt)
{
    // RECONSTRUCTION
    // PointsetSurface
    PointsetSurfacePtr<Vec> surface;
    surface = PointsetSurfacePtr<Vec>( new AdaptiveKSearchSurface<Vec>(buffer,
        "FLANN",
        opt.kn(),
        opt.ki(),
        opt.kd(),
        1
    ) );


    // // Create an empty mesh
    HalfEdgeMesh<Vec> mesh;

    float resolution = opt.getVoxelsize();

    auto grid = std::make_shared<PointsetGrid<Vec, BilinearFastBox<Vec>>>(
        resolution,
        surface,
        surface->getBoundingBox(),
        true,
        true
    );

    grid->calcDistanceValues();

    auto reconstruction = make_unique<FastReconstruction<Vec, BilinearFastBox<Vec>>>(grid);
    reconstruction->getMesh(mesh);


    SimpleFinalizer<Vec> fin;
    MeshBufferPtr res = fin.apply(mesh);

    ModelPtr m( new Model( res ) );

    cout << timestamp << "Saving mesh." << endl;
    ModelFactory::saveModel( m, "triangle_mesh.ply");
}

int main(int argc, char** argv){


    cl_normals::Options opt(argc, argv);
    cout << opt << endl;


    boost::filesystem::path inFile(opt.inputFile());

    if(boost::filesystem::is_directory(inFile))
    {
        vector<float> all_points;
        vector<float> all_normals;

        boost::filesystem::directory_iterator lastFile;
        for(boost::filesystem::directory_iterator it(inFile); it != lastFile; it++ )
        {
            boost::filesystem::path p = boost::filesystem::canonical(it->path());
            string currentFile = p.filename().string();

            if(string(p.extension().string().c_str()) == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(currentFile.c_str(), "scan%3d", &num))
                {
                    cout << timestamp << "Processing " << p.string() << endl;
                    PointBufferPtr buffer(new PointBuffer);

                    computeNormals(p.string(), opt, buffer);
                    transformPointCloudAndAppend(buffer, p, all_points, all_normals);

                }
            }
        }

        if(!opt.reconstruct() || opt.exportPointNormals() )
        {
            writePointsAndNormals(all_points, all_normals, opt.outputFile());
        }

        if(opt.reconstruct() )
        {
            PointBufferPtr buffer(new PointBuffer);

            floatArr points = floatArr(&all_points[0]);
            size_t num_points = all_points.size() / 3;

            floatArr normals = floatArr(&all_normals[0]);
            size_t num_normals = all_normals.size() / 3;

            buffer->setPointArray(points, num_points);
            buffer->setNormalArray(normals, num_normals);

            reconstructAndSave(buffer, opt);
        }

    }
    else
    {
        PointBufferPtr buffer(new PointBuffer);
        computeNormals(opt.inputFile(), opt, buffer);

        if(!opt.reconstruct() || opt.exportPointNormals() )
        {
            ModelPtr out_model(new Model(buffer));
            ModelFactory::saveModel(out_model, opt.outputFile());
        }

        if(opt.reconstruct())
        {
            reconstructAndSave(buffer, opt);
        }

    }


}
