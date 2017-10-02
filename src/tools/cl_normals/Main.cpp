/*
 * This file is part of cudaNormals.
 *
 * cudaNormals is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Foobar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cudaNormals.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * calcNormalsCuda.h
 *
 * @author Alexander Mock
 * @author Matthias Greshake
 */

#include <boost/filesystem.hpp>

#include <lvr/reconstruction/opencl/ClSurface.hpp>

#include <lvr/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/reconstruction/PointsetSurface.hpp>
#include <lvr/reconstruction/PointsetGrid.hpp>
#include <lvr/reconstruction/BilinearFastBox.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>

#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/IOUtils.hpp>
#include "Options.hpp"


using namespace lvr;

typedef PointsetSurface<ColorVertex<float, unsigned char> > psSurface;
typedef AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> > akSurface;

void computeNormals(string filename, cl_normals::Options& opt, PointBufferPtr& buffer)
{
    ModelPtr model = ModelFactory::readModel(filename);
    size_t num_points;

    floatArr points;
    if (model && model->m_pointCloud )
    {
        points = model->m_pointCloud->getPointArray(num_points);
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
    buffer->setPointNormalArray(normals, num_points);

    gpu_surface.freeGPU();
}

void reconstructAndSave(PointBufferPtr& buffer, cl_normals::Options& opt)
{
    // RECONSTRUCTION
    // PointsetSurface
    akSurface* aks = new akSurface(
                buffer, "FLANN",
                opt.kn(),
                opt.ki(),
                opt.kd(),
                true
        );

    psSurface::Ptr surface = psSurface::Ptr(aks);

    // // Create an empty mesh
    HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );

    float resolution = opt.getVoxelsize();

    GridBase* grid;
    FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
    BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;

    grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), true);
    PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
    ps_grid->calcDistanceValues();
    reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
    reconstruction->getMesh(mesh);

    mesh.finalize();

    ModelPtr m( new Model( mesh.meshBuffer() ) );

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
            buffer->setPointNormalArray(normals, num_normals);

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
