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


/**
 * IOFactory.cpp
 *
 *  @date 24.08.2011
 *  @author Thomas Wiemann
 */

#include <lvr2/io/AsciiIO.hpp>
#include <lvr2/io/PLYIO.hpp>
//#include <lvr2/io/Hdf5IO.hpp>
#include <lvr2/io/UosIO.hpp>
//#include <lvr2/io/ObjIO.hpp>
#include <lvr2/io/LasIO.hpp>
#include <lvr2/io/BoctreeIO.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/io/DatIO.hpp>
#include <lvr2/io/STLIO.hpp>

#include <lvr2/io/Timestamp.hpp>
#include <lvr2/io/Progress.hpp>

// PCL related includes
#ifdef LVR_USE_PCL
#include <lvr2/io/PCDIO.hpp>
#endif

#include <boost/filesystem.hpp>

namespace lvr2
{

CoordinateTransform ModelFactory::m_transform;

ModelPtr ModelFactory::readModel( std::string filename )
{
    ModelPtr m;

    // Check extension
    boost::filesystem::path selectedFile( filename );
    std::string extension = selectedFile.extension().string();

    // Try to parse given file
    BaseIO* io = 0;
    if(extension == ".ply")
    {
        io = new PLYIO;
    }
    else if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        io = new AsciiIO;
    }
    else if (extension == ".obj")
    {
        //io = new ObjIO;
    }
    else if (extension == ".las")
    {
        io = new LasIO;
    }
    else if (extension ==".dat")
    {
        io = new DatIO;
    }
    else if (extension == ".h5")
    {
        //io = new Hdf5IO;
    }
#ifdef LVR_USE_PCL
    else if (extension == ".pcd")
    {
        io = new PCDIO;
    }
#endif /* LVR_USE_PCL */
    else if (extension == "")
    {
        bool found_3d = false;
        bool found_boctree = false;

        // Check for supported data in directory.
        boost::filesystem::directory_iterator lastFile;

        for(boost::filesystem::directory_iterator it(filename); it != lastFile; it++ )
        {
            boost::filesystem::path p = it->path();

            // Check for 3d files
            if(p.extension().string() == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
                {
                    found_3d = true;
                }
            }

            // Check for .oct files
            if(p.extension().string() == ".oct")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
                {
                    found_boctree = true;
                }
            }


        }

        // Check and create io
        if(!found_boctree && found_3d)
        {
            io = new UosIO;
        }
        else if(found_boctree && found_3d)
        {
            cout << timestamp << "Found 3d files and octrees. Loading octrees per default." << endl;
            io = new BoctreeIO;
        }
        else if(found_boctree && !found_3d)
        {
            io = new BoctreeIO;
        }
        else
        {
            cout << timestamp << "Given directory does not contain " << endl;
        }
    }

    // Return data model
    if( io )
    {
        m = io->read( filename );

        if(m_transform.convert)
        {
            // Convert coordinates in model
            PointBuffer2Ptr points = m->m_pointCloud;
            size_t n_points = points->numPoints();
            size_t n_normals = 0;
            unsigned dummy;

            floatArr p = points->getPointArray();
            floatArr n = points->getFloatArray("normals", n_normals, dummy);

            // If normals are present every point should habe one
            if(n_normals)
            {
                assert(n_normals == n_points);
            }

            // Convert coordinates
            float point[3];
            float normal[3];

            for(size_t i = 0; i < n_points; i++)
            {
                // Re-order and scale point coordinates
                point[0] = p[3 * i + m_transform.x] * m_transform.sx;
                point[1] = p[3 * i + m_transform.y] * m_transform.sy;
                point[2] = p[3 * i + m_transform.z] * m_transform.sz;

                p[3 * i]         = point[0];
                p[3 * i + 1]    = point[1];
                p[3 * i + 2]    = point[2];
                if(n_normals)
                {
                    normal[0] = n[3 * i + m_transform.x] * m_transform.sx;
                    normal[1] = n[3 * i + m_transform.y] * m_transform.sy;
                    normal[2] = n[3 * i + m_transform.z] * m_transform.sz;

                    n[3 * i]         = normal[0];
                    n[3 * i + 1]    = normal[1];
                    n[3 * i + 2]    = normal[2];
                }
            }
        }

        delete io;
    }

    return m;

}

void ModelFactory::saveModel( ModelPtr m, std::string filename)
{
    // Get file exptension
    boost::filesystem::path selectedFile(filename);
    std::string extension = selectedFile.extension().string();

    BaseIO* io = 0;

    // Create suitable io
    if(extension == ".ply")
    {
        io = new PLYIO;
    }
    else if (extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        io = new AsciiIO;
    }
    else if ( extension == ".obj" )
    {
        //io = new ObjIO;
    }
    else if (extension == ".stl")
    {
        io = new STLIO;
    }
    else if (extension == ".h5")
    {
        //io = new Hdf5IO;
    }
#ifdef LVR_USE_PCL
    else if (extension == ".pcd")
    {
        io = new PCDIO;
    }
#endif

    // Save model
    if(io)
    {
        io->save( m, filename );
        delete io;
    }
    else
    {
        cout << timestamp << "File format " << extension
            << " is currently not supported." << endl;
    }



}

} // namespace lvr2
