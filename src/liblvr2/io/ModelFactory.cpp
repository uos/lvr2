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
 * IOFactory.cpp
 *
 *  @date 24.08.2011
 *  @author Thomas Wiemann
 */

#include "lvr2/io/modelio/AsciiIO.hpp"
#include "lvr2/io/modelio/PLYIO.hpp"
#include "lvr2/io/modelio/UosIO.hpp"
#include "lvr2/io/modelio/ObjIO.hpp"
#include "lvr2/io/modelio/LasIO.hpp"
#include "lvr2/io/modelio/DatIO.hpp"
#include "lvr2/io/modelio/STLIO.hpp"
#include "lvr2/io/modelio/B3dmIO.hpp"
#include "lvr2/io/modelio/RdbxIO.hpp"

// #include "lvr2/io/HDF5IO.hpp"
// #include "lvr2/io/WaveformIO.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/Progress.hpp"

// PCL related includes
#ifdef LVR2_USE_PCL
#include "lvr2/io/PCDIO.hpp"
#endif

// RiVLib
#ifdef LVR2_USE_RIVLIB
#include "lvr2/io/RxpIO.hpp"
#endif

#include <boost/filesystem.hpp>

namespace lvr2
{

CoordinateTransform<float> ModelFactory::m_transform;

ModelPtr ModelFactory::readModel( std::string filename )
{
    ModelPtr m;

    // Check extension
    boost::filesystem::path selectedFile( filename );
    std::string extension = selectedFile.extension().string();

    // Try to parse given file
    ModelIOBase* io = 0;
    if(extension == ".ply")
    {
        io = new PLYIO;
    }
    else if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        io = new AsciiIO;
    }
    else if(extension == ".rdbx")
    {
        io = new RdbxIO;
    }

#ifdef LVR2_USE_RIVLIB
    else if(extension == ".rxp")
    {
        io = new RxpIO;
    }
#endif
    else if (extension == ".obj")
    {
        io = new ObjIO;
    }
    else if (extension == ".las")
    {
        io = new LasIO;
    }
    else if (extension == ".dat")
    {
        io = new DatIO;
    }
#ifdef LVR2_USE_3DTILES
    else if (extension == ".b3dm")
    {
        io = new B3dmIO;
    }
#endif
    // else if (extension ==".lwf")
    // {
	// io = new WaveformIO;
    // }
#ifdef LVR2_USE_PCL
    else if (extension == ".pcd")
    {
        io = new PCDIO;
    }
#endif /* LVR2_USE_PCL */
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
        }

        // Check and create io
        if(!found_boctree && found_3d)
        {
            io = new UosIO;
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

        if( m_transform.transforms())
        {
            // Convert coordinates in model
            PointBufferPtr points = m->m_pointCloud;
            size_t n_points = points->numPoints();
            size_t n_normals = 0;
            size_t dummy;

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

    ModelIOBase* io = 0;

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
        io = new ObjIO;
    }
    else if (extension == ".stl")
    {
        io = new STLIO;
    }
#ifdef LVR2_USE_3DTILES
    else if (extension == ".b3dm")
    {
        io = new B3dmIO;
    }
#endif
#ifdef LVR2_USE_PCL
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
