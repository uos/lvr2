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

#include "io/AsciiIO.hpp"
#include "io/PLYIO.hpp"
#include "io/UosIO.hpp"
#include "io/ObjIO.hpp"
#include "io/LasIO.hpp"
#include "io/BoctreeIO.hpp"
#include "io/ModelFactory.hpp"

#include "io/Timestamp.hpp"
#include "io/Progress.hpp"

// PCL related includes
#ifdef _USE_PCL_
#include "io/PCDIO.hpp"
#endif

#include <boost/filesystem.hpp>

namespace lvr
{

ModelPtr ModelFactory::readModel( std::string filename )
{
    ModelPtr m;

    // Check extension
    boost::filesystem::path selectedFile( filename );
    std::string extension = selectedFile.extension().c_str();

    // Try to parse given file
    BaseIO* io = 0;
    if(extension == ".ply")
    {
        io = new PLYIO;
    }
    else if(extension == ".pts" || extension == ".3d" || extension == ".xyz")
    {
        io = new AsciiIO;
    }
    else if (extension == ".obj")
    {
        io = new ObjIO;
    }
    else if (extension == ".las")
    {
        io = new LasIO;
    }
#ifdef _USE_PCL_
    else if (extension == ".pcd")
    {
        io = new PCDIO;
    }
#endif /* _USE_PCL_ */
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
            if(string(p.extension().c_str()) == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().c_str(), "scan%3d", &num))
                {
                    found_3d = true;
                }
            }

            // Check for .oct files
            if(string(p.extension().c_str()) == ".oct")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().c_str(), "scan%3d", &num))
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
        delete io;
    }

    return m;

}

void ModelFactory::saveModel( ModelPtr m, std::string filename)
{
    // Get file exptension
    boost::filesystem::path selectedFile(filename);
    std::string extension = selectedFile.extension().c_str();

    BaseIO* io = 0;

    // Create suitable io
    if(extension == ".ply")
    {
        io = new PLYIO;
    }
    else if (extension == ".pts" || extension == ".3d" || extension == ".xyz")
    {
        io = new AsciiIO;
    }
    else if ( extension == ".obj" )
    {
        io = new ObjIO;
    }
#ifdef _USE_PCL_
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

}
