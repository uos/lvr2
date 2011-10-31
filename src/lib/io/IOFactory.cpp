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

#include "AsciiIO.hpp"
#include "PLYIO.hpp"
#include "UosIO.hpp"
#include "IOFactory.hpp"

#include "Timestamp.hpp"
#include "Progress.hpp"

#include <boost/filesystem.hpp>

namespace lssr
{

Model* IOFactory::readModel( string filename )
{
    Model* m = 0;

    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

    // Try to parse given file
    BaseIO* io = 0;
    if(extension == ".ply")
    {
        cout << "New ply io" << endl;
        io = new PLYIO;
    }
    else if(extension == ".pts" || extension == ".3d" || extension == ".xyz")
    {
        cout << "New ascii io" << endl;
        io = new AsciiIO;
    }
    else if (extension == ".obj")
    {
        /// TODO: Integrate ObJIO in factory
        cout << "New obj io" << endl;
    }

    // Return data model
    if(io)
    {
        cout << "New model" << endl;
        m = io->read(filename);
    }
    cout << "Model pointer: " << m << endl;
    return m;

}

void IOFactory::saveModel(Model* m, string filename)
{
    // Get file exptension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

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

    // Save model
    if(io)
    {
        io->save(m, filename);
    }
    else
    {
        cout << timestamp << "File format " << extension << " is currrently not supported." << endl;
    }



}

}
