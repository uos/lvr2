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
#include "ObjIO.hpp"
#include "ModelFactory.hpp"

#include "Timestamp.hpp"
#include "Progress.hpp"

#include <boost/filesystem.hpp>

namespace lssr
{

Model* ModelFactory::readModel( string filename )
{
    Model* m = 0;

    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

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
        /// TODO: Integrate ObJIO in factory

    }
    else if (extension == "")
    {
        io = new UosIO;
    }

    // Return data model
    if(io)
    {
        m = io->read(filename);
    }

    return m;

}

void ModelFactory::saveModel(Model* m, string filename)
{
    cout << 1 << " " << m->m_pointCloud->getNumPoints() << endl;

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
	 else if ( extension == ".obj" )
	 {
		 io = new ObjIO;
	 }
    else if (extension == "")
    {
        // Try to load UOS format data from directory in
        io = new UosIO;
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
