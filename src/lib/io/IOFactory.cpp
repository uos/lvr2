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

#include <boost/filesystem.hpp>

namespace lssr
{

void IOFactory::read( string filename )
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

    // Create objects
    if ( extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt" )
    {
        AsciiIO* a_io = new AsciiIO;
        m_pointLoader = (PointIO*)      a_io;
        m_baseIO      = (BaseIO*)       a_io;
    }
    else if ( extension == ".ply" )
    {
        PLYIO* ply_io = new PLYIO;
        m_pointLoader = (PointIO*)      ply_io;
        m_meshLoader  = (MeshIO*)   ply_io;
        m_baseIO      = (BaseIO*)       ply_io;
    }
    else if ( extension == "" )
    {
        UosIO* uos_io =  new UosIO;
        m_pointLoader = (PointIO*) uos_io;
        m_baseIO      = (BaseIO*)  uos_io;
    }

    if(m_baseIO)
    {
        m_baseIO->read(filename);
    }

}

void IOFactory::save(string filename)
{
    // Get file exptension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

    BaseIO* io;
    // Create suitable io


}

}
