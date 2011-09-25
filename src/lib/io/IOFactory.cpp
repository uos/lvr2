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

IOFactory::IOFactory(string filename) : m_meshLoader(0), m_pointLoader(0), m_baseIO(0)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension().c_str();

    // Create objects
    if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        AsciiIO* a_io = new AsciiIO;
        m_pointLoader = (PointLoader*)  a_io;
        m_baseIO =      (BaseIO*)       a_io;
    }
    else if(extension == ".ply")
    {
        PLYIO* ply_io = new PLYIO;
        m_pointLoader = (PointLoader*)  ply_io;
        m_meshLoader  = (MeshLoader*)   ply_io;
        m_baseIO      = (BaseIO*)       ply_io;
    }
    else if(extension == "")
    {
        UosIO* uos_io =  new UosIO;
        m_pointLoader = (PointLoader*)  uos_io;
        m_baseIO      = (BaseIO*)       uos_io;
    }

    // Read file data
    if(m_baseIO)
    {
        m_baseIO->read(filename);
    }
}

}
