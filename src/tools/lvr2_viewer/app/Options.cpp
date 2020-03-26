#include "Options.hpp"

#include <iostream>
#include "lvr2/config/BaseOption.hpp"

namespace viewer 
{

using boost::program_options::command_line_parser;
using boost::program_options::value;
using std::cout;
using std::endl;

Options::Options(int argc, char** argv) : lvr2::BaseOption(argc, argv)
{
    // Create option descriptions
    m_descr.add_options()("help",
                          "Produce help message")
        ("chunkedMesh", boost::program_options::bool_switch()->default_value(false), "Is a chunked Mesh")
        ("inputFile", value< std::vector<string> >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
        ("layers", value<std::vector<string>>()->multitoken(), "Input file names.")
        ("cacheSize", value<int>()->default_value(200), "Multilayer chunked mesh the maximum number of high resolution chunks in RAM")
        ("highResDistance", value<float>()->default_value(150.0f), "The distance of the far plane for the high resolution");

    // setup in baseoption
    setup();

}

bool Options::printUsage() const
{
    if (m_variables.count("help"))
    {
        cout << m_descr << endl;
        return true;
    }

    if (!m_variables.count("inputFile"))
    {
        cout << "Error: You must specify an input file." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }

    return false;
}

//std::vector<string> Options::getInputFiles() const
//{
//    return m_variables["inputFiles"].as<std::vector<string>>();
//}

std::string Options::getInputFileName() const
{
    return (m_variables["inputFile"].as< std::vector<string> >())[0];
}
std::vector<string> Options::getLayers() const
{
    if(m_variables.count("layers"))
    {
        return m_variables["layers"].as<std::vector<string>>();
    }
    else
    {
        return std::vector<std::string>({"mesh0, mesh1"});
    
    }
}

int Options::getCacheSize() const
{
    int size = m_variables["cacheSize"].as<int>();
    if(size > 0)
    {
        return size;
    }
    return 200;
}

float Options::getHighResDistance() const
{
    return m_variables["highResDistance"].as<float>();
}

bool Options::isChunkedMesh() const
{
    return m_variables["chunkedMesh"].as<bool>();
}

Options::~Options()
{
    // TODO Auto-generated destructor stub
}

} // namespace chunking
