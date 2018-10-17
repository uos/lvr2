#pragma once

#include <ostream>
#include <string>
#include <vector>

#include <lvr2/io/ScanprojectIO.hpp>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
namespace pt = boost::property_tree;



namespace lvr2
{

class RieglProject {
    public:
        RieglProject(std::string dir);        
        bool parse_project(unsigned int start, unsigned int end); 

    //private:
        fs::path m_project_dir;
        std::vector<ScanPosition> m_scan_positions;
        void parse_scanpositions(pt::ptree project_ptree, unsigned int start, unsigned int end);
        void parse_images_per_scanpos(ScanPosition &scanpos, pt::ptree scanpos_ptree, pt::ptree project_ptree);
};

std::ostream& operator<<(std::ostream &lhs, const RieglProject &rhs); 
std::ostream& operator<<(std::ostream &lhs, const ScanPosition &rhs);

} // namespace lvr2
