#include "Options.hpp"
#include <iterator>

#include <boost/filesystem.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>


#include "lvr2/io/HDF5IO.hpp"
#include "lvr2/display/PointOctree.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/PointBuffer.hpp"

//const hdf5tool2::Options* options;
namespace qi = boost::spirit::qi;
using namespace lvr2;
template <typename Iterator>
bool parse_filename(Iterator first, Iterator last, int& i)
{

    using qi::lit;
    using qi::uint_parser;
    using qi::parse;
    using boost::spirit::qi::_1;
    using boost::phoenix::ref;

    uint_parser<unsigned, 10, 3, 3> uint_3_d;

    bool r = parse(
            first,                          /*< start iterator >*/
            last,                           /*< end iterator >*/
            ((lit("scan")|lit("Scan")) >> uint_3_d[ref(i) = _1])   /*< the parser >*/
            );

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

bool sortScans(boost::filesystem::path firstScan, boost::filesystem::path secScan)
{
    std::string firstStem = firstScan.stem().string();
    std::string secStem   = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_filename(firstStem.begin(), firstStem.end(), i);
    bool sec = parse_filename(secStem.begin(), secStem.end(), j);

    if(first && sec)
    {
        return (i < j);
    }
    else
    {
        // this causes non valid files being at the beginning of the vector.
        if(sec)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}




int main(int argc, char** argv)
{
    hdf5tool2::Options options(argc, argv);
    boost::filesystem::path inputDir(options.getInputDir());

    if(!boost::filesystem::exists(inputDir))
    {
        std::cout << timestamp << "Error: Directory " << options.getInputDir() << " does not exist" << std::endl;
        exit(-1);
    }
    
    boost::filesystem::path outputPath(options.getOutputDir());

    // Check if output dir exists
    if(!boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "Creating directory " << options.getOutputDir() << std::endl;
        if(!boost::filesystem::create_directory(outputPath))
        {
            std::cout << timestamp << "Error: Unable to create " << options.getOutputDir() << std::endl;
            exit(-1);
        }
    }
    
    outputPath /= options.getOutputFile();
    if(boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "Error: File exists " << outputPath << std::endl;
        exit(-1);
    }
    
    HDF5IO hdf(outputPath.string(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    std::vector<boost::filesystem::path> scans;
    for(boost::filesystem::directory_iterator it(inputDir); it != boost::filesystem::directory_iterator(); ++it)
    {
//        std::cout << *it << std::endl;
        scans.push_back(*it);
    
    }
    std::sort(scans.begin(), scans.end(), sortScans);
    int count = 0;
    for(auto p: scans)
    {
        char buffer[64];
        ScanData scan;

        boost::filesystem::path ply;
        boost::filesystem::path spectral;
        bool ply_exists = false;
        bool spectral_exists = false;
        for(boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator(); ++it)
        {
            if(boost::filesystem::is_directory((*it).path()) &&
               (*it).path().stem() == "spectral")
            {
                for(boost::filesystem::directory_iterator it2((*it).path()); it2 != boost::filesystem::directory_iterator(); ++it2)
                {
                    std::cout << *it2 << std::endl;
                    if((*it2).path().extension() == ".png")
                    {
                        spectral = (*it2).path();
                        spectral_exists = true;
                    }
                    else
                    {
                        std::cout << "No spectral information in: " << p << std::endl;
                    }
                }
       
            }
                    
            if((*it).path().extension() == ".ply")
            {
                ply = *it;
                ply_exists = true;
            }
        }

        if(!ply_exists)
        {
            std::cout << "aaaaaaa" << std::endl;
            exit(-1);
        }
            

        if(!spectral_exists)
        {
           std::cout << "No spectral information in: " << p << std::endl;
           exit(-1);
        }
        
        ModelPtr model = ModelFactory::readModel(ply.string());
        
        PointBufferPtr pc = model->m_pointCloud;
        // TODO bb
        scan.m_points = pc;
        floatArr points = pc->getPointArray();
        for(int i = 0; i < pc->numPoints(); i++)
        {
            scan.m_boundingBox.expand(BaseVector<float>(
                                      points[3 * i],
                                      points[3 * i + 1],
                                      points[3 * i + 2]));
        }

        hdf.addRawScanData(count, scan);

    }
}
