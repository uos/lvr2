#include "Options.hpp"
#include <iterator>

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "lvr2/io/HDF5IO.hpp"
#include "lvr2/display/PointOctree.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/PointBuffer.hpp"

#include "lvr2/io/PlutoMetaDataIO.hpp"

//const hdf5tool2::Options* options;
namespace qi = boost::spirit::qi;
using namespace lvr2;
template <typename Iterator>
bool parse_scan_filename(Iterator first, Iterator last, int& i)
{
    using qi::lit;
    using qi::uint_parser;
    using qi::parse;
    using boost::spirit::qi::_1;
    using boost::phoenix::ref;

    uint_parser<unsigned, 10, 1, -1> uint_3_d;

    bool r = parse(
            first,                          /*< start iterator >*/
            last,                           /*< end iterator >*/
            (uint_3_d[ref(i) = _1])   /*< the parser >*/
            );

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

namespace qi = boost::spirit::qi;
using namespace lvr2;
template <typename Iterator>
bool parse_png_filename(Iterator first, Iterator last, int& i)
{

    using qi::lit;
    using qi::uint_parser;
    using qi::parse;
    using boost::spirit::qi::_1;
    using boost::phoenix::ref;

    uint_parser<unsigned, 10, 1, -1> uint_3_d;

    bool r = parse(
            first,                          /*< start iterator >*/
            last,                           /*< end iterator >*/
            (uint_3_d[ref(i) = _1])   /*< the parser >*/
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

    bool first = parse_scan_filename(firstStem.begin(), firstStem.end(), i);
    bool sec   = parse_scan_filename(secStem.begin(), secStem.end(), j);

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

bool sortPanoramas(boost::filesystem::path firstScan, boost::filesystem::path secScan)
{
    std::string firstStem = firstScan.stem().string();
    std::string secStem   = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_png_filename(firstStem.begin(), firstStem.end(), i);
    bool sec   = parse_png_filename(secStem.begin(), secStem.end(), j);

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

bool spectralIO(const boost::filesystem::path& p, int number, HDF5IO& hdf)
{
    std::vector<boost::filesystem::path> spectral;
    char group[256];
    sprintf(group, "/raw/spectral/position_%05d", number);
    floatArr angles;
    std::cout << p << std::endl;
   
    size_t count = 0;
    size_t size  = 0;
    // count files and get all png pathes.
    for(boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator(); ++it)
    {
        if(it->path().extension() == ".yaml")
        {
            size = PlutoMetaDataIO::readSpectralMetaData(it->path(), angles);
        }

        if(it->path().extension() == ".png")
        {
            spectral.push_back(*it);
            count++;
        }
    }

    if(size != count)
    {
        std::cout << "Incosistent" << std::endl;
        exit(-1);
    }

    std::sort(spectral.begin(), spectral.end(), sortPanoramas);
    std::cout << "sorted " << std::endl;

    // we assume that every frame has the same resolution
    // TODO change dimensions when writing
    cv::Mat img = cv::imread(spectral[0].string(), CV_LOAD_IMAGE_GRAYSCALE);
    ucharArr data(new unsigned char[count * img.cols * img.rows]);
    std::memcpy(data.get() + (img.rows * img.cols),
                img.data,
                img.rows * img.cols * sizeof(unsigned char));
    
    std::vector<size_t> dim = {count, 
                               static_cast<size_t>(img.rows),
                               static_cast<size_t>(img.cols)};

    for(size_t i = 1; i < spectral.size(); ++ i)
    {   

        cv::Mat img = cv::imread(spectral[i].string(), CV_LOAD_IMAGE_GRAYSCALE);
        std::memcpy(data.get() + i * (img.rows * img.cols),
                    img.data,
                    img.rows * img.cols * sizeof(unsigned char));
    }
    
    std::vector<hsize_t> chunks = {50, 50, 50};
    hdf.addArray(group, "spectral", dim, chunks, data);
    std::cout << "wrote spectral" << std::endl;
    
    hdf.addArray(group, "angles", size, angles);
    std::cout << "wrote angles" << std::endl;

    // TODO write aperture. 47.5 deg oder so
    // TODO panorama?

    // check if correct number of pngs
    return true;
}

bool scanIO(const boost::filesystem::path& p,  int number, const boost::filesystem::path& yaml, HDF5IO& hdf)
{
        std::cout << "read scan " << p << std::endl;
        ModelPtr model = ModelFactory::readModel(p.string());
        ScanData scan;

        
        PointBufferPtr pc = model->m_pointCloud;
        scan.m_points = pc;
        floatArr points = pc->getPointArray();
        for(int i = 0; i < pc->numPoints(); i++)
        {
            scan.m_boundingBox.expand(BaseVector<float>(
                                      points[3 * i],
                                      points[3 * i + 1],
                                      points[3 * i + 2]));
        }

        // TODO parse yaml
        PlutoMetaDataIO::readScanMetaData(yaml, scan);
        hdf.addRawScanData(number, scan);

        // needed?
        return true;
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
        scans.push_back(*it);
    }

    std::sort(scans.begin(), scans.end(), sortScans);
    int count = 0;
    for(auto p: scans)
    {
        char buffer[64];
        boost::filesystem::path ply;

        bool ply_exists = false;
        bool spectral_exists = false;
        for(boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator(); ++it)
        {
            if(boost::filesystem::is_directory((*it).path()) &&
               (*it).path().stem() == "spectral")
            {
                spectral_exists = spectralIO(it->path(), count, hdf);
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
        scanIO(ply, count, p/std::string("scan.yaml"), hdf);
    }
}
