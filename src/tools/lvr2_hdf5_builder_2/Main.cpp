#include "Options.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/hdf5/ArrayIO.hpp"
#include "lvr2/io/hdf5/ChannelIO.hpp"
#include "lvr2/io/hdf5/MatrixIO.hpp"
#include "lvr2/io/hdf5/PointCloudIO.hpp"
#include "lvr2/io/hdf5/VariantChannelIO.hpp"
#include "lvr2/types/Scan.hpp"

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <regex>
#include <yaml-cpp/yaml.h>

// const hdf5tool2::Options* options;
namespace qi = boost::spirit::qi;
using namespace lvr2;

using HDF5IO = lvr2::Hdf5IO<lvr2::hdf5features::ArrayIO,
                            lvr2::hdf5features::ChannelIO,
                            lvr2::hdf5features::VariantChannelIO,
                            lvr2::hdf5features::PointCloudIO,
                            lvr2::hdf5features::MatrixIO>;

bool m_usePreviews;
int m_previewReductionFactor;

bool parse_scan_filename(std::string path, int& i)
{
    // check whether the foldername ends with at least one number
    const std::regex reg("\\d+$");
    std::smatch match;

    bool r = std::regex_search(path, match, reg);
    if (r)
    {
        // set i depending on match
        i = std::stoi(match[0]);
    }

    return r;
}

namespace qi = boost::spirit::qi;
using namespace lvr2;
template <typename Iterator>
bool parse_png_filename(Iterator first, Iterator last, int& i)
{

    using boost::phoenix::ref;
    using boost::spirit::qi::_1;
    using qi::lit;
    using qi::parse;
    using qi::uint_parser;

    uint_parser<unsigned, 10, 1, -1> uint_3_d;

    bool r = parse(first,                  /*< start iterator >*/
                   last,                   /*< end iterator >*/
                   (uint_3_d[ref(i) = _1]) /*< the parser >*/
    );

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

bool sortScans(boost::filesystem::path firstScan, boost::filesystem::path secScan)
{
    std::string firstStem = firstScan.stem().string();
    std::string secStem = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_scan_filename(firstStem, i);
    bool sec = parse_scan_filename(secStem, j);

    if (first && sec)
    {
        return (i < j);
    }
    else
    {
        // this causes non valid files being at the beginning of the vector.
        if (sec)
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
    std::string secStem = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_png_filename(firstStem.begin(), firstStem.end(), i);
    bool sec = parse_png_filename(secStem.begin(), secStem.end(), j);

    if (first && sec)
    {
        return (i < j);
    }
    else
    {
        // this causes non valid files being at the beginning of the vector.
        if (sec)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

template <typename T>
boost::shared_array<T> reduceData(boost::shared_array<T> data,
                                  size_t dataCount,
                                  size_t dataWidth,
                                  unsigned int reductionFactor,
                                  size_t* reducedDataCount)
{
    *reducedDataCount = dataCount / reductionFactor + 1;
    boost::shared_array<T> reducedData =
        boost::shared_array<T>(new T[(*reducedDataCount) * dataWidth]);

    size_t reducedDataIdx = 0;
    for (size_t i = 0; i < dataCount; i++)
    {
        if (i % reductionFactor == 0)
        {
            std::copy(data.get() + i * dataWidth,
                      data.get() + (i + 1) * dataWidth,
                      reducedData.get() + reducedDataIdx * dataWidth);

            reducedDataIdx++;
        }
    }

    return reducedData;
}

bool saveScan(int nr, ScanPtr scan, HDF5IO hdf5)
{
    // Check scan data
    if (scan->m_points->numPoints())
    {
        std::cout << timestamp << "Saving data" << std::endl;
        // Setup group for scan data
        char buffer[128];
        sprintf(buffer, "position_%05d", nr);
        string nr_str(buffer);

        std::string groupName = "/raw/scans/" + nr_str;

        // Generate tuples for field of view and resolution parameters
        floatArr fov(new float[2]);
        fov[0] = scan->m_hFieldOfView;
        fov[1] = scan->m_vFieldOfView;

        floatArr res(new float[2]);
        res[0] = scan->m_hResolution;
        res[1] = scan->m_vResolution;

        // Generate pose estimation matrix array
        //    float* pose_data = new float[16];
        //    float* reg_data = new float[16];
        //
        //    std::copy(scan->m_poseEstimation.data(), scan->m_poseEstimation.data() + 16,
        //    pose_data); std::copy(scan->m_registration.data(), scan->m_registration.data() + 16,
        //    reg_data);
        //
        //    floatArr pose_estimate(pose_data);
        //    floatArr registration(reg_data);

        // Generate bounding box representation
        floatArr bb(new float[6]);

        auto bb_min = scan->m_boundingBox.getMin();
        auto bb_max = scan->m_boundingBox.getMax();
        bb[0] = bb_min.x;
        bb[1] = bb_min.y;
        bb[2] = bb_min.z;

        bb[3] = bb_max.x;
        bb[4] = bb_max.y;
        bb[5] = bb_max.z;

        // Add data to group
        std::vector<size_t> dim = {4, 4};
        std::vector<size_t> scan_dim = {scan->m_points->numPoints(), 3};
        //    hdf5.MatrixIO::save()
        hdf5.save(groupName, "fov", 2, fov);
        hdf5.save(groupName, "resolution", 2, res);
        hdf5.MatrixIO::save(groupName, "initialPose", scan->m_poseEstimation);
        hdf5.save(groupName, "finalPose", scan->m_registration);
        hdf5.save(groupName, "boundingBox", 6, bb);
        hdf5.save(groupName, "points", scan_dim, scan->m_points->getPointArray());

        // Add spectral annotation channel
        size_t an;
        size_t aw;
        ucharArr spectral = scan->m_points->getUCharArray("spectral_channels", an, aw);

        if (spectral)
        {
            size_t chunk_w = std::min<size_t>(an, 1000000); // Limit chunk size
            std::vector<hsize_t> chunk_annotation = {chunk_w, aw};
            std::vector<size_t> dim_annotation = {an, aw};
            hdf5.save(
                "/annotation/" + nr_str, "spectral", dim_annotation, chunk_annotation, spectral);
        }

        // Add preview data if wanted
        if (m_usePreviews)
        {
            std::string previewGroupName = "/preview/" + nr_str;

            // Add point preview
            floatArr points = scan->m_points->getPointArray();
            if (points)
            {
                size_t numPreview;
                floatArr previewData = reduceData(
                    points, scan->m_points->numPoints(), 3, m_previewReductionFactor, &numPreview);

                std::vector<size_t> previewDim = {numPreview, 3};
                hdf5.save(previewGroupName, "points", previewDim, previewData);
            }

            // Add spectral preview
            if (spectral)
            {
                size_t numPreview;
                ucharArr previewData =
                    reduceData(spectral, an, aw, m_previewReductionFactor, &numPreview);
                std::vector<size_t> previewDim = {numPreview, aw};
                hdf5.save(previewGroupName, "spectral", previewDim, previewData);
            }
        }
    }
}

bool channelIO(const boost::filesystem::path& p, int number, HDF5IO& hdf)
{
    std::cout << timestamp << "Start processing channels" << std::endl;
    std::vector<boost::filesystem::path> spectral;
    char group[256];
    sprintf(group, "/raw/spectral/position_%05d", number);
    // floatArr angles;
    // std::cout << p << std::endl;

    // count files and get all png pathes.
    for (boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator();
         ++it)
    {
        std::string fn = it->path().stem().string();
        int position;
        if ((it->path().extension() == ".png") &&
            parse_png_filename(fn.begin(), fn.end(), position))
        {
            spectral.push_back(*it);
        }
    }

    std::sort(spectral.begin(), spectral.end(), sortPanoramas);
    // std::cout << "sorted " << std::endl;

    // we assume that every frame has the same resolution
    // TODO change dimensions when writing
    cv::Mat img = cv::imread(spectral[0].string(), CV_LOAD_IMAGE_GRAYSCALE);
    ucharArr data(new unsigned char[spectral.size() * img.cols * img.rows]);
    std::memcpy(
        data.get() + (img.rows * img.cols), img.data, img.rows * img.cols * sizeof(unsigned char));

    std::vector<size_t> dim = {
        spectral.size(), static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)};

    for (size_t i = 1; i < spectral.size(); ++i)
    {

        cv::Mat img = cv::imread(spectral[i].string(), CV_LOAD_IMAGE_GRAYSCALE);
        std::memcpy(data.get() + i * (img.rows * img.cols),
                    img.data,
                    img.rows * img.cols * sizeof(unsigned char));
    }

    std::vector<hsize_t> chunks = {50, 50, 50};
    hdf.save(group, "channels", dim, chunks, data);
    // std::cout << "wrote channels" << std::endl;

    // TODO write aperture. 47.5 deg oder so
    // TODO panorama?

    std::cout << timestamp << "Finished processing channels" << std::endl;
    return true;
}

size_t readSpectralMetaData(const boost::filesystem::path& fn,
                            Channel<long>& timestamps,
                            Channel<double>& angleOffset)
{
    // TODO use double or long
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(fn.string());
    size_t size = 0;
    for (auto& n : root)
    {
        timestamps = Channel<long>(n.size() - 1, 1);
        //        std::cout << n.size() << std::endl;
        size = n.size();
        for (YAML::const_iterator it = n.begin(); it != n.end(); ++it)
        {
            // not sorted. key as index.
            if (it->first.as<std::string>() == std::string("offset_angle"))
            {
                angleOffset = Channel<double>(1, 1);
                angleOffset[0] = it->second.as<double>();
            }
            else
            {
                timestamps[it->first.as<int>()] = it->second["timestamp"].as<long>();
            }
        }
    }

    return size;
}

void readScanMetaData(const boost::filesystem::path& fn, ScanPtr& scan_ptr)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(fn.string());
    for (auto& n : root)
    {
        for (YAML::const_iterator it = n.begin(); it != n.end(); ++it)
        {
            if (it->first.as<string>() == "Start")
            {
                // Parse start time
                float sec = it->second["sec"].as<float>();
                float nsec = it->second["nsec"].as<float>();

                std::cout << "Start: " << sec << "; " << nsec << std::endl;
            }
            else if (it->first.as<string>() == "End")
            {
                // Parse end time
                float sec = it->second["sec"].as<float>();
                float nsec = it->second["nsec"].as<float>();

                std::cout << "End: " << sec << "; " << nsec << std::endl;
            }
            else if (it->first.as<string>() == "Pose")
            {
                Transformd pose = Transformd::Identity();
                size_t row = 0;
                size_t col = 0;
                for (auto& i : it->second)
                {
                    pose(row, col) = i.as<double>();
                    if (col >= 3)
                    {
                        row++;
                        col = 0;
                        // std::cout << std::endl;
                    }
                    else
                    {
                        col++;
                    }
                }
                scan_ptr->m_poseEstimation = pose;

                // Parse Position
                // if (it->second["Position"]) {
                //  YAML::Node tmp = it->second["Position"];
                //  float x = tmp["x"].as<float>();
                //  float y = tmp["y"].as<float>();
                //  float z = tmp["z"].as<float>();

                //  std::cout << "Pos: " << x << ", " << y << ", " << z << std::endl;
                //}
                // if (it->second["Rotation"]) {
                //  YAML::Node tmp = it->second["Rotation"];
                //  float x = tmp["x"].as<float>();
                //  float y = tmp["y"].as<float>();
                //  float z = tmp["z"].as<float>();

                //  std::cout << "Rot: " << x << ", " << y << ", " << z << std::endl;
                //}
            }
            else if (it->first.as<string>() == "Config")
            {
                // Parse Angles
                if (it->second["Theta"])
                {
                    YAML::Node tmp = it->second["Theta"];
                    float min = tmp["min"].as<float>();
                    float max = tmp["max"].as<float>();

                    scan_ptr->m_vFieldOfView = max - min;
                    scan_ptr->m_vResolution = tmp["delta"].as<float>();
                    // std::cout << "T: " << scan_ptr->m_vFieldOfView << "; "
                    //          << scan_ptr->m_vResolution << std::endl;
                }
                if (it->second["Phi"])
                {
                    YAML::Node tmp = it->second["Phi"];
                    float min = tmp["min"].as<float>();
                    float max = tmp["max"].as<float>();

                    scan_ptr->m_hFieldOfView = max - min;
                    scan_ptr->m_hResolution = tmp["delta"].as<float>();
                    // std::cout << "P: " << scan_ptr->m_hFieldOfView << "; "
                    //          << scan_ptr->m_hResolution << std::endl;
                }
            }
        }
        // std::cout << std::endl;
    }
    return;
}

bool spectralIO(const boost::filesystem::path& p, int number, HDF5IO& hdf)
{
    std::cout << timestamp << "Start processing frames" << std::endl;
    std::vector<boost::filesystem::path> spectral;
    char group[256];
    sprintf(group, "/raw/spectral/position_%05d", number);
    Channel<long> timestamps;
    Channel<double> angleOffsets;
    // std::cout << p << std::endl;

    size_t size = 0;
    // count files and get all png pathes.
    bool yaml = false;
    for (boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator();
         ++it)
    {
        if (it->path().extension() == ".yaml")
        {
            std::cout << timestamp << "Load yaml " << it->path() << std::endl;
            size = readSpectralMetaData(it->path(), timestamps, angleOffsets);
            yaml = true;
        }

        std::string fn = it->path().stem().string();
        int position;
        if ((it->path().extension() == ".png") &&
            parse_png_filename(fn.begin(), fn.end(), position))
        {
            spectral.push_back(*it);
        }
    }

    if (!yaml)
    {
        std::cout << timestamp << "No yaml config found" << std::endl;
    }

    if (size - 1 != spectral.size())
    {
        std::cout << timestamp << "Inconsistent"
                  << " " << size - 1 << " " << spectral.size() << std::endl;
    }

    std::sort(spectral.begin(), spectral.end(), sortPanoramas);

    // we assume that every frame has the same resolution
    // TODO change dimensions when writing
    cv::Mat img = cv::imread(spectral[0].string(), CV_LOAD_IMAGE_GRAYSCALE);
    ucharArr data(new unsigned char[spectral.size() * img.cols * img.rows]);
    std::memcpy(
        data.get() + (img.rows * img.cols), img.data, img.rows * img.cols * sizeof(unsigned char));

    std::vector<size_t> dim = {
        spectral.size(), static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)};

    for (size_t i = 1; i < spectral.size(); ++i)
    {

        cv::Mat img = cv::imread(spectral[i].string(), CV_LOAD_IMAGE_GRAYSCALE);
        std::memcpy(data.get() + i * (img.rows * img.cols),
                    img.data,
                    img.rows * img.cols * sizeof(unsigned char));
    }

    std::vector<hsize_t> chunks = {50, 50, 50};

    hdf.save(group, "frames", dim, chunks, data);

    if (size)
    {
        hdf.save(group, "timestamps", timestamps);
        hdf.save(group, "offset_angle", angleOffsets);
    }

    // TODO write aperture. 47.5 deg oder so
    // TODO panorama?

    // check if correct number of pngs
    std::cout << timestamp << "Finished processing frames" << std::endl;
    return true;
}

bool scanIO(const boost::filesystem::path& p,
            int number,
            const boost::filesystem::path& yaml,
            HDF5IO& hdf5)
{
    std::cout << timestamp << "Load scan " << p.string() << std::endl;
    ModelPtr model = ModelFactory::readModel(p.string());
    std::cout << timestamp << "Loaded " << model->m_pointCloud->numPoints() << " points"
              << std::endl;
    ScanPtr scan_ptr(new Scan());

    PointBufferPtr pc = model->m_pointCloud;
    scan_ptr->m_points = pc;
    floatArr points = pc->getPointArray();
    for (int i = 0; i < pc->numPoints(); i++)
    {
        scan_ptr->m_boundingBox.expand(
            BaseVector<float>(points[3 * i], points[3 * i + 1], points[3 * i + 2]));
    }

    // TODO parse yaml
    if (boost::filesystem::exists(yaml))
    {
        readScanMetaData(yaml, scan_ptr);
    }
    else
    {
        std::cout << timestamp << "No scan config found" << std::endl;
    }

    saveScan(number, scan_ptr, hdf5);

    // needed?
    return true;
}

int main(int argc, char** argv)
{
    hdf5tool2::Options options(argc, argv);
    boost::filesystem::path inputDir(options.getInputDir());

    m_usePreviews = options.getPreview();
    m_previewReductionFactor = options.getPreviewReductionRatio();

    int fileCounterIncr = 0;
    HDF5IO hdf;

    if (!boost::filesystem::exists(inputDir))
    {
        std::cout << timestamp << "Error: Directory " << options.getInputDir() << " does not exist"
                  << std::endl;
        exit(-1);
    }

    boost::filesystem::path outputPath(options.getOutputDir());

    // Check if output dir exists
    if (!boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "Creating directory " << options.getOutputDir() << std::endl;
        if (!boost::filesystem::create_directory(outputPath))
        {
            std::cout << timestamp << "Error: Unable to create " << options.getOutputDir()
                      << std::endl;
            exit(-1);
        }
    }

    outputPath /= options.getOutputFile();
    if (boost::filesystem::exists(outputPath))
    {
        std::cout << timestamp << "File already exists. Expanding File..." << std::endl;

        // get existing scans
        hdf.open(outputPath.string());
        HighFive::Group hfscans = hdf5util::getGroup(hdf.m_hdf5_file, "raw/scans");
        fileCounterIncr = hfscans.listObjectNames().size();
        std::cout << timestamp << "Using counter-increment " << fileCounterIncr << std::endl;
    }
    else
    {
        hdf.open(outputPath.string());
    }

    std::vector<boost::filesystem::path> scans;
    for (boost::filesystem::directory_iterator it(inputDir);
         it != boost::filesystem::directory_iterator();
         ++it)
    {
        scans.push_back(*it);
    }

    std::sort(scans.begin(), scans.end(), sortScans);
    int count = 0;
    for (auto p : scans)
    {
        std::cout << timestamp << "Reading path " << p << std::endl;
        char buffer[64];
        boost::filesystem::path ply;
        std::string fn = p.stem().string();

        // check if foldername matches [a-zA-z]*\d+
        if (!parse_scan_filename(fn, count))
        {
            std::cout << timestamp << "Invalid path " << p << std::endl;
            continue;
        }

        std::cout << timestamp << "Processing scan " << count << std::endl;

        bool ply_exists = false;
        bool spectral_exists = false;
        for (boost::filesystem::directory_iterator it(p);
             it != boost::filesystem::directory_iterator();
             ++it)
        {
            if (boost::filesystem::is_directory((*it).path()) && (*it).path().stem() == "spectral")
            {
                spectral_exists = spectralIO(it->path(), count + fileCounterIncr, hdf);
            }

            if (boost::filesystem::is_directory((*it).path()) && (*it).path().stem() == "channels")
            {
                channelIO(it->path(), count + fileCounterIncr, hdf);
            }

            if ((*it).path().extension() == ".ply")
            {
                ply = *it;
                ply_exists = true;
            }
        }

        if (!spectral_exists)
        {
            std::cout << timestamp << "No spectral information in: " << p << std::endl;
        }
        if (!ply_exists)
        {
            std::cout << timestamp << "No scan found" << std::endl;
        }
        else
        {
            scanIO(ply, count + fileCounterIncr, p / std::string("scan.yaml"), hdf);
            std::cout << timestamp << "Finished" << std::endl;
            std::cout << std::endl;
        }
    }
    std::cout << timestamp << "Program finished" << std::endl;
}
