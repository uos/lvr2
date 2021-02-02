#include "Hdf5ReaderOld.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

using namespace hdf5util;

std::string string_shift(size_t nspaces)
{
    std::stringstream ss("");

    for(size_t i=0; i<nspaces; i++)
    {
        ss << " ";
    }

    return ss.str();
}

ScanPtr loadScanOld(HighFive::Group g)
{
    size_t level = 2 * 4;
    std::cout << string_shift(level) << "[Scan - load]" << std::endl;
    ScanPtr ret(new Scan);

    ret->points.reset(new PointBuffer);

    // HighFive::Group g_data = g.getGroup("data");

    auto splitted = split(g);

    std::cout << string_shift(level) << "Groups: " << std::endl;
    for(auto key : splitted.groups)
    {
        std::cout << string_shift(level) << "- " << key << std::endl;
    }

    std::cout << string_shift(level) << "Meta: " << std::endl;
    for(auto key : splitted.datasets)
    {
        std::cout << string_shift(level) << "- " << key << std::endl;
    }

    std::string channel_name = "points";
    if(g.exist(channel_name))
    {
        HighFive::DataSet ds_points = g.getDataSet(channel_name);
        std::vector<size_t> dims;
        floatArr points = hdf5util::getArray<float>(g, channel_name, dims);
        Channel<float> point_channel(dims[0], dims[1], points);
        ret->points->add(channel_name, point_channel);
    }

    std::string matrix_name = "registration";
    if(g.exist(matrix_name))
    {
        ret->transformation = *hdf5util::getMatrix<Transformd>(g, matrix_name);
        std::cout << string_shift(level) << "Loaded transformation" << std::endl;
        std::cout << ret->transformation << std::endl;
    }

    matrix_name = "poseEstimation";
    if(g.exist(matrix_name))
    {
        ret->poseEstimation = *hdf5util::getMatrix<Transformd>(g, matrix_name);
        std::cout << string_shift(level) << "Loaded poseEstimation" << std::endl;
        std::cout << ret->poseEstimation << std::endl;
    }

    return ret;
}

ScanPositionPtr loadScanPositionOld(HighFive::Group g)
{
    size_t level = 1 * 4;
    std::cout << string_shift(level) << "[ScanPosition - load]" << std::endl;
    ScanPositionPtr ret(new ScanPosition);

    LIDARPtr lidar(new LIDAR);

    auto splitted = split(g);

    std::cout << string_shift(level) << "Groups: " << std::endl;
    for(auto key : splitted.groups)
    {
        std::cout << string_shift(level) << "- " << key << std::endl;
        HighFive::Group g_scans = g.getGroup(key).getGroup("data");

        // iterate over all scans
        for(std::string scan_name : g_scans.listObjectNames())
        {
            HighFive::Group g_scan = g_scans.getGroup(scan_name);
            ScanPtr scan = loadScanOld(g_scan);
            lidar->scans.push_back(scan);
        }
        
    }

    std::cout << string_shift(level) << "Meta: " << std::endl;
    for(auto key : splitted.datasets)
    {
        std::cout << string_shift(level) << "- " << key << std::endl;
    }

    // LOAD TRANSFORMATIONS
    std::string matrix_name = "registration";
    if(g.exist(matrix_name))
    {
        ret->transformation = *hdf5util::getMatrix<Transformd>(g, matrix_name);
        std::cout << string_shift(level) << "Loaded transformation" << std::endl;
        std::cout << ret->transformation << std::endl;
    }

    matrix_name = "poseEstimation";
    if(g.exist(matrix_name))
    {
        ret->poseEstimation = *hdf5util::getMatrix<Transformd>(g, matrix_name);
        std::cout << string_shift(level) << "Loaded poseEstimation" << std::endl;
        std::cout << ret->poseEstimation << std::endl;
    }

    ret->lidars.push_back(lidar);

    return ret;
}

ScanProjectPtr loadScanProjectOld(HighFive::Group g)
{
    size_t level = 0;
    std::cout << "[ScanProject - load]" << std::endl;

    ScanProjectPtr ret(new ScanProject);

    auto splitted = split(g);

    std::cout << string_shift(level) << "Groups: " << std::endl;
    
    for(std::string key : splitted.groups)
    {
        std::cout << string_shift(level) << "- " << key << std::endl;
        HighFive::Group g_scan_pos = g.getGroup(key);
        ret->positions.push_back(loadScanPositionOld(g_scan_pos));
    }

    std::cout << string_shift(level) <<  "Meta: " << std::endl;
    for(std::string key : splitted.datasets)
    {
        HighFive::DataSet d = g.getDataSet(key);

        std::vector<size_t> dims = d.getSpace().getDimensions();

        std::cout << string_shift(level) << key << " [Meta] (";
        for(size_t dim : dims)
        {
            std::cout << dim << ",";
        }
        std::cout << ")" << std::endl;
    }

    return ret;
}


ScanProjectPtr loadOldHDF5(std::string filename)
{
    ScanProjectPtr ret;

    auto h5file = hdf5util::open(filename, HighFive::File::ReadOnly);

    // raw
    
    std::string scan_project_name = "raw";

    if(h5file->exist(scan_project_name))
    {
        HighFive::Group g_scan_project = h5file->getGroup(scan_project_name);

        ret = loadScanProjectOld(g_scan_project);
    }

    return ret;
}


} // namespace lvr2