#include "lvr2/io/HDF5IO.hpp"

#include <boost/filesystem.hpp>

#include <chrono>
#include <ctime>
#include <algorithm>

namespace lvr2
{

HDF5IO::HDF5IO(std::string filename) : m_hdf5_file(nullptr)
{
    open(filename);
}

HDF5IO::~HDF5IO()
{
    if(m_hdf5_file)
    {
        delete m_hdf5_file;
    }
}

ModelPtr HDF5IO::read(std::string filename)
{
    return ModelPtr(new Model);
}

bool HDF5IO::open(std::string filename)
{
    // If file alredy exists, don't rewrite base structurec++11 init vector
    bool have_to_init = false;

    boost::filesystem::path path(filename);
    if(!boost::filesystem::exists(path))
    {
        have_to_init = true;
    }

    // Try to open the given HDF5 file
    m_hdf5_file = new HighFive::File(
                "out.h5",
                HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    if (!m_hdf5_file->isValid())
    {
        return false;
    }


    if(have_to_init)
    {
        // Write default groupts to new HDF5 file
        write_base_structure();
    }

    return true;
}

void HDF5IO::write_base_structure()
{
    int version = 1;
    m_hdf5_file->createDataSet<int>("version", HighFive::DataSpace::From(version)).write(version);
    HighFive::Group raw_data_group = m_hdf5_file->createGroup("/raw_data");

    // Create string with current time
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t t_now= std::chrono::system_clock::to_time_t(now);
    std::string time(ctime(&t_now));

    // Add current time to raw data group
    raw_data_group.createDataSet<std::string>("created", HighFive::DataSpace::From(time)).write(time);
    raw_data_group.createDataSet<std::string>("changed", HighFive::DataSpace::From(time)).write(time);

    // Create empty reference frame
    vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    raw_data_group.createDataSet<float>("reference_frame", HighFive::DataSpace::From(frame)).write(frame);

}

void HDF5IO::save(std::string filename)
{

}

void HDF5IO::addFloatArray(
        std::string group, std::string name,
        unsigned int size, floatArr data)
{
    if(m_hdf5_file)
    {
        std::vector<size_t> dim = {size, 1};
        HighFive::Group g = getGroup(group);
        addFloatArray(g, name, dim, data);
    }
}

void HDF5IO::addFloatArray(
        HighFive::Group& g,
        std::string datasetName, std::vector<size_t>& dim, floatArr data)
{
    if(m_hdf5_file)
    {
        HighFive::DataSet dataset = g.createDataSet<float>(datasetName, HighFive::DataSpace(dim));
        dataset.write(data.get());
    }
}

void HDF5IO::addFloatArray(
        std::string groupName, std::string datasetName,
        std::vector<size_t>& dimensions, floatArr data)
{
    if(m_hdf5_file)
    {
        HighFive::Group g = getGroup(groupName);
        addFloatArray(g, datasetName, dimensions, data);
    }
}

void HDF5IO::addUcharArray(std::string group, std::string name, unsigned int size, ucharArr data)
{
    if(m_hdf5_file)
    {
        vector<size_t> dim = {size, 1};
        HighFive::Group g = getGroup(group);
        addUcharArray(g, name, dim, data);
    }
}

void HDF5IO::addUcharArray(std::string group, std::string name, std::vector<size_t> dim, ucharArr data)
{
    if(m_hdf5_file)
    {
        HighFive::Group g = getGroup(group);
        addUcharArray(g, name, dim, data);
    }
}

void HDF5IO::addUcharArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim, ucharArr data)
{
    HighFive::DataSet dataset = g.createDataSet<unsigned char>(datasetName, HighFive::DataSpace(dim));
    dataset.write(data.get());
}

void HDF5IO::addImage(std::string group, std::string name, cv::Mat& img)
{
    if(m_hdf5_file)
    {

        HighFive::Group g = getGroup(group);
        addImage(g, name, img);
    }
}

void HDF5IO::addImage(HighFive::Group& g, std::string datasetName, cv::Mat& img)
{
    int w = img.cols;
    int h = img.rows;
    H5IMmake_image_8bit(g.getId(), datasetName.c_str(), w, h, img.data);
}

void HDF5IO::addRawScanData(int nr, ScanData &scan)
{
    try
    {
        HighFive::Group s_g = m_hdf5_file->createGroup("/raw_data/scans");
        std::cout << "Creating scans group..." << std::endl;
    }
    catch(...)
    {

    }

    if(m_hdf5_file)
    {
        // Check scan data
        if(scan.m_points->numPoints())
        {
            // Setup group for scan data
            char buffer[128];
            sprintf(buffer, "pose%05d", nr);
            string nr_str(buffer);


            std::string groupName = "/raw_data/scans/" + nr_str;
            std::cout << "Creating group " <<  groupName << std::endl;

            // Generate tuples for field of view and resolution parameters
            floatArr fov(new float[2]);
            fov[0] = scan.m_hFieldOfView;
            fov[1] = scan.m_vFieldOfView;

            floatArr res(new float[2]);
            res[0] = scan.m_hResolution;
            res[1] = scan.m_vResolution;

            // Generate pose estimation matrix array
            floatArr pose_estimate(scan.m_poseEstimation.toFloatArray());
            floatArr registration(scan.m_registration.toFloatArray());

            // Generate bounding box representation
            floatArr bb(new float[6]);

            auto bb_min = scan.m_boundingBox.getMin();
            auto bb_max = scan.m_boundingBox.getMax();
            bb[0] = bb_min.x;
            bb[1] = bb_min.y;
            bb[2] = bb_min.z;

            bb[3] = bb_max.x;
            bb[4] = bb_max.y;
            bb[5] = bb_max.z;

            // Add data to group
            addFloatArray(groupName, "fov", 2, fov);
            addFloatArray(groupName, "resolution", 2, res);
            addFloatArray(groupName, "pose_estimation", 16, pose_estimate);
            addFloatArray(groupName, "registration", 16, registration);
            addFloatArray(groupName, "bounding_box", 6, bb);
            addFloatArray(groupName, "points", 3 * scan.m_points->numPoints(), scan.m_points->getPointArray());
        }
    }
}

void HDF5IO::addRawDataHeader(std::string description, Matrix4<BaseVector<float>> &referenceFrame)
{

}


HighFive::Group HDF5IO::getGroup(const std::string &groupName) {
    if(m_hdf5_file->exist(groupName)) {
        // return the existing group
        return m_hdf5_file->getGroup(groupName);
    } else {
        return m_hdf5_file->createGroup(groupName);
    }

}


} // namespace lvr2
