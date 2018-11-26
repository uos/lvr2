/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "lvr2/io/HDF5IO.hpp"

#include <boost/filesystem.hpp>

#include <chrono>
#include <ctime>
#include <algorithm>

namespace lvr2
{

HDF5IO::HDF5IO(std::string filename) :
    m_hdf5_file(nullptr),
    m_compress(true),
    m_chunkSize(1e7)
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

void HDF5IO::setCompress(bool compress)
{
    m_compress = compress;
}

void HDF5IO::setChunkSize(const size_t& size)
{
    m_chunkSize = size;
}

bool HDF5IO::compress()
{
    return m_compress;
}

size_t HDF5IO::chunkSize()
{
    return m_chunkSize;
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
                filename,
                HighFive::File::ReadWrite | (have_to_init ? HighFive::File::Create : 0));

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
    HighFive::Group raw_data_group = m_hdf5_file->createGroup("raw");

    // Create string with current time
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t t_now= std::chrono::system_clock::to_time_t(now);
    std::string time(ctime(&t_now));

    // Add current time to raw data group
    raw_data_group.createDataSet<std::string>("created", HighFive::DataSpace::From(time)).write(time);
    raw_data_group.createDataSet<std::string>("changed", HighFive::DataSpace::From(time)).write(time);

    // Create empty reference frame
    vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    std::cout << frame.size() << std::endl;
    raw_data_group.createDataSet<float>("position", HighFive::DataSpace::From(frame)).write(frame);

}

void HDF5IO::save(std::string filename)
{

}

floatArr HDF5IO::getFloatArray(
        std::string groupName, std::string datasetName,
        unsigned int& size)
{
    floatArr ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            std::vector<size_t> dim;
            ret = getFloatArray(g, datasetName, dim);

            size = 1;

            // if you use this function, you expect a one dimensional array
            // and therefore we calculate the toal amount of elements
            for (auto cur : dim)
                size *= cur;
        }
    }

    return ret;
}

floatArr HDF5IO::getFloatArray(
        std::string groupName, std::string datasetName,
        std::vector<size_t>& dim)
{
    floatArr ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            ret = getFloatArray(g, datasetName, dim);
        }
    }

    return ret;
}

floatArr HDF5IO::getFloatArray(
        HighFive::Group& g, std::string datasetName,
        std::vector<size_t>& dim)
{
    floatArr ret;

    if(m_hdf5_file)
    {
        if (g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if(elementCount)
            {
                ret = floatArr(new float[elementCount]);

                dataset.read(ret.get());
            }
        }
    }

    return ret;
}

ucharArr HDF5IO::getUcharArray(
        std::string groupName, std::string datasetName,
        unsigned int& size)
{
    ucharArr ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            std::vector<size_t> dim;
            ret = getUcharArray(g, datasetName, dim);

            size = 1;

            // if you use this function, you expect a one dimensional array
            // and therefore we calculate the toal amount of elements
            for (auto cur : dim)
                size *= cur;
        }
    }

    return ret;
}

ucharArr HDF5IO::getUcharArray(
        std::string groupName, std::string datasetName,
        std::vector<size_t>& dim)
{
    ucharArr ret;

    if(m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            ret = getUcharArray(g, datasetName, dim);
        }
    }

    return ret;
}

ucharArr HDF5IO::getUcharArray(
        HighFive::Group& g, std::string datasetName,
        std::vector<size_t>& dim)
{
    ucharArr ret;

    if(m_hdf5_file)
    {
        if (g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if(elementCount)
            {
                ret = ucharArr(new unsigned char[elementCount]);

                dataset.read(ret.get());
            }
        }
    }

    return ret;
}

Texture HDF5IO::getImage(std::string groupName, std::string datasetName)
{

    Texture ret;

    if (m_hdf5_file)
    {
        if (exist(groupName))
        {
            HighFive::Group g = getGroup(groupName, false);
            ret = getImage(g, datasetName);
        }
    }

    return ret;
}

Texture HDF5IO::getImage(HighFive::Group& g, std::string datasetName)
{
    Texture ret;

    if (m_hdf5_file)
    {
        if (g.exist(datasetName))
        {
            long long unsigned int width, height, planes;
            long long int npals;
            char interlace[256];

            if (H5IMget_image_info(
                        g.getId(), datasetName.c_str(), &width, &height,
                        &planes, interlace, &npals) >= 0)
            {
                if (width && height && planes && npals == 0)
                {
                    ret = Texture(0, width, height, planes, 1, 1.0);

                    if (H5IMread_image(g.getId(), datasetName.c_str(), ret.m_data) < 0)
                    {
                        ret = Texture();
                    }
                }
            }
        }
    }

    return ret;
}

std::vector<ScanData> HDF5IO::getRawScanData(bool load_points)
{
    std::vector<ScanData> ret;

    if (!exist("/raw_data/"))
    {
        return ret;
    }

    HighFive::Group raw_group = getGroup("/raw_data/");
    size_t num_objects = raw_group.getNumberObjects();

    for (size_t i = 0; i < num_objects; i++)
    {
        int pos_num;

        if (std::sscanf(raw_group.getObjectName(i).c_str(), "pose%5d", &pos_num))
        {
            HighFive::Group pos_grp = raw_group.getGroup(raw_group.getObjectName(i));

            ScanData cur_pos = getRawScanData(pos_num, load_points);
            ret.push_back(cur_pos);
        }
    }

    return ret;
}

ScanData HDF5IO::getRawScanData(int nr, bool load_points)
{
    ScanData ret;

    if (m_hdf5_file)
    {
        char buffer[128];
        sprintf(buffer, "pose%05d", nr);
        string nr_str(buffer);

        std::string groupName = "/raw_data/" + nr_str;

        HighFive::Group g = getGroup(groupName);

        unsigned int dummy;
        floatArr fov           = getFloatArray(groupName, "fov", dummy);
        floatArr res           = getFloatArray(groupName, "resolution", dummy);
        floatArr pose_estimate = getFloatArray(groupName, "pose_estimation", dummy);
        floatArr registration  = getFloatArray(groupName, "registration", dummy);
        floatArr bb            = getFloatArray(groupName, "bounding_box", dummy);

        if (load_points)
        {
            floatArr points        = getFloatArray(groupName, "points", dummy);
            ret.m_points = PointBufferPtr(new PointBuffer(points, dummy/3));
        }

        ret.m_hFieldOfView = fov[0];
        ret.m_vFieldOfView = fov[1];

        ret.m_hResolution = res[0];
        ret.m_vResolution = res[1];

        ret.m_registration   = Matrix4<BaseVector<float> >(registration.get());
        ret.m_poseEstimation = Matrix4<BaseVector<float> >(pose_estimate.get());

        ret.m_boundingBox = BoundingBox<BaseVector<float> >(
                {bb[0], bb[1], bb[2]}, {bb[3], bb[4], bb[5]});

        ret.m_pointsLoaded = load_points;
        ret.m_positionNumber = nr;
    }

    return ret;
}

floatArr HDF5IO::getFloatChannelFromRawScanData(std::string name, int nr, unsigned int& n, unsigned& w)
{
    floatArr ret;

    if (m_hdf5_file)
    {
        char buffer[128];
        sprintf(buffer, "pose%05d", nr);
        string nr_str(buffer);

        std::string groupName = "/raw_data/" + nr_str;

        HighFive::Group g = getGroup(groupName);

        std::vector<size_t> dim;
        ret = getFloatArray(g, name, dim);

        if (dim.size() != 2)
        {
            throw std::runtime_error(
                "HDF5IO - getFloatchannelFromRawScanData() Error: dim.size() != 2");
        }

        n = dim[0];
        w = dim[1];
    }

    return ret;
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

void HDF5IO::addFloatChannelToRawScanData(
        std::string name, int nr, size_t n, unsigned w, floatArr data)
{
    try
    {
        HighFive::Group g = getGroup("raw/scans");
    }
    catch(HighFive::Exception& e)
    {
        std::cout << timestamp << "Error adding raw scan data: "
                  << e.what() << std::endl;
        throw e;
    }

    if(data != nullptr && n > 0 && w > 0 && m_hdf5_file)
    {
        // Setup group for scan data
        char buffer[128];
        sprintf(buffer, "position_%05d", nr);
        string nr_str(buffer);
        std::string groupName = "/raw/scans/" + nr_str;
        std::vector<size_t> dim = {n, w};
        addArray(groupName, name, dim, data);
    }
    else
    {
        std::cout << timestamp << "Error adding float channel '" << name
                               << "'to raw scan data" << std::endl;
    }
}

void HDF5IO::addHyperspectralCalibration(int position, const HyperspectralCalibration& calibration)
{
    try
    {
        HighFive::Group g = getGroup("raw/spectral");
    }
    catch(HighFive::Exception& e)
    {
        std::cout << timestamp << "Error adding hyperspectral calibration data: "
                  << e.what() << std::endl;
        throw e;
    }

    // Add calibration values
    if(m_hdf5_file)
    {
        // Setup group for scan data
        char buffer[128];
        sprintf(buffer, "position_%05d", position);
        string nr_str(buffer);
        std::string groupName = "/raw/spectral/" + nr_str;

        floatArr a(new float[3]);
        a[0] = calibration.a0;
        a[1] = calibration.a1;
        a[2] = calibration.a2;

        floatArr rotation(new float[3]);
        a[0] = calibration.angle_x;
        a[1] = calibration.angle_y;
        a[2] = calibration.angle_z;

        floatArr origin(new float[3]);
        origin[0] = calibration.origin_x;
        origin[1] = calibration.origin_y;
        origin[2] = calibration.origin_z;

        floatArr principal(new float[2]);
        principal[0] = calibration.principal_x;
        principal[1] = calibration.principal_y;

        addArray(groupName, "distortion", 3, a);
        addArray(groupName, "rotation", 3, rotation);
        addArray(groupName, "origin", 3, origin);
        addArray(groupName, "prinzipal", 2, principal);
    }
}

void HDF5IO::addRawScanData(int nr, ScanData &scan)
{
    try
    {
        HighFive::Group g = getGroup("raw/scans");
    }
    catch(HighFive::Exception& e)
    {
        std::cout << timestamp << "Error adding raw scan data: "
                  << e.what() << std::endl;
        throw e;
    }

    if(m_hdf5_file)
    {
        // Check scan data
        if(scan.m_points->numPoints())
        {
            // Setup group for scan data
            char buffer[128];
            sprintf(buffer, "position_%05d", nr);
            string nr_str(buffer);


            std::string groupName = "/raw/scans/" + nr_str;

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
            std::vector<size_t> dim = {4,4};
            addArray(groupName, "fov", 2, fov);
            addArray(groupName, "resolution", 2, res);
            addArray(groupName, "initialPose", dim, pose_estimate);
            addArray(groupName, "finalPose", dim, registration);
            addArray(groupName, "boundingbox", 6, bb);
            std::vector<size_t> scan_dim = {scan.m_points->numPoints(), 3};
            addArray(groupName, "points", scan_dim, scan.m_points->getPointArray());
        }
    }
}

void HDF5IO::addRawDataHeader(std::string description, Matrix4<BaseVector<float>> &referenceFrame)
{

}

std::vector<std::string> HDF5IO::splitGroupNames(const std::string &groupName)
{
    std::vector<std::string> ret;

    std::string remainder = groupName;
    size_t delimiter_pos = 0;

    while ( (delimiter_pos = remainder.find('/', delimiter_pos)) != std::string::npos)
    {
        if (delimiter_pos > 0)
        {
            ret.push_back(remainder.substr(0, delimiter_pos));
        }

        remainder = remainder.substr(delimiter_pos + 1);

        delimiter_pos = 0;
    }

    if (remainder.size() > 0)
    {
        ret.push_back(remainder);
    }

    return ret;
}


HighFive::Group HDF5IO::getGroup(const std::string &groupName, bool create)
{
    std::vector<std::string> groupNames = splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {
        cur_grp = m_hdf5_file->getGroup("/");

        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (cur_grp.exist(groupNames[i]))
            {
                cur_grp = cur_grp.getGroup(groupNames[i]);
            }
            else if (create)
            {
                cur_grp = cur_grp.createGroup(groupNames[i]);
            }
            else
            {
                // Throw exception because a group we searched
                // for doesn't exist and create flag was false
                throw std::runtime_error("HDF5IO - getGroup(): Groupname '"
                    + groupNames[i] + "' doesn't exist and create flag is false");
            }
        }
    }
    catch(HighFive::Exception& e)
    {
        std::cout << timestamp
                  << "Error in getGroup (with group name '"
                  << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return cur_grp;
}

bool HDF5IO::exist(const std::string &groupName)
{
    std::vector<std::string> groupNames = splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {
        cur_grp = m_hdf5_file->getGroup("/");

        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (cur_grp.exist(groupNames[i]))
            {
                cur_grp = cur_grp.getGroup(groupNames[i]);
            }
            else
            {
                return false;
            }
        }
    }
    catch (HighFive::Exception& e)
    {
        std::cout << timestamp
                  << "Error in exist (with group name '"
                  << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;

    }

    return true;
}


} // namespace lvr2
