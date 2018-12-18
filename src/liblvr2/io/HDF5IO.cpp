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

HDF5IO::HDF5IO(std::string filename, bool truncate) :
    m_hdf5_file(nullptr),
    m_compress(true),
    m_chunkSize(1e7)
{
    open(filename, truncate);
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

bool HDF5IO::open(std::string filename, bool truncate)
{
    // If file alredy exists, don't rewrite base structurec++11 init vector
    bool have_to_init = false;

    boost::filesystem::path path(filename);
    if(!boost::filesystem::exists(path) | truncate)
    {
        have_to_init = true;
    }

    // Try to open the given HDF5 file
    m_hdf5_file = new HighFive::File(
                filename,
                HighFive::File::OpenOrCreate | (truncate ? HighFive::File::Truncate : 0));

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

            if (!H5IMis_image(g.getId(), datasetName.c_str()))
            {
                return ret;
            }

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

bool HDF5IO::isScansGroup(std::string groupName)
{
    if (!exist(groupName))
    {
        return false;
    }

    HighFive::Group grp = getGroup(groupName);

    // very simple condition to be a scansgroup atm
    if (exist(groupName + "/position_00000") || exist(groupName + "/position_00001"))
    {
        return true;
    }

    return false;
}

std::vector<std::string> HDF5IO::getScanDataGroups()
{
    return getScanDataGroups("");
}

std::vector<std::string> HDF5IO::getScanDataGroups(std::string root)
{
    std::vector<std::string> ret;

    if (!exist(root))
    {
        return ret;
    }

    HighFive::Group grp;

    try
    {
        grp = getGroup(root);
    }
    catch (HighFive::Exception &e)
    {
        return ret;
    }

    // if it is a scansgroup we don't search in its leafs
    if (isScansGroup(root))
    {
        ret.push_back(root);
        return ret;
    }

    std::vector<std::string> leaf_objects = grp.listObjectNames();

    for (std::string leaf : leaf_objects)
    {
        if (isGroup(grp, leaf))
        {
            std::vector<std::string> leaf_ret = getScanDataGroups(root + "/" + leaf);

            for (std::string tmp : leaf_ret)
            {
                ret.push_back(tmp);
            }
        }
    }
}

std::vector<ScanData> HDF5IO::getScanData(std::string scanDataRoot, bool load_points)
{

    std::vector<ScanData> ret;

    if (!exist(scanDataRoot))
    {
        return ret;
    }

    HighFive::Group root_group = getGroup(scanDataRoot);
    size_t num_objects = root_group.getNumberObjects();

    for (size_t i = 0; i < num_objects; i++)
    {
        int pos_num;
        std::string cur_scan_pos = root_group.getObjectName(i);

        if (std::sscanf(cur_scan_pos.c_str(), "position_%5d", &pos_num))
        {
            ScanData cur_pos = getSingleScanData(scanDataRoot, pos_num, load_points);
            ret.push_back(cur_pos);
        }
    }

    return ret;

}

ScanData HDF5IO::getSingleScanData(std::string root, int nr, bool load_points)
{
    /// --->> TODO: QUICK FIX TO ALLOW LINKING !!!! @aloehr: Check getter signatures!!!!
    ///
    return getSingleRawScanData( nr,  load_points);
}

std::vector<ScanData> HDF5IO::getRawScanData(bool load_points)
{
    return getScanData("/raw/scans/", load_points);
}

ScanData HDF5IO::getSingleRawScanData(int nr, bool load_points)
{
    ScanData ret;

    if (m_hdf5_file)
    {
        char buffer[128];
        sprintf(buffer, "position_%05d", nr);

        string nr_str(buffer);
        std::string groupName = "/raw/scans/" + nr_str;

        unsigned int dummy;
        floatArr fov           = getArray<float>(groupName, "fov", dummy);
        floatArr res           = getArray<float>(groupName, "resolution", dummy);
        floatArr pose_estimate = getArray<float>(groupName, "initialPose", dummy);
        floatArr registration  = getArray<float>(groupName, "finalPose", dummy);
        floatArr bb            = getArray<float>(groupName, "boundingBox", dummy);

        if (load_points)
        {
            floatArr points    = getArray<float>(groupName, "points", dummy);

            if (points)
            {
                ret.m_points = PointBufferPtr(new PointBuffer(points, dummy/3));

                // annotation hack
                // if (is_annotation)
                {
                    std::vector<size_t> dim;
                    ucharArr spectral = getArray<unsigned char>(groupName + "/" + buffer, "spectral", dim);

                    if (spectral)
                    {
                        ret.m_points->addUCharChannel(spectral, "spectral_channels", dim[0], dim[1]);
                        ret.m_points->addIntAttribute(400, "spectral_wavelength_min");
                        ret.m_points->addIntAttribute(400 + 4 * dim[1], "spectral_wavelength_max");
                    }
                }
            }
        }

        floatArr preview = getArray<float>(groupName, "preview", dummy);

        if (preview)
        {
            ret.m_preview = PointBufferPtr( new PointBuffer(preview, dummy/3) );
        }

        if (fov)
        {
            ret.m_hFieldOfView = fov[0];
            ret.m_vFieldOfView = fov[1];
        }

        if (res)
        {
            ret.m_hResolution = res[0];
            ret.m_vResolution = res[1];
        }

        if (registration)
        {
            ret.m_registration   = Matrix4<BaseVector<float> >(registration.get());
        }

        if (pose_estimate)
        {
            ret.m_poseEstimation = Matrix4<BaseVector<float> >(pose_estimate.get());
        }

        if (bb)
        {
            ret.m_boundingBox = BoundingBox<BaseVector<float> >(
                    {bb[0], bb[1], bb[2]}, {bb[3], bb[4], bb[5]});
        }

        ret.m_pointsLoaded = load_points;
        ret.m_positionNumber = nr;

        ret.m_scanDataRoot = groupName;
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
        ret = getArray<float>(g, name, dim);

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

//            cout << "Copy float to int..." << endl;
//            intArray ints(new int[scan.m_points->numPoints() * 3]);
//            floatArr tmp_pts = scan.m_points->getPointArray();
//            for(size_t i = 0; i < scan.m_points->numPoints() * 3; i++)
//            {
//                ints[i] = (int)tmp_pts[i] * 10000;
//            }
//            cout << "Done" << endl;
            // Add data to group
            std::vector<size_t> dim = {4,4};
            std::vector<size_t> scan_dim = {scan.m_points->numPoints(), 3};
            addArray(groupName, "fov", 2, fov);
            addArray(groupName, "resolution", 2, res);
            addArray(groupName, "initialPose", dim, pose_estimate);
            addArray(groupName, "finalPose", dim, registration);
            addArray(groupName, "boundingBox", 6, bb);
            addArray(groupName, "points", scan_dim, scan.m_points->getPointArray());


            // Add spectral annotation channel
            size_t an;
            unsigned aw;
            ucharArr spectral = scan.m_points->getUCharArray("spectral_channels", an, aw);

            if (spectral)
            {
                size_t chunk_w = std::min<size_t>(an, 1000000);    // Limit chunk size
                std::vector<hsize_t> chunk_annotation = {chunk_w, aw};
                std::vector<size_t> dim_annotation = {an, aw};
                addArray("/annotation/" + nr_str, "spectral", dim_annotation, chunk_annotation, spectral);
            }

            int reduction_factor = 20;

            // Add point preview
            floatArr points = scan.m_points->getPointArray();
            if (points)
            {
                unsigned int num_preview = scan.m_points->numPoints() / reduction_factor;
                floatArr preview_data = floatArr( new float[3 * num_preview + 3] );

                size_t preview_idx = 0;
                for (size_t i = 0; i < scan.m_points->numPoints(); i++)
                {
                    if (i % reduction_factor == 0)
                    {
                        preview_data[preview_idx*3 + 0] = points[i*3 + 0];
                        preview_data[preview_idx*3 + 1] = points[i*3 + 1];
                        preview_data[preview_idx*3 + 2] = points[i*3 + 2];
                        preview_idx++;
                    }
                }

                std::vector<size_t> preview_dim = {num_preview, 3};
                addArray("/preview/" + nr_str, "points", preview_dim, preview_data);
            }


            // Add spectral preview
            if (spectral)
            {
                unsigned int num_preview = an / reduction_factor;
                ucharArr preview_data = ucharArr( new unsigned char[(num_preview + 1) * aw] );

                size_t preview_idx = 0;
                for (size_t i = 0; i < an; i++)
                {
                    if (i % reduction_factor == 0)
                    {
                        for (size_t j = 0; j < aw; j++)
                        {
                            preview_data[preview_idx*aw + j] = spectral[i*aw + j];
                        }
                        preview_idx++;
                    }
                }

                std::vector<size_t> preview_dim = {num_preview, aw};
                addArray("/preview/" + nr_str, "spectral", preview_dim, preview_data);
            }
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
                if (i < groupNames.size() -1)
                {
                    cur_grp = cur_grp.getGroup(groupNames[i]);
                }
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

bool HDF5IO::isGroup(HighFive::Group grp, std::string objName)
{
    H5G_stat_t stats;

    if (H5Gget_objinfo(grp.getId(), objName.c_str(), true, &stats) < 0)
    {
        return false;
    }

    if (stats.type == H5G_GROUP)
    {
        return true;
    }

    return false;
}


} // namespace lvr2
