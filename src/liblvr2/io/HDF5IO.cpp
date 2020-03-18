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

const std::string HDF5IO::vertices_name = "vertices";
const std::string HDF5IO::indices_name = "indices";
const std::string HDF5IO::meshes_group = "meshes";

HDF5IO::HDF5IO(const std::string filename, const std::string part_name, int open_flag) :
    m_hdf5_file(nullptr),
    m_compress(true),
    m_chunkSize(1e7),
    m_usePreviews(true),
    m_previewReductionFactor(20),
    m_part_name(part_name),
    m_mesh_path(meshes_group+"/"+part_name)
{
    std::cout << timestamp << " Try to open file \"" << filename << "\"..." << std::endl;
    if(!open(filename, open_flag))
    {
        std::cerr << timestamp << " Could not open file \"" << filename << "\"!" << std::endl;
    }
}

HDF5IO::HDF5IO(std::string filename, int open_flag) :
    m_hdf5_file(nullptr),
    m_compress(true),
    m_chunkSize(1e7),
    m_usePreviews(true),
    m_previewReductionFactor(20),
    m_part_name("")
{
    open(filename, open_flag); // TODO Open should not be in the constructor
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

void HDF5IO::setPreviewReductionFactor(const unsigned int factor)
{
    if (factor >= 1)
    {
        m_previewReductionFactor = factor;
    }
    else
    {
        m_previewReductionFactor = 20;
    }
}

void HDF5IO::setUsePreviews(bool use)
{
    m_usePreviews = use;
}

bool HDF5IO::compress()
{
    return m_compress;
}

bool HDF5IO::deleteDataset(const char* name)
{
    // delete returning non-negative means success
    return H5Ldelete(m_hdf5_file->getId(), name, H5P_DEFAULT) >= 0; // laplid = H5P_DEFAULT
}

size_t HDF5IO::chunkSize()
{
    return m_chunkSize;
}

ModelPtr HDF5IO::read(std::string filename)
{
    open(filename, HighFive::File::ReadOnly);
    ModelPtr model_ptr(new Model);

    std::cout << timestamp << "HDF5IO: loading..." << std::endl;
    // read mesh information
    if(readMesh(model_ptr))
    {
        std::cout << timestamp << " Mesh successfully loaded." << std::endl;
    } else {
        std::cout << timestamp << " Mesh could not be loaded." << std::endl;
    }

    // read pointcloud information out of scans
    if(readPointCloud(model_ptr))
    {
        std::cout << timestamp << " PointCloud successfully loaded." << std::endl;
    } else {
        std::cout << timestamp << " PointCloud could not be loaded." << std::endl;
    }
    
    return model_ptr;
}

bool HDF5IO::readPointCloud(ModelPtr model_ptr)
{
    std::vector<ScanPtr> scans = getRawScans(true);
    if(scans.size() == 0)
    {
        return false;
    }

    size_t n_points_total = 0;
    for(const ScanPtr& scan : scans)
    {
        n_points_total += scan->points->numPoints();
    }

    floatArr points(new float[n_points_total * 3]);
    BaseVector<float>* points_raw_it = reinterpret_cast<BaseVector<float>* >(
        points.get()
    );

    for(int i=0; i<scans.size(); i++)
    {
        size_t num_points = scans[i]->points->numPoints();
        floatArr pts = scans[i]->points->getPointArray();

        Transformd T = scans[i]->poseEstimation;
        T.transpose();

        BaseVector<float>* begin = reinterpret_cast<BaseVector<float>* >(pts.get());
        BaseVector<float>* end = begin + num_points;

        while(begin != end)
        {
            const BaseVector<float>& cp = *begin;
            *points_raw_it = T * cp;

            begin++;
            points_raw_it++;
        }
    }

    model_ptr->m_pointCloud.reset(new PointBuffer(points, n_points_total));

    return true;
}

bool HDF5IO::readMesh(ModelPtr model_ptr)
{
    const std::string mesh_resource_path = "meshes/" + m_part_name;
    const std::string vertices("vertices");
    const std::string indices("indices");

    if(!exist(mesh_resource_path)){
        return false;
    } else {
        auto mesh = getGroup(mesh_resource_path);
        
        if(!mesh.exist(vertices) || !mesh.exist(indices)){
            std::cout << timestamp << " The mesh has to contain \"" << vertices
                << "\" and \"" << indices << "\"" << std::endl;
            std::cout << timestamp << " Return empty model pointer!" << std::endl;
            return false;
        }

        std::vector<size_t> vertexDims;
        std::vector<size_t> faceDims;

        // read mesh buffer
        floatArr vbuffer = getArray<float>(mesh_resource_path, vertices, vertexDims);
        indexArray ibuffer = getArray<unsigned int>(mesh_resource_path, indices, faceDims);

        if(vertexDims[0] == 0)
        {
            return false;
        }
        if(!model_ptr->m_mesh)
        {
            model_ptr->m_mesh.reset(new MeshBuffer());
        }

        model_ptr->m_mesh->setVertices(vbuffer, vertexDims[0]);

        model_ptr->m_mesh->setFaceIndices(ibuffer, faceDims[0]);
    }
    return true;
}

bool HDF5IO::open(std::string filename, int open_flag)
{
    // If file alredy exists, don't rewrite base structurec++11 init vector
    bool have_to_init = false;

    boost::filesystem::path path(filename);
    if( (!boost::filesystem::exists(path)) || (open_flag == HighFive::File::Truncate))
    {
        have_to_init = true;
    }
    // Try to open the given HDF5 file
    m_hdf5_file = new HighFive::File(filename, open_flag);

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
    //vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    //std::cout << frame.size() << std::endl;
    //raw_data_group.createDataSet<float>("position", HighFive::DataSpace::From(frame)).write(frame);

}

void HDF5IO::save(std::string filename)
{

}

void HDF5IO::save(ModelPtr model, std::string filename)
{
    open(filename, HighFive::File::ReadWrite);

    if(saveMesh(model))
    {
        std::cout << timestamp << " Mesh succesfully saved to " << filename << std::endl;
    } else {
        std::cout << timestamp << " Mesh could not saved to " << filename << std::endl;
    }
}

bool HDF5IO::saveMesh(ModelPtr model_ptr)
{
    if(!model_ptr->m_mesh)
    {
        std::cout << timestamp << " Model does not contain a mesh" << std::endl;
        return false;
    }
    
    const std::string mesh_resource_path = "meshes/" + m_part_name;
    const std::string vertices("vertices");
    const std::string indices("indices");

    
    if(exist(mesh_resource_path)){
        std::cout << timestamp << " Mesh already exists in file!" << std::endl;
        return false;
    } else {

        auto mesh = getGroup(mesh_resource_path);

        if(mesh.exist(vertices) || mesh.exist(indices)){
            std::cout << timestamp << " The mesh has to contain \"" << vertices
                << "\" and \"" << indices << "\"" << std::endl;
            std::cout << timestamp << " Return empty model pointer!" << std::endl;
            return false;
        }

        std::vector<size_t> vertexDims = {model_ptr->m_mesh->numVertices(), 3};
        std::vector<size_t> faceDims = {model_ptr->m_mesh->numFaces(), 3};

        if(vertexDims[0] == 0)
        {
            std::cout << timestamp << " The mesh has 0 vertices" << std::endl;
            return false;
        }
        if(faceDims[0] == 0)
        {
            std::cout << timestamp << " The mesh has 0 faces" << std::endl;
            return false;
        }

        addArray<float>(
            mesh_resource_path,
            vertices,
            vertexDims,
            model_ptr->m_mesh->getVertices()
        );

        addArray<unsigned int>(
            mesh_resource_path,
            indices,
            faceDims,
            model_ptr->m_mesh->getFaceIndices()
        );
        
    }

    return true;

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

std::vector<ScanPtr> HDF5IO::getRawScans(bool load_points)
{
    std::string groupName = "/raw/scans/";
    std::vector<ScanPtr> ret;

    if (!exist(groupName))
    {
        return ret;
    }

    HighFive::Group root_group = getGroup(groupName);
    size_t num_objects = root_group.getNumberObjects();

    for (size_t i = 0; i < num_objects; i++)
    {
        int pos_num;
        std::string cur_scan_pos = root_group.getObjectName(i);

        if (std::sscanf(cur_scan_pos.c_str(), "position_%5d", &pos_num))
        {
            ScanPtr cur_pos = getSingleRawScan(pos_num, load_points);
            ret.push_back(cur_pos);
        }
    }

    return ret;

}



// template<template<typename Base> typename ...ComponentTs>
// class Hdf5IO;


// template<template<typename Base> typename ComponentT>
// class Hdf5IO<ComponentT<Hdf5IO<ComponentT> > > : public ComponentT<Hdf5IO<ComponentT> >, public BaseHdf5IO
// {
// public:
//     using ComponentT<Hdf5IO<ComponentT> >::save;
// };

// // template<template<typename Base> typename ...ComponentT >
// template<template<typename Base1> typename ComponentT, template<typename Base2> typename ...ComponentTs >
// class Hdf5IO<ComponentT<Hdf5IO<ComponentT> >, ComponentTs<Hdf5IO<ComponentTs...> >...> 
// : public ComponentT<Hdf5IO<ComponentT> >, public Hdf5IO<ComponentTs<Hdf5IO<ComponentTs...> >...>
// {
// public:
//     using ComponentT<Hdf5IO<ComponentT> >::save;
//     using Hdf5IO<ComponentTs<Hdf5IO<ComponentTs...> >...>::save;
// };

std::vector<std::vector<ScanImage> > HDF5IO::getRawCamData(bool load_image_data)
{
    std::vector<std::vector<ScanImage> > ret;
    
    if(m_hdf5_file) 
    {
        std::string groupNamePhotos = "/raw/photos/";

        if(!exist(groupNamePhotos))
        {
            return ret;
        }

        HighFive::Group photos_group = getGroup(groupNamePhotos);

        size_t num_scans = photos_group.getNumberObjects();


        for (size_t i = 0; i < num_scans; i++)
        {

            std::string cur_scan_pos = photos_group.getObjectName(i);
            HighFive::Group photo_group = getGroup(photos_group, cur_scan_pos);

            std::vector<ScanImage> cam_data;

            size_t num_photos = photo_group.getNumberObjects();
            for(size_t j=0; j< num_photos; j++)
            {
                ScanImage cam = getSingleRawCamData(i, j, load_image_data);
                cam_data.push_back(cam);
            }

            ret.push_back(cam_data);
        }

    }

    return ret;
}

ScanPtr HDF5IO::getSingleRawScan(int nr, bool load_points)
{
    ScanPtr ret(new Scan());

    if (m_hdf5_file)
    {
        char buffer[128];
        sprintf(buffer, "position_%05d", nr);

        string nr_str(buffer);
        std::string groupName         = "/raw/scans/"  + nr_str;
        std::string spectralGroupName = "/annotation/" + nr_str;

        unsigned int dummy;
        doubleArr fov            = getArray<double>(groupName, "fov", dummy);
        doubleArr res            = getArray<double>(groupName, "resolution", dummy);
        doubleArr pose_estimate = getArray<double>(groupName, "initialPose", dummy);
        doubleArr registration  = getArray<double>(groupName, "finalPose", dummy);
        floatArr bb             = getArray<float>(groupName, "boundingBox", dummy);

        if (load_points || m_usePreviews)
        {
            if (!load_points)
            {
                groupName         = "/preview/" + nr_str;
                spectralGroupName = groupName;
            }

            floatArr points    = getArray<float>(groupName, "points", dummy);

            if (points)
            {
                ret->points = PointBufferPtr(new PointBuffer(points, dummy/3));

                std::vector<size_t> dim;
                ucharArr spectral = getArray<unsigned char>(spectralGroupName, "spectral", dim);

                if (spectral)
                {
                    ret->points->addUCharChannel(spectral, "spectral_channels", dim[0], dim[1]);
                    ret->points->addIntAtomic(400, "spectral_wavelength_min");
                    ret->points->addIntAtomic(400 + 4 * dim[1], "spectral_wavelength_max");
                }
            }
        }

        if (fov)
        {
            // ret->m_hFieldOfView = fov[0];
            // ret->m_vFieldOfView = fov[1];
        }

        if (res)
        {
            ret->hResolution = res[0];
            ret->vResolution = res[1];
        }

        if (registration)
        {
            ret->registration = Transformd(registration.get());
        }

        if (pose_estimate)
        {
            ret->poseEstimation = Transformd(pose_estimate.get());
        }

        if (bb)
        {
            ret->boundingBox = BoundingBox<BaseVector<float> >(
                    BaseVector<float>(bb[0], bb[1], bb[2]), BaseVector<float>(bb[3], bb[4], bb[5]));
        }

        ret->pointsLoaded = load_points;
        ret->positionNumber = nr;

        ret->scanRoot = groupName;
    }

    return ret;
}


ScanImage HDF5IO::getSingleRawCamData(int scan_id, int img_id, bool load_image_data)
{
    ScanImage ret;
     
    if (m_hdf5_file)
    {
        char buffer1[128];
        sprintf(buffer1, "position_%05d", scan_id);
        string scan_id_str(buffer1);
        char buffer2[128];
        sprintf(buffer2, "photo_%05d", img_id);
        string img_id_str(buffer2);


        std::string groupName  = "/raw/photos/"  + scan_id_str + "/" + img_id_str;
        
        HighFive::Group g;
        
        try
        {
            g = getGroup(groupName);
        }
        catch(HighFive::Exception& e)
        {
            std::cout << timestamp << "Error getting cam data: "
                    << e.what() << std::endl;
            throw e;
        }

        unsigned int dummy;
        doubleArr intrinsics_arr = getArray<double>(groupName, "intrinsics", dummy);
        doubleArr extrinsics_arr = getArray<double>(groupName, "extrinsics", dummy);
        
        if(intrinsics_arr)
        {
            //ret.camera.setIntrinsics(Intrinsicsd(intrinsics_arr.get()));
        }

        if(extrinsics_arr)
        {
            //ret.camera.setExtrinsics(Extrinsicsd(extrinsics_arr.get()));
        }

        if(load_image_data)
        {
            getImage(g, "image", ret.image);
        }

    }

    return ret;
}

floatArr HDF5IO::getFloatChannelFromRawScan(std::string name, int nr, unsigned int& n, unsigned& w)
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
                "HDF5IO - getFloatchannelFromRawScan() Error: dim.size() != 2");
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
    const char* interlace = "INTERLACE_PIXEL";

    if(img.type() == CV_8U)
    {
        // 1 channel image
        H5IMmake_image_8bit(g.getId(), datasetName.c_str(), w, h, img.data);
    } else if(img.type() == CV_8UC3) {
        // 3 channel image
        H5IMmake_image_24bit(g.getId(), datasetName.c_str(), w, h, interlace, img.data);
    }

}

void HDF5IO::getImage(HighFive::Group& g, std::string datasetName, cv::Mat& img)
{
    long long unsigned int w,h,planes;
    long long int npals;
    char interlace[256];

    H5IMget_image_info(g.getId(), datasetName.c_str(), &w, &h, &planes, interlace, &npals);

    if(planes == 1)
    {
        // 1 channel image
        img = cv::Mat(h, w, CV_8U);
    } else if(planes == 3) {
        // 3 channel image
        img = cv::Mat(h, w, CV_8UC3);
    }

    H5IMread_image(g.getId(), datasetName.c_str(), img.data);
}

void HDF5IO::addFloatChannelToRawScan(
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

// void HDF5IO::addHyperspectralCalibration(int position, const HyperspectralPanorama& calibration)
// {
//     try
//     {
//         HighFive::Group g = getGroup("raw/spectral");
//     }
//     catch(HighFive::Exception& e)
//     {
//         std::cout << timestamp << "Error adding hyperspectral calibration data: "
//                   << e.what() << std::endl;
//         throw e;
//     }

//     // Add calibration values
//     if(m_hdf5_file)
//     {
//         // Setup group for scan data
//         char buffer[128];
//         sprintf(buffer, "position_%05d", position);
//         string nr_str(buffer);
//         std::string groupName = "/raw/spectral/" + nr_str;

//         floatArr a(new float[3]);
//         a[0] = calibration.distortion(1, 0);
//         a[1] = calibration.distortion(2, 0);
//         a[2] = calibration.distortion(3, 0);

//         floatArr rotation(new float[3]);
//         a[0] = calibration.rotation(1, 0);
//         a[1] = calibration.rotation(2, 0);
//         a[2] = calibration.rotation(3, 0);

//         floatArr origin(new float[3]);
//         origin[0] = calibration.origin(1, 0);
//         origin[1] = calibration.origin(2, 0);
//         origin[2] = calibration.origin(3, 0);

//         floatArr principal(new float[2]);
//         principal[0] = calibration.principal(1, 0);
//         principal[1] = calibration.principal(2, 0);

//         addArray(groupName, "distortion", 3, a);
//         addArray(groupName, "rotation", 3, rotation);
//         addArray(groupName, "origin", 3, origin);
//         addArray(groupName, "prinzipal", 2, principal);
//     }
// }

void HDF5IO::addRawScan(int nr, ScanPtr scan)
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
        if(scan->points->numPoints())
        {
            // Setup group for scan data
            char buffer[128];
            sprintf(buffer, "position_%05d", nr);
            string nr_str(buffer);


            std::string groupName = "/raw/scans/" + nr_str;

            // Generate tuples for field of view and resolution parameters
            floatArr fov(new float[2]);
            // fov[0] = scan->m_hFieldOfView;
            // fov[1] = scan->m_vFieldOfView;

            floatArr res(new float[2]);
            res[0] = scan->hResolution;
            res[1] = scan->vResolution;

            // Generate pose estimation matrix array
            float* pose_data = new float[16];
            float* reg_data = new float[16];

            std::copy(scan->poseEstimation.data(), scan->poseEstimation.data() + 16, pose_data);
            std::copy(scan->registration.data(), scan->registration.data() + 16, reg_data);

            floatArr pose_estimate(pose_data);
            floatArr registration(reg_data);

            // Generate bounding box representation
            floatArr bb(new float[6]);

            auto bb_min = scan->boundingBox.getMin();
            auto bb_max = scan->boundingBox.getMax();
            bb[0] = bb_min.x;
            bb[1] = bb_min.y;
            bb[2] = bb_min.z;

            bb[3] = bb_max.x;
            bb[4] = bb_max.y;
            bb[5] = bb_max.z;

            // Testing code to store point data as integers
//            cout << "Copy float to int..." << endl;
//            intArray ints(new int[scan->m_points->numPoints() * 3]);
//            floatArr tmp_pts = scan->m_points->getPointArray();
//            for(size_t i = 0; i < scan->m_points->numPoints() * 3; i++)
//            {
//                ints[i] = (int)tmp_pts[i] * 10000;
//            }
//            cout << "Done" << endl;


            // Add data to group
            std::vector<size_t> dim = {4,4};
            std::vector<size_t> scan_dim = {scan->points->numPoints(), 3};
            addArray(groupName, "fov", 2, fov);
            addArray(groupName, "resolution", 2, res);
            addArray(groupName, "initialPose", dim, pose_estimate);
            addArray(groupName, "finalPose", dim, registration);
            addArray(groupName, "boundingBox", 6, bb);
            addArray(groupName, "points", scan_dim, scan->points->getPointArray());

            // Uncomment this to store interger points
            // addArray(groupName, "points", scan_dim, ints);


            // Add spectral annotation channel
            size_t an;
            size_t aw;
            ucharArr spectral = scan->points->getUCharArray("spectral_channels", an, aw);

            if (spectral)
            {
                size_t chunk_w = std::min<size_t>(an, 1000000);    // Limit chunk size
                std::vector<hsize_t> chunk_annotation = {chunk_w, aw};
                std::vector<size_t> dim_annotation = {an, aw};
                addArray("/annotation/" + nr_str, "spectral", dim_annotation, chunk_annotation, spectral);
            }

            // Add preview data if wanted
            if (m_usePreviews)
            {
                std::string previewGroupName = "/preview/" + nr_str;


                // Add point preview
                floatArr points = scan->points->getPointArray();
                if (points)
                {
                    size_t numPreview;
                    floatArr previewData = reduceData(points, scan->points->numPoints(), 3, m_previewReductionFactor, &numPreview);

                    std::vector<size_t> previewDim = {numPreview, 3};
                    addArray(previewGroupName, "points", previewDim, previewData);
                }


                // Add spectral preview
                if (spectral)
                {

                    size_t numPreview;
                    ucharArr previewData = reduceData(spectral, an, aw, m_previewReductionFactor, &numPreview);
                    std::vector<size_t> previewDim = {numPreview, aw};
                    addArray(previewGroupName, "spectral", previewDim, previewData);
                }
            }
        }
    }
}

void HDF5IO::addRawCamData( int scan_id, int img_id, ScanImage& cam_data )
{
    if(m_hdf5_file)
    {

        char buffer1[128];
        sprintf(buffer1, "position_%05d", scan_id);
        string scan_id_str(buffer1);

        char buffer2[128];
        sprintf(buffer2, "photo_%05d", img_id);
        string photo_id_str(buffer2);

        std::string groupName = "/raw/photos/" + scan_id_str + "/" + photo_id_str;

        HighFive::Group photo_group;

        try
        {
            photo_group = getGroup(groupName);
        }
        catch(HighFive::Exception& e)
        {
            std::cout << timestamp << "Error adding raw image data: "
                    << e.what() << std::endl;
            throw e;
        }
        
        // add image to scan_image_group
        doubleArr intrinsics_arr(new double[9]);
        //Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(intrinsics_arr.get()) = cam_data.camera.intrinsics();


        doubleArr extrinsics_arr(new double[16]);
        //Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(extrinsics_arr.get()) = cam_data.camera.extrinsics();

        std::vector<size_t> dim_4 = {4,4};
        std::vector<size_t> dim_3 = {3,3};

        std::vector<hsize_t> chunks;
        for(auto i: dim_4)
        {
                chunks.push_back(i);
        }

        addArray(photo_group, "intrinsics", dim_4, chunks, intrinsics_arr);
        addArray(photo_group, "extrinsics", dim_3, chunks, extrinsics_arr);
        addImage(photo_group, "image", cam_data.image);

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

HighFive::Group HDF5IO::getGroup(HighFive::Group& g, const std::string &groupName, bool create)
{
    std::vector<std::string> groupNames = splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {

        for (size_t i = 0; i < groupNames.size(); i++)
        {

            if (g.exist(groupNames[i]))
            {
                cur_grp = g.getGroup(groupNames[i]);
            }
            else if (create)
            {
                cur_grp = g.createGroup(groupNames[i]);
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

boost::optional<HighFive::Group> HDF5IO::getMeshGroup(bool create){
    if(!create && !exist(m_mesh_path)){
        std::cout << timestamp << " No mesh with the part name \""
                  << m_part_name << "\" given in the HDF5 file \"" << std::endl;
        return boost::none;
    }
    return getGroup(m_mesh_path);
}


FloatChannelOptional HDF5IO::getVertices(){
    auto mesh_opt = getMeshGroup();
    if(!mesh_opt) return boost::none;
    auto mesh = mesh_opt.get();
    if(!mesh.exist(vertices_name))
    {
        std::cout << timestamp << " Could not find mesh vertices in the given HDF5 file." << std::endl;
        return boost::none;
    }

    std::vector<size_t >dims;
    auto values = getArray<float>(mesh, vertices_name, dims);
    return FloatChannel(dims[0], dims[1], values);
}


IndexChannelOptional HDF5IO::getIndices(){
    auto mesh_opt = getMeshGroup();
    if(!mesh_opt) return boost::none;
    auto mesh = mesh_opt.get();
    if(!mesh.exist(indices_name))
    {
        std::cout << timestamp << " Could not find mesh face indices in the given HDF5 file." << std::endl;
        return boost::none;
    }

    std::vector<size_t >dims;
    auto values = getArray<unsigned int>(mesh, indices_name, dims);
    return IndexChannel(dims[0], dims[1], values);
}

bool HDF5IO::addVertices(const FloatChannel& channel){
    auto mesh = getMeshGroup(true).get();
    std::vector<size_t > dims = {channel.numElements(), channel.width()};
    addArray<float>(m_mesh_path, vertices_name, dims, channel.dataPtr());
    return true;
}

bool HDF5IO::addIndices(const IndexChannel& channel){
    auto mesh = getMeshGroup(true).get();
    std::vector<size_t > dims = {channel.numElements(), channel.width()};
    addArray<unsigned int>(m_mesh_path, indices_name, dims, channel.dataPtr());
    return true;
}


bool HDF5IO::getChannel(const std::string group, const std::string name, FloatChannelOptional& channel){
    return getChannel<float>(group, name, channel);
}

bool HDF5IO::getChannel(const std::string group, const std::string name, IndexChannelOptional& channel){
    return getChannel<unsigned int>(group, name, channel);
}

bool HDF5IO::getChannel(const std::string group, const std::string name, UCharChannelOptional& channel){
    return getChannel<unsigned char>(group, name, channel);
}

bool HDF5IO::addChannel(const std::string group, const std::string name, const FloatChannel& channel){
    return addChannel<float>(group, name, channel);
}

bool HDF5IO::addChannel(const std::string group, const std::string name, const IndexChannel& channel){
    return addChannel<unsigned int>(group, name, channel);
}

bool HDF5IO::addChannel(const std::string group, const std::string name, const UCharChannel& channel){
    return addChannel<unsigned char>(group, name, channel);
}

} // namespace lvr2
