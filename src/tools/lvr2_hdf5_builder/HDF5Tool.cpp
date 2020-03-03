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

/**
 * @file       HDF5Tool.cpp
 * @brief      Reads spectral PNGs and point clouds and writes them into a
 *             HDF5 file.
 * @details    
 * @author     Thomas Wiemann
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <string.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/HDF5IO.hpp"
#include "lvr2/types/Hyperspectral.hpp"
#include "lvr2/display/ColorMap.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/IOUtils.hpp"

#include "Options.hpp"

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace lvr2;
using boost::filesystem::path;
using boost::filesystem::directory_iterator;

bool compare_path(boost::filesystem::path p1, boost::filesystem::path p2)
{
    std::string ply_file_name1 = p1.stem().string();
    std::string number1 = ply_file_name1.substr(15);

    std::string ply_file_name2 = p2.stem().string();
    std::string number2 = ply_file_name2.substr(15);

    std::stringstream ss1(number1);
    std::stringstream ss2(number2);
    int first1 = 0;
    int second1 = 0;

    int first2 = 0;
    int second2 = 0;
    char dummy;
    ss1 >> first1 >> dummy >> second1;
    ss2 >> first2 >> dummy >> second2;

    return ((first1 << 16) + second1) < ((first2 << 16) + second2);
}

bool checkPNGDir(path& dataDir, std::string number, int numExspected)
{
    bool consistency = true;
    path png_dir = dataDir/"panoramas_fixed"/("panorama_channels_"+number);
    try
    {
        int numPNGs = std::count_if(
            directory_iterator(png_dir),
            directory_iterator(),
            static_cast<bool(*)(const path&)>(boost::filesystem::is_regular_file) );

        if(numPNGs != numExspected)
        {
            consistency = false;
        }
    }
    catch(boost::filesystem::filesystem_error)
    {
        consistency = false;
    }
    return consistency;
}

// HyperspectralPanorama getSpectralCalibration(path& dataDir, std::string number)
// {
//     HyperspectralPanorama pano;
//     path calibrationFile = dataDir/("calibration_"+number+".txt");
//     std::ifstream in(calibrationFile.string());
//     if(in.good())
//     {
//         in >> pano.distortion(0, 0) >> pano.distortion(1, 0) >> pano.distortion(2, 0);
//         in >> pano.rotation(0, 0)   >> pano.rotation(1, 0)   >> pano.rotation(2, 0);
//         in >> pano.origin(0, 0)     >> pano.origin(1, 0)     >> pano.origin(2, 0);
//         in >> pano.principal(0, 0);
//     }
//     else
//     {
//         std::cout << timestamp << "Could not open calibration file "
//                   << calibrationFile.string() << std::endl;
//     }
//     return pano;
// }

int main( int argc, char ** argv )
{
    hdf5tool::Options options(argc, argv);

    path dataDir(options.getDataDir());

    HDF5IO hdf5("hyper.h5", true);

    // Find all annotated scans and sort them
    vector<boost::filesystem::path> annotated_scans;
    directory_iterator end;
    for(directory_iterator it(dataDir); it != end; ++it)
    {
        std::string ext = it->path().extension().string();
        std::string stem = it->path().stem().string();
        if(ext == ".ply" && stem.find("scan_annotated_") != string::npos)
        {
            annotated_scans.push_back(it->path());
        }
    }
    std::sort(annotated_scans.begin(), annotated_scans.end(), compare_path);

    // Create scan data objects and add them to hdf5 file
    int scanNr = 0;
    for(auto it : annotated_scans)
    {


        std::string ply_file_name = it.stem().string();
        std::string number = ply_file_name.substr(15);
        size_t numExspectedPNGs = (size_t)options.numPanoramaImages();
        // Check panoram dir for correct number of scans
        if(checkPNGDir(dataDir, number, numExspectedPNGs))
        {
            // Read transformation
            path matrix_file = dataDir/path("scan_" + number + "_transformation.txt");
            std::cout << timestamp << "Reading transformation: " << matrix_file.string() << std::endl;
            Transformd transformation = loadFromFile<double>(matrix_file.string());

            // Read scan data
            std::cout << timestamp << "Reading scan data: " << it << std::endl;
            ModelPtr model = ModelFactory::readModel(it.string());

            // Compute bounding box
            PointBufferPtr pointCloud = model->m_pointCloud;

            std::cout << timestamp << "Calculating bounding box..." << std::endl;
            BoundingBox<BaseVector<float> > bBox;
            floatArr points = pointCloud->getPointArray();
            for(int i = 0; i < pointCloud->numPoints(); i++)
            {
                bBox.expand(BaseVector<float>(
                                points[3 * i],
                                points[3 * i + 1],
                                points[3 * i + 2]));
            }

            // Setup scan data object
            ScanPtr data = ScanPtr(new Scan());
            data->m_points = pointCloud;
            data->m_boundingBox = bBox;
            data->m_registration = transformation;

            std::cout << timestamp << " Adding raw scan data" << endl;
            // Add objects to hdf5 file
            hdf5.addRawScan(scanNr, data);


            // Get hyperspectral calibration parameters
            // HyperspectralPanorama cal = getSpectralCalibration(dataDir, number);
            // hdf5.addHyperspectralCalibration(scanNr, cal);

            // Create "hyperspectral cube"
            path imgFile = dataDir/"panoramas_fixed"/("panorama_channels_"+number)/"channel0.png";
            cv::Mat img = cv::imread(imgFile.string(), cv::IMREAD_GRAYSCALE);

            size_t img_x = img.cols;
            size_t img_y = img.rows;
            unsigned char* cube = new unsigned char[numExspectedPNGs * img.rows * img.cols];
            for(int i = 0; i < numExspectedPNGs; i++)
            {
                char buffer[256];
                sprintf(buffer, "channel%d.png", i);
                path imgFile = dataDir/"panoramas_fixed"/("panorama_channels_"+number)/buffer;
                cv::Mat img = cv::imread(imgFile.string(),  cv::IMREAD_GRAYSCALE);
                memcpy(cube + i * (img_y * img_x), img.data, img_y * img_x * sizeof(unsigned char));
            }

            char groupName[256];
            std::vector<size_t> dim = {numExspectedPNGs, img_y, img_x};

            // Priliminary experiments showed that this chunking delivered
            // best compression of hyperspectral image data
            std::vector<hsize_t> chunks =
                {options.getHSPChunk0(), options.getHSPChunk1(), options.getHSPChunk2()};

            sprintf(groupName, "/raw/spectral/position_%05d", scanNr);
            std::cout << timestamp << "Adding spectral dataset to " << groupName << " with dims "
                      << options.getHSPChunk0() << " " <<  options.getHSPChunk1() << " " << options.getHSPChunk2() << endl;

            hdf5.addArray(groupName, "spectral", dim, chunks, ucharArr(cube));


            scanNr++;
        }
        else
        {
            std::cout << timestamp << "Will not add data from "
                      << ply_file_name <<". Spectral data is not consistent." << std::endl;
        }
    }

    return 0;
}
//    // Parse command line arguments
//    hdf5tool::Options options(argc, argv);

//    std::cout << options << ::std::endl;

//    // Get directory with hyperspectral PNG files
//    boost::filesystem::path pngPath(options.getPNGDir());
//    boost::filesystem::directory_iterator end;

//    // Count files in directory
//    int numPNGs = 0;
//    for(boost::filesystem::directory_iterator it(pngPath); it != end; ++it)
//    {
//        std::string ext = it->path().extension().string();
//        if(ext == ".png")
//        {
//            numPNGs++;
//        }
//    }

//    // Sort files according to channel nummer
//    char buffer[512];
//    std::vector<string> pngFiles(numPNGs);
//    int channel_nr = 0;
//    for(boost::filesystem::directory_iterator it(pngPath); it != end; ++it)
//    {
//        std::string ext = it->path().extension().string();
//        if(ext == ".png")
//        {
//            string pngFilePath = it->path().string();
//            string pngFileName = it->path().filename().string();
//            sscanf(pngFileName.c_str(), "channel%d.png", &channel_nr);
//            pngFiles[channel_nr] = pngFilePath;
//        }
//    }

//    // Try to open the given HDF5 file
//    HighFive::File hdf5_file(
//                "out.h5",
//                HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

//    if (!hdf5_file.isValid())
//    {
//        throw "Could not open file.";
//    }

//    // Generate high level structure of nested groups for raw scan and spectral
//    // data within the hdf5 file
//    HighFive::Group raw_data_group = hdf5_file.createGroup("/raw_data");
//    HighFive::Group pose_group = hdf5_file.createGroup("raw_data/pose001");
//    HighFive::Group spectral_group = hdf5_file.createGroup("raw_data/pose001/spectral");
//    HighFive::Group scan_group = hdf5_file.createGroup("raw_data/pose001/scan");

//    // Write information about number of channels and sprectral range
//    int numChannels = 150;
//    int minSpectral = 400;
//    int maxSpectral = 1000;
//    spectral_group.createDataSet<int>("numChannels", HighFive::DataSpace::From(numChannels)).write(numChannels);
//    spectral_group.createDataSet<int>("minSpectral", HighFive::DataSpace::From(minSpectral)).write(minSpectral);
//    spectral_group.createDataSet<int>("maxSpectral", HighFive::DataSpace::From(maxSpectral)).write(maxSpectral);

//    // Create groups for the spectral pngs if the single channles
//    int i = 0;
//    for(auto it : pngFiles)
//    {
//        // Parse channel name
//        sprintf(buffer, "channel%03d", i);
//        string group_name = "raw_data/pose001/spectral/" + string(buffer);

//        // Create group
//        HighFive::Group channel_group = hdf5_file.createGroup(group_name);

//        // Read png images and write them into the created group
//        cv::Mat image = cv::imread(it, cv::IMREAD_GRAYSCALE);

//        int w = image.cols;
//        int h = image.rows;

//        channel_group.createDataSet<int>("width", HighFive::DataSpace::From(w)).write(w);
//        channel_group.createDataSet<int>("height", HighFive::DataSpace::From(h)).write(h);
//        H5IMmake_image_8bit(channel_group.getId(), "spectral_image", w, h, image.data);
//        i++;
//    }

//    // Read .ply file with scan data
//    ModelPtr model = ModelFactory::readModel(options.getPLYFile());
//    PointBufferPtr point_buffer = model->m_pointCloud;
//    size_t n = point_buffer->numPoints();
//    floatArr points = point_buffer->getPointArray();

//    // Write scan data to group
//    scan_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n)).write(n);
//    scan_group.createDataSet<float>("points", HighFive::DataSpace(n * 3)).write(points.get());

//    size_t n_spec;
//    unsigned n_channels;
//    floatArr spec = point_buffer->getFloatArray("spectral_channels", n_spec, n_channels);

//    if (n_spec)
//    {
//        HighFive::Group pointclouds_group = hdf5_file.createGroup("/pointclouds");
//        HighFive::Group cloud001_group = hdf5_file.createGroup("pointclouds/cloud001");
//        HighFive::Group cloud001_points_group = hdf5_file.createGroup("pointclouds/cloud001/points");
//        HighFive::Group cloud001_spectral_group = hdf5_file.createGroup("pointclouds/cloud001/spectralChannels");

//        cloud001_points_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n_spec)).write(n_spec);
//        cloud001_points_group.createDataSet<float>("points", HighFive::DataSpace(n_spec * 3)).write(points.get());

//        cloud001_spectral_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n_spec)).write(n_spec);
//        cloud001_spectral_group.createDataSet<int>("numChannels", HighFive::DataSpace::From(n_channels)).write(n_channels);
//        cloud001_spectral_group.createDataSet<int>("minSpectral", HighFive::DataSpace::From(minSpectral)).write(minSpectral);
//        cloud001_spectral_group.createDataSet<int>("maxSpectral", HighFive::DataSpace::From(maxSpectral)).write(maxSpectral);
//        cloud001_spectral_group.createDataSet<float>("spectralChannels", HighFive::DataSpace(n_spec * n_channels)).write(spec.get());
//    }

//    std::cout << "Done" << std::endl;

