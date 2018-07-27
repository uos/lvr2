/*******************************************************************************
 * Copyright © 2011 Universität Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA  02111-1307, USA
 ******************************************************************************/


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

#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/display/ColorMap.hpp>
#include "Options.hpp"

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <H5Tpublic.h>
#include <hdf5_hl.h>
#include <highfive/H5File.hpp>

using namespace lvr;
using namespace std;


int main( int argc, char ** argv )
{

    // Parse command line arguments
    hdf5tool::Options options(argc, argv);

    ::std::cout << options << ::std::endl;

    // Get directory with hyperspectral PNG files
    boost::filesystem::path pngPath(options.getPNGDir());
    boost::filesystem::directory_iterator end;

    // Count files in directory
    int numPNGs = 0;
    for(boost::filesystem::directory_iterator it(pngPath); it != end; ++it)
    {
        std::string ext = it->path().extension().string();
        if(ext == ".png")
        {
            numPNGs++;
        }
    }

    // Sort files according to channel nummer
    char buffer[512];
    std::vector<string> pngFiles(numPNGs);
    int channel_nr = 0;
    for(boost::filesystem::directory_iterator it(pngPath); it != end; ++it)
    {
        std::string ext = it->path().extension().string();
        if(ext == ".png")
        {
            string pngFilePath = it->path().string();
            string pngFileName = it->path().filename().string();
            sscanf(pngFileName.c_str(), "channel%d.png", &channel_nr);
            pngFiles[channel_nr] = pngFilePath;
        }
    }

    // Try to open the given HDF5 file
    HighFive::File hdf5_file(
                "out.h5",
                HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    if (!hdf5_file.isValid())
    {
        throw "Could not open file.";
    }

    // Generate high level structure of nested groups for raw scan and spectral
    // data within the hdf5 file
    HighFive::Group raw_data_group = hdf5_file.createGroup("/raw_data");
    HighFive::Group pose_group = hdf5_file.createGroup("raw_data/pose001");
    HighFive::Group spectral_group = hdf5_file.createGroup("raw_data/pose001/spectral");
    HighFive::Group scan_group = hdf5_file.createGroup("raw_data/pose001/scan");

    // Write information about number of channels and sprectral range
    int numChannels = 150;
    int minSpectral = 400;
    int maxSpectral = 1000;
    spectral_group.createDataSet<int>("numChannels", HighFive::DataSpace::From(numChannels)).write(numChannels);
    spectral_group.createDataSet<int>("minSpectral", HighFive::DataSpace::From(minSpectral)).write(minSpectral);
    spectral_group.createDataSet<int>("maxSpectral", HighFive::DataSpace::From(maxSpectral)).write(maxSpectral);

    // Create groups for the spectral pngs if the single channles
    int i = 0;
    for(auto it : pngFiles)
    {
        // Parse channel name
        sprintf(buffer, "channel%03d", i);
        string group_name = "raw_data/pose001/spectral/" + string(buffer);

        // Create group
        HighFive::Group channel_group = hdf5_file.createGroup(group_name);

        // Read png images and write them into the created group
        cv::Mat image = cv::imread(it, cv::IMREAD_GRAYSCALE);

        int w = image.cols;
        int h = image.rows;

        channel_group.createDataSet<int>("width", HighFive::DataSpace::From(w)).write(w);
        channel_group.createDataSet<int>("height", HighFive::DataSpace::From(h)).write(h);
        H5IMmake_image_8bit(channel_group.getId(), "spectral_image", w, h, image.data);
        i++;
    }

    // Read .ply file with scan data
    ModelPtr model = ModelFactory::readModel(options.getPLYFile());
    PointBufferPtr point_buffer = model->m_pointCloud;
    size_t n;
    floatArr points = point_buffer->getPointArray(n);

    // Write scan data to group
    scan_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n)).write(n);
    scan_group.createDataSet<float>("points", HighFive::DataSpace(n * 3)).write(points.get());

    size_t n_spec, n_channels;
    floatArr spec = point_buffer->getPointSpectralChannelsArray(n_spec, n_channels);

    if (n_spec)
    {
        HighFive::Group pointclouds_group = hdf5_file.createGroup("/pointclouds");
        HighFive::Group cloud001_group = hdf5_file.createGroup("pointclouds/cloud001");
        HighFive::Group cloud001_points_group = hdf5_file.createGroup("pointclouds/cloud001/points");
        HighFive::Group cloud001_spectral_group = hdf5_file.createGroup("pointclouds/cloud001/spectralChannels");

        cloud001_points_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n_spec)).write(n_spec);
        cloud001_points_group.createDataSet<float>("points", HighFive::DataSpace(n_spec * 3)).write(points.get());

        cloud001_spectral_group.createDataSet<int>("numPoints", HighFive::DataSpace::From(n_spec)).write(n_spec);
        cloud001_spectral_group.createDataSet<int>("numChannels", HighFive::DataSpace::From(n_channels)).write(n_channels);
        cloud001_spectral_group.createDataSet<float>("spectralChannels", HighFive::DataSpace(n_spec * n_channels)).write(spec.get());
    }

    std::cout << "Done" << std::endl;

}
