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
 * Main.cpp
 *
 *  Created on: Aug 9, 2013
 *      Author: Thomas Wiemann
 */

#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <utility>
#include <iterator>
using namespace std;

#include <boost/filesystem.hpp>


#include <Eigen/Dense>

#include "Options.hpp"

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#ifdef LVR2_USE_PCL
#include "lvr2/reconstruction/PCLFiltering.hpp"
#endif

#define BUF_SIZE 1024

namespace slam6dmerger
{

using namespace lvr2;

boost::filesystem::path getCorrespondingPath(const boost::filesystem::path& scan, const string& extension)
{
    std::stringstream ss;
    ss << scan.stem().string() << extension;
    return boost::filesystem::path(ss.str());
}



} // namespace slam6dmerger


int main(int argc, char** argv)
{
    using namespace slam6dmerger;
    using boost::filesystem::path;
    using boost::filesystem::directory_iterator;

    Options options(argc, argv);

    std::cout << options << std::endl;

    // CHECK PARAMETERS ---------------------------------------------------------------------------------

    path transformPath(options.getTransformFile());
    if(!exists(transformPath) || !is_regular_file(transformPath))
    {
        std::cout << timestamp << "Could not open transformation file " << options.getTransformFile() << std::endl;
        exit(-1);
    }

    Transformf transform = getTransformationFromFrames<float>(transformPath);
    //transform = inverseTransform(transform);
    BaseVector<float> transform_position;
    BaseVector<float> transform_angles;
    getPoseFromMatrix(transform_position, transform_angles, transform);

    std::cout << timestamp << "Transforming: " << std::endl << std::endl;
    std::cout << transform << std::endl << std::endl;

    path inputDir(options.getInputDir());
    if(!is_directory(inputDir))
    {
        std::cout << timestamp << "Input directory is not valid: " << options.getInputDir() << std::endl;
        exit(-1);
    }

    path outputDir(options.getOutputDir());
    if(!is_directory(inputDir))
    {
        std::cout << timestamp << "Output directory is not valid: " << options.getOutputDir() << std::endl;
        exit(-1);
    }

    if(inputDir == outputDir)
    {
        std::cout << timestamp << "Input directory and output directory should not be equal." << std::endl;
        exit(-1);
    }

    path mergeDir(options.getMergeDir());
    if(!is_directory(mergeDir))
    {
        std::cout << timestamp << "Merge directory is not valid: " << options.getMergeDir() << std::endl;
        exit(-1);
    }

    if(mergeDir == outputDir)
    {
        std::cout << timestamp << "Merge directory and output directory should not be equal." << std::endl;
        exit(-1);
    }


    /// PARSE DIRECTORIES  ---------------------------------------------------------------------------------

    vector<path>    input_scans;
    vector<path>    merge_scans;

    directory_iterator end;
    for(directory_iterator it(inputDir); it != end; ++it)
    {
        string extension = it->path().extension().string();
        if(extension == ".3d")
        {
            input_scans.push_back(it->path());
        }
    }

    for(directory_iterator it(mergeDir); it != end; ++it)
    {
        string extension = it->path().extension().string();
        if(extension == ".3d")
        {
            merge_scans.push_back(it->path());
        }
    }

    std::sort(input_scans.begin(), input_scans.end());
    std::sort(merge_scans.begin(),  merge_scans.end());

    // Copy files from input directory and merge directory
    // and assure consistent numbering
    int scan_counter = 0;
    char name_buffer[256];
    for(auto current_path : input_scans)
    {
        // -------->>>> SCAN FILE
        sprintf(name_buffer, "scan%03d.3d", scan_counter);
        path target_path = outputDir / path(name_buffer);
        std::cout << timestamp << "Copying " << current_path.string() << " to " << target_path.string() << "." << std::endl;
        boost::filesystem::copy(current_path, target_path);

        // -------->>>> OCT FILE

        path oct_in = inputDir / getCorrespondingPath(current_path, ".oct");
        if(exists(oct_in))
        {
            sprintf(name_buffer, "scan%03d.oct", scan_counter);
            path oct_out = outputDir / path(name_buffer);
            std::cout << timestamp << "Copying " << oct_in.string() << " to " << oct_out.string() << "." << std::endl;
            boost::filesystem::copy(oct_in, oct_out);
        }

        // -------->>>> FRAMES

        // Try to find frames file for current scan
        path frames_in = inputDir / getCorrespondingPath(current_path, ".frames");

        // Generate target path for frames file
        sprintf(name_buffer, "scan%03d.frames", scan_counter);
        path frames_out = outputDir / path(name_buffer);

        // Check for exisiting frames file
        if(!exists(frames_in))
        {
            std::cout << timestamp << "Warning: Could not find " << frames_in.string() << std::endl;
        }
        else
        {
            std::cout << timestamp << "Copying " << frames_in.string() << " to " << frames_out.string() << "." << std::endl;
            boost::filesystem::copy(frames_in, frames_out);
        }

        // ------->>>> POSE

        // Try to find pose file for current scan
        path pose_in = inputDir / getCorrespondingPath(current_path, ".pose");

        // Generate target path for frames file
        sprintf(name_buffer, "scan%03d.pose", scan_counter);
        path pose_out = outputDir / path(name_buffer);

        // Check for exisiting frames file
        if(!exists(pose_in))
        {
            std::cout << timestamp << "Warning: Could not find " << pose_in.string() << std::endl;
        }
        else
        {
            std::cout << timestamp << "Copying " << pose_in.string() << " to " << pose_out.string() << "." << std::endl;
            boost::filesystem::copy(pose_in, pose_out);
        }


        scan_counter++;
    }

    for(auto current_path : merge_scans)
    {
        // -------->>>> SCAN
        // Copy scan file
        sprintf(name_buffer, "scan%03d.3d", scan_counter);
        path target_path = outputDir / path(name_buffer);
        std::cout << timestamp << "Copying " << current_path.string() << " to " << target_path.string() << "." << std::endl;
        boost::filesystem::copy(current_path, target_path);

        // -------->>>> OCT FILE

        path oct_in = inputDir / getCorrespondingPath(current_path, ".oct");
        if(exists(oct_in))
        {
            sprintf(name_buffer, "scan%03d.oct", scan_counter);
            path oct_out = outputDir / path(name_buffer);
            std::cout << timestamp << "Copying " << oct_in.string() << " to " << oct_out.string() << "." << std::endl;
            boost::filesystem::copy(oct_in, oct_out);
        }

        // -------->>>> FRAMES

        // Try to find frames file for current scan
        path frames_in = mergeDir / getCorrespondingPath(current_path, ".frames");

        // Generate target path for frames file
        sprintf(name_buffer, "scan%03d.frames", scan_counter);
        path frames_out = outputDir / path(name_buffer);

        // Check for exisiting frames file
        if(!exists(frames_in))
        {
            std::cout << timestamp << "Warning: Could not find " << frames_in.string() << std::endl;
        }
        else
        {
            // Get transformation from file and transform
            std::cout << timestamp << "Transforming " << frames_in.string() << std::endl;
            Transformf registration = getTransformationFromFrames<float>(frames_in);
            //registration *= transform;
            Transformf t_reg = transformRegistration<float>(transform, registration);

            std::cout << timestamp << "Writing transformed registration to " << frames_out.string() << std::endl;
            writeFrame(t_reg, frames_out);

        }

        // ------->>>> POSE

        path pose_in = mergeDir / getCorrespondingPath(current_path, ".pose");

        // Generate target path for frames file
        sprintf(name_buffer, "scan%03d.pose", scan_counter);
        path pose_out = outputDir / path(name_buffer);

        // Check for exisiting frames file
        if(!exists(frames_in))
        {
            std::cout << timestamp << "Warning: Could not find " << frames_in.string() << std::endl;
        }
        else
        {
            // Get transformation from file and transform
            std::cout << timestamp << "Transforming " << pose_in.string() << std::endl;
            BaseVector<float> pos;
            BaseVector<float> ang;
            getPoseFromFile(pos, ang, pose_in);

            pos += transform_position;
            ang += transform_angles;

            std::cout << timestamp << "Writing transformed pose estimat to " << pose_out.string() << std::endl;
            writePose(pos, ang, pose_out);

        }


        scan_counter++;
    }

    return 0;
}
