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
#include <lvr2/io/Timestamp.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Matrix4.hpp>

#ifdef LVR_USE_PCL
#include <lvr2/reconstruction/PCLFiltering.hpp>
#endif

#define BUF_SIZE 1024

namespace slam6dmerger
{

using namespace lvr2;

using Vector3f = BaseVector<float>;

size_t countPointsInFile(boost::filesystem::path& inFile)
{
    ifstream in(inFile.c_str());
    cout << timestamp << "Counting points in " << inFile.filename().string() << "..." << endl;

    // Count lines in file
    size_t n_points = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        n_points++;
    }
    in.close();

    cout << timestamp << "File " << inFile.filename().string() << " contains " << n_points << " points." << endl;

    return n_points;
}

Eigen::Matrix4d buildTransformation(double* alignxf)
{
    Eigen::Matrix3d rotation;
    Eigen::Vector4d translation;

    rotation  << alignxf[0],  alignxf[4],  alignxf[8],
    alignxf[1],  alignxf[5],  alignxf[9],
    alignxf[2],  alignxf[6],  alignxf[10];

    translation << alignxf[12], alignxf[13], alignxf[14], 1.0;

    Eigen::Matrix4d transformation;
    transformation.setIdentity();
    transformation.block<3,3>(0,0) = rotation;
    transformation.rightCols<1>() = translation;

    return transformation;
}

Eigen::Matrix4d getTransformationFromPose(boost::filesystem::path& pose)
{
    ifstream poseIn(pose.c_str());
    if(poseIn.good())
    {
        double rPosTheta[3];
        double rPos[3];
        double alignxf[16];

        poseIn >> rPos[0] >> rPos[1] >> rPos[2];
        poseIn >> rPosTheta[0] >> rPosTheta[1] >> rPosTheta[2];

        rPosTheta[0] *= 0.0174533;
        rPosTheta[1] *= 0.0174533;
        rPosTheta[2] *= 0.0174533;

        double sx = sin(rPosTheta[0]);
        double cx = cos(rPosTheta[0]);
        double sy = sin(rPosTheta[1]);
        double cy = cos(rPosTheta[1]);
        double sz = sin(rPosTheta[2]);
        double cz = cos(rPosTheta[2]);

        alignxf[0]  = cy*cz;
        alignxf[1]  = sx*sy*cz + cx*sz;
        alignxf[2]  = -cx*sy*cz + sx*sz;
        alignxf[3]  = 0.0;
        alignxf[4]  = -cy*sz;
        alignxf[5]  = -sx*sy*sz + cx*cz;
        alignxf[6]  = cx*sy*sz + sx*cz;
        alignxf[7]  = 0.0;
        alignxf[8]  = sy;
        alignxf[9]  = -sx*cy;
        alignxf[10] = cx*cy;

        alignxf[11] = 0.0;

        alignxf[12] = rPos[0];
        alignxf[13] = rPos[1];
        alignxf[14] = rPos[2];
        alignxf[15] = 1;

        return buildTransformation(alignxf);
    }
    else
    {
        return Eigen::Matrix4d::Identity();
    }
}

Eigen::Matrix4d getTransformationFromFrames(boost::filesystem::path& frames)
{
    double alignxf[16];
    int color;

    std::ifstream in(frames.c_str());
    int c = 0;
    while(in.good())
    {
        c++;
        for(int i = 0; i < 16; i++)
        {
            in >> alignxf[i];
        }

        in >> color;

        if(!in.good())
        {
            c = 0;
            break;
        }
    }

    return buildTransformation(alignxf);
}

Eigen::Matrix4d transformFrames(Eigen::Matrix4d frames)
{
    Eigen::Matrix3d basisTrans;
    Eigen::Matrix3d reflection;
    Eigen::Vector3d tmp;
    std::vector<Eigen::Vector3d> xyz;
    xyz.push_back(Eigen::Vector3d(1,0,0));
    xyz.push_back(Eigen::Vector3d(0,1,0));
    xyz.push_back(Eigen::Vector3d(0,0,1));

    reflection.setIdentity();


    // axis reflection
    frames.block<3,3>(0,0) *= reflection;

    // We are always transforming from the canonical base => T = (B')^(-1)
    basisTrans.col(0) = xyz[0];
    basisTrans.col(1) = xyz[1];
    basisTrans.col(2) = xyz[2];

    // Transform the rotation matrix
    frames.block<3,3>(0,0) = basisTrans.inverse() * frames.block<3,3>(0,0) * basisTrans;

    // Setting translation vector
    tmp = frames.block<3,1>(0,3);
    tmp = basisTrans.inverse() * tmp;

    (frames.rightCols<1>())(0) = tmp(0);
    (frames.rightCols<1>())(1) = tmp(1);
    (frames.rightCols<1>())(2) = tmp(2);
    (frames.rightCols<1>())(3) = 1.0;

    return frames;
}

boost::filesystem::path getCorrespondingPath(const boost::filesystem::path& scan, const string& extension)
{
    std::stringstream ss;
    ss << scan.stem().string() << extension;
    return boost::filesystem::path(ss.str());
}

void writeFrame(Eigen::Matrix4d transform, const boost::filesystem::path& framesOut)
{
    std::ofstream out(framesOut.c_str());

    // write the rotation matrix
    out << transform.col(0)(0) << " " << transform.col(0)(1) << " " << transform.col(0)(2) << " " << 0 << " "
        << transform.col(1)(0) << " " << transform.col(1)(1) << " " << transform.col(1)(2) << " " << 0 << " "
        << transform.col(2)(0) << " " << transform.col(2)(1) << " " << transform.col(2)(2) << " " << 0 << " ";

    // write the translation vector
    out << transform.col(3)(0) << " "
        << transform.col(3)(1) << " "
        << transform.col(3)(2) << " "
        << transform.col(3)(3);

    out << " 1";

    out.close();
}

void writePose(const Vector3f& position, const Vector3f& angles, const boost::filesystem::path& out)
{
    ofstream o(out.c_str());
    if(o.good())
    {
        o << position[0] << " " << position[1] << " " << position[2] << std::endl;
        o << angles[0] << " " << angles[1] << " " << angles[2];
    }
}

Eigen::Matrix4d inverseTransform(const Eigen::Matrix4d& transform)
{
    Eigen::Matrix3d rotation = transform.block<3,3>(0, 0);
    rotation.transposeInPlace();

    Eigen::Matrix4d inv;
    inv.block<3, 3>(0, 0) = rotation;

    (inv.rightCols<1>())(0) = -transform.col(3)(0);
    (inv.rightCols<1>())(1) = -transform.col(3)(1);
    (inv.rightCols<1>())(2) = -transform.col(3)(2);
    (inv.rightCols<1>())(3) = 1.0;

    return inv;
}


void getPoseFromFile(Vector3f& position, Vector3f& angles, const boost::filesystem::path file)
{
    ifstream in(file.c_str());
    if(in.good())
    {
        in >> position.x >> position.y >> position.z;
        in >> angles.y >> angles.y >> angles.z;
    }
    else
    {
        cout << timestamp << "Unable to open " << file.string() << endl;
    }
}

void transformToEuler(Vector3f& position, Vector3f& angles, Eigen::Matrix4d mat)
{
    double m[16];

    m[0]  = mat(0, 0);
    m[1]  = mat(0, 1);
    m[2]  = mat(0, 2);
    m[3]  = mat(0, 3);

    m[4]  = mat(1, 0);
    m[5]  = mat(1, 1);
    m[6]  = mat(1, 2);
    m[7]  = mat(1, 3);

    m[8]  = mat(2, 0);
    m[9]  = mat(2, 1);
    m[10] = mat(2, 2);
    m[11] = mat(2, 3);

    m[12] = mat(3, 0);
    m[13] = mat(3, 1);
    m[14] = mat(3, 2);
    m[15] = mat(3, 3);

    float _trX, _trY;
    if(m[0] > 0.0) {
       angles.y = asin(m[8]);
    } else {
       angles.y = (float)M_PI - asin(m[8]);
    }
    // rPosTheta[1] =  asin( m[8]);      // Calculate Y-axis angle

    float  C    =  cos(angles.y );
    if ( fabs( C ) > 0.005 )  {          // Gimball lock?
        _trX      =  m[10] / C;          // No, so get X-axis angle
        _trY      =  -m[9] / C;
        angles.x  = atan2( _trY, _trX );
        _trX      =  m[0] / C;           // Get Z-axis angle
        _trY      = -m[4] / C;
        angles.z  = atan2( _trY, _trX );
    } else {                             // Gimball lock has occurred
        angles.x = 0.0;                   // Set X-axis angle to zero
        _trX      =  m[5];  //1          // And calculate Z-axis angle
        _trY      =  m[1];  //2
        angles.z  = atan2( _trY, _trX );
    }

    //cout << angles.x << " " <<angles.y << " " << angles.z << endl;

    position.x = m[12];
    position.y = m[13];
    position.z = m[14];

}

Eigen::Matrix4d transformRegistration(const Eigen::Matrix4d& transform, const Eigen::Matrix4d& registration)
{
    Eigen::Matrix3d rotation_trans;
    Eigen::Matrix3d rotation_registration;

    rotation_trans = transform.block<3,3>(0, 0);
    rotation_registration = registration.block<3,3>(0, 0);

    Eigen::Matrix3d rotation = rotation_trans * rotation_registration;

    Eigen::Matrix4d result;
    result.block<3,3>(0, 0) = rotation;

    Eigen::Vector3d tmp;
    tmp = registration.block<3,1>(0,3);
    tmp = rotation_trans * tmp;

    (result.rightCols<1>())(0) = transform.col(3)(0) + tmp(0);
    (result.rightCols<1>())(1) = transform.col(3)(1) + tmp(1);
    (result.rightCols<1>())(2) = transform.col(3)(2) + tmp(2);
    (result.rightCols<1>())(3) = 1.0;

    return result;

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

    Eigen::Matrix4d transform = getTransformationFromFrames(transformPath);
    //transform = inverseTransform(transform);
    Vector3f transform_position;
    Vector3f transform_angles;
    transformToEuler(transform_position, transform_angles, transform);


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
            Eigen::Matrix4d registration = getTransformationFromFrames(frames_in);
            //registration *= transform;
            Eigen::Matrix4d t_reg = transformRegistration(transform, registration);

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
            Vector3f pos;
            Vector3f ang;
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
