/* Copyright (C) 2013 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 *
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

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include <Eigen/Dense>

#include "Options.hpp"
#include <lvr/io/BaseIO.hpp>
#include <lvr/io/DatIO.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/AsciiIO.hpp>
#ifdef LVR_USE_PCL
#include <lvr/reconstruction/PCLFiltering.hpp>
#endif

#define BUF_SIZE 1024

using namespace lvr;

namespace qi = boost::spirit::qi;

const kaboom::Options* options;

// This is dirty 
bool lastScan = false;

ModelPtr filterModel(ModelPtr p, int k, float sigma)
{
    if(p)
    {
        if(p->m_pointCloud)
        {
#ifdef LVR_USE_PCL
            PCLFiltering filter(p->m_pointCloud);
            cout << timestamp << "Filtering outliers with k=" << k << " and sigma=" << sigma << "." << endl;
            size_t original_size = p->m_pointCloud->getNumPoints();
            filter.applyOutlierRemoval(k, sigma);
            PointBufferPtr pb( filter.getPointBuffer() );
            ModelPtr out_model( new Model( pb ) );
            cout << timestamp << "Filtered out " << original_size - out_model->m_pointCloud->getNumPoints() << " points." << endl;
            return out_model;
#else
            cout << timestamp << "Can't create a PCL Filter without PCL installed." << endl;
            return NULL;
#endif

        }
    }
    return NULL;
}

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

void writeFrames(Eigen::Matrix4d transform, const boost::filesystem::path& framesOut)
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

    out.close();
}

size_t writeModel( ModelPtr model,const  boost::filesystem::path& outfile)
{
    size_t n_ip;
    floatArr arr = model->m_pointCloud->getPointArray(n_ip);

    // ModelFactory::saveModel(model, outfile.string());

    return n_ip;
}

size_t writeAscii(ModelPtr model, std::ofstream& out)
{
    size_t n_ip, n_colors;

    floatArr arr = model->m_pointCloud->getPointArray(n_ip);

    ucharArr colors = model->m_pointCloud->getPointColorArray(n_colors);
    for(int a = 0; a < n_ip; a++)
    {
        out << arr[a * 3] << " " << arr[a * 3 + 1] << " " << arr[a * 3 + 2];

        if(n_colors)
        {
            out << " " << (int)colors[a * 3] << " " << (int)colors[a * 3 + 1] << " " << (int)colors[a * 3 + 2];
        }
        out << endl;

    }

    return n_ip;
}

size_t writePly(ModelPtr model, std::fstream& out) 
{
    size_t n_ip, n_colors;

    floatArr arr = model->m_pointCloud->getPointArray(n_ip);

    ucharArr colors = model->m_pointCloud->getPointColorArray(n_colors);

    if(n_colors)
    {
        if(n_colors != n_ip)
        {
            std::cout << timestamp << "Numbers of points and colors needs to be identical" << std::endl;
            return 0;
        }

        for(int a = 0; a < n_ip; a++)
        {
            // x y z
            out.write((char*) (arr.get() + (3 * a)), sizeof(float) * 3);

            // r g b
            out.write((char*) (colors.get() + (3 * a)), sizeof(unsigned char) * 3);
        }
    }
    else
    {
        // simply write whole points array
        out.write((char*) arr.get(), sizeof(float) * n_ip * 3);
    }

    return n_ip;

}

size_t writePlyHeader(std::ofstream& out, size_t n_points, bool colors)
{
    out << "ply" << std::endl;
    out << "format binary_little_endian 1.0" << std::endl;

    out << "element point " << n_points << std::endl;
    out << "property float32 x" << std::endl;
    out << "property float32 y" << std::endl;
    out << "property float32 z" << std::endl;

    if(colors)
    {
        out << "property uchar red" << std::endl;
        out << "property uchar green" << std::endl;
        out << "property uchar blue" << std::endl;
    }
    out << "end_header" << std::endl;
}

int asciiReductionFactor(boost::filesystem::path& inFile)
{

    int reduction = options->getTargetSize();

    /*
     * If reduction is less than the number of points it will segfault
     * because the modulo operation is not defined for n mod 0
     * and we have to keep all points anyways.
     * Same if no targetSize was given.
     */
    if(reduction != 0)
    {
        // Count lines in file
        size_t n_points = countPointsInFile(inFile);

        if(reduction < n_points)
        {
            return (int)n_points / reduction;
        }
    }

    /* No reduction write all points */
    return 1;

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

    if(options->sx() < 0)
    {
        reflection.block<3,1>(0,0) = (-1) * xyz[0];
    }

    if(options->sy() < 0)
    {
        reflection.block<3,1>(0,1) = (-1) * xyz[1];
    }

    if(options->sz() < 0)
    {
        reflection.block<3,1>(0,2) = (-1) * xyz[2];
    }

    // axis reflection
    frames.block<3,3>(0,0) *= reflection;

    // We are always transforming from the canonical base => T = (B')^(-1)
    basisTrans.col(0) = xyz[options->x()];
    basisTrans.col(1) = xyz[options->y()];
    basisTrans.col(2) = xyz[options->z()];

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

void transformFromOptions(ModelPtr& model, int modulo)
{
    size_t n_ip, n_colors;
    size_t cntr = 0;

    floatArr arr = model->m_pointCloud->getPointArray(n_ip);
    ucharArr colors = model->m_pointCloud->getPointColorArray(n_colors);

    // Plus one because it might differ because of the 0-index
    // better waste memory for one float than having not enough space.
    // TO-DO think about exact calculation.
    size_t targetSize = (3 * ((n_ip)/modulo)) + modulo;
    floatArr points(new float[targetSize ]);
    ucharArr newColorsArr;

    if(n_colors)
    {
        newColorsArr = ucharArr(new unsigned char[targetSize]);
    }

    for(int i = 0; i < n_ip; i++)
    {
        if(i % modulo == 0)
        {
            if(options->sx() != 1)
            {
                arr[i * 3]         *= options->sx();
            }

            if(options->sy() != 1)
            {
                arr[i * 3 + 1]     *= options->sy();
            }

            if(options->sz() != 1)
            {
                arr[i * 3 + 2]     *= options->sz();
            }

            if((cntr * 3) < targetSize)
            {
                points[cntr * 3]     = arr[i * 3 + options->x()];
                points[cntr * 3 + 1] = arr[i * 3 + options->y()];
                points[cntr * 3 + 2] = arr[i * 3 + options->z()];
            }
            else
            {
                std::cout << "The following is for debugging purpose" << std::endl;
                std::cout << "Cntr: " << (cntr * 3) << " targetSize: " << targetSize << std::endl;
                std::cout << "nip : " << n_ip << " modulo " << modulo << std::endl;
                break;
            }
            
            if(n_colors)
            {
                newColorsArr[cntr * 3]     = colors[i * 3];
                newColorsArr[cntr * 3 + 1] = colors[i * 3 + 1];
                newColorsArr[cntr * 3 + 2] = colors[i * 3 + 2];
            }

            cntr++;
        }
    }

    // Pass counter because it is the actual number of points used after reduction
    // it might be 1 less than the size
    model->m_pointCloud->setPointArray(points, cntr);

    if(n_colors)
    {
        model->m_pointCloud->setPointColorArray(newColorsArr, cntr);
    }
}

// transforming with Matrix from frames/pose
void transformModel(ModelPtr model, Eigen::Matrix4d transformation)
{
    cout << timestamp << "Transforming model." << endl;
    size_t numPoints;

    floatArr arr = model->m_pointCloud->getPointArray(numPoints);

    for(int i = 0; i < numPoints; i++)
    {
        float x = arr[3 * i];
        float y = arr[3 * i + 1];
        float z = arr[3 * i + 2];

        Eigen::Vector4d v(x,y,z,1);
        Eigen::Vector4d tv = transformation * v;

        arr[3 * i]     = tv[0];
        arr[3 * i + 1] = tv[1];
        arr[3 * i + 2] = tv[2];
    }
}

void processSingleFile(boost::filesystem::path& inFile)
{
    cout << timestamp << "Processing " << inFile << endl;



    cout << timestamp << "Reading point cloud data from file " << inFile.filename().string() << "." << endl;

    ModelPtr model = ModelFactory::readModel(inFile.string());

    if(0 == model)
    {
        throw "ERROR: Could not create Model for: ";
    }

    if(options->getOutputFile() != "")
    { 
        char frames[1024];
        char pose[1024];
        sprintf(frames, "%s/%s.frames", inFile.parent_path().c_str(), inFile.stem().c_str());
        sprintf(pose, "%s/%s.pose", inFile.parent_path().c_str(), inFile.stem().c_str());

        boost::filesystem::path framesPath(frames);
        boost::filesystem::path posePath(pose);

        size_t reductionFactor = asciiReductionFactor(inFile);

        if(options->transformBefore())
        {
            transformFromOptions(model, reductionFactor);
        }

        if(boost::filesystem::exists(framesPath))
        {
            std::cout << timestamp << "Getting transformation from frame: " << framesPath << std::endl;
            Eigen::Matrix4d transform = getTransformationFromFrames(framesPath);
            transformModel(model, transform);
        }
        else if(boost::filesystem::exists(posePath))
        {

            std::cout << timestamp << "Getting transformation from pose: " << posePath << std::endl;
            Eigen::Matrix4d transform = getTransformationFromPose(posePath);
            transformModel(model, transform);
        }

        if(!options->transformBefore())
        {
            transformFromOptions(model, reductionFactor);
        }

        static size_t points_written = 0;


        if(options->getOutputFormat() == "ASCII" || options->getOutputFormat() == "")
        {
            // Merge (only ASCII)
            std::ofstream out;

            /* If points were written we want to append the next scans, otherwise we want an empty file */
            if(points_written != 0)
            {
                out.open(options->getOutputFile().c_str(), std::ofstream::out | std::ofstream::app);
            }
            else
            {
                out.open(options->getOutputFile().c_str(), std::ofstream::out | std::ofstream::trunc);
            }

            points_written += writeAscii(model, out);

            out.close();
        }
        else if(options->getOutputFormat() == "PLY")
        {
            char tmp_file[1024];

            sprintf(tmp_file, "%s/tmp.ply", inFile.parent_path().c_str());

            std::fstream tmp;

            if(points_written != 0)
            {
                tmp.open(tmp_file, std::fstream::in | std::fstream::out | std::fstream::app | std::fstream::binary);
            }
            else
            {
                tmp.open(tmp_file, std::fstream::in | std::fstream::out | std::fstream::trunc | std::fstream::binary);
            }
            
            if(tmp.is_open())
            {
                points_written += writePly(model, tmp);
            }
            else
            {
                std::cout << "could not open " << tmp_file << std::endl;
            }
            
            if(true == lastScan)
            {
                std::ofstream out;

                // write the header -> open in text_mode 
                out.open(options->getOutputFile().c_str(), std::ofstream::out | std::ofstream::trunc);
                // check if we have color information
                size_t n_colors;
                ucharArr colors = model->m_pointCloud->getPointColorArray(n_colors);
                if(n_colors)
                {
                    writePlyHeader(out, points_written, true);
                }
                else
                {
                    writePlyHeader(out, points_written, false);
                }

                out.close();

                // determine size of the complete binary blob
                tmp.seekg(0, std::fstream::end);
                size_t blob_size = tmp.tellg();
                tmp.seekg(0, std::fstream::beg);

                // open the actual output file for binary blob write
                out.open(options->getOutputFile(), std::ofstream::out | std::ofstream::app | std::ofstream::binary);
                
                char buffer[BUF_SIZE];
                
                while(blob_size)
                {
                    if(blob_size < BUF_SIZE)
                    { 
                        // read the rest from tmp file (binary blob)
                        tmp.read(buffer, blob_size);
                        // write the rest to actual ply file
                        out.write(buffer, blob_size);

                        blob_size -= blob_size;
                    }
                    else
                    {
                        // reading from tmp file (binary blob)
                        tmp.read(buffer, BUF_SIZE);
                        // write to actual ply file
                        out.write(buffer, BUF_SIZE);

                        blob_size -= BUF_SIZE;
                    }
                }

                out.close();
                tmp.close();

                std::remove(tmp_file);

                std::cout << timestamp << "Wrote " << points_written << " points." << std::endl;
            }
            else
            {
                tmp.close();
            }

        }
    }
    else
    {
        if(options->getOutputFormat() == "")
        {
            // Infer format from file extension, convert and write out
            char name[1024];
            char frames[1024];
            char pose[1024];
            char framesOut[1024];
            char poseOut[1024];

            model = ModelFactory::readModel(inFile.string());
            sprintf(frames, "%s/%s.frames", inFile.parent_path().c_str(), inFile.stem().c_str());
            sprintf(pose, "%s/%s.pose", inFile.parent_path().c_str(), inFile.stem().c_str());
            sprintf(framesOut, "%s/%s.frames", options->getOutputDir().c_str(), inFile.stem().c_str());
            sprintf(poseOut, "%s/%s.pose", options->getOutputDir().c_str(), inFile.stem().c_str());
            sprintf(name, "%s/%s", options->getOutputDir().c_str(), inFile.filename().c_str());

            boost::filesystem::path framesPath(frames);
            boost::filesystem::path posePath(pose);

            // Transform the frames
            if(boost::filesystem::exists(framesPath))
            {
                std::cout << timestamp << "Transforming frame: " << framesPath << std::endl;
                Eigen::Matrix4d transformed = transformFrames(getTransformationFromFrames(framesPath));
                writeFrames(transformed, framesOut);
            }

            ofstream out(name);
            transformFromOptions(model, asciiReductionFactor(inFile));
            size_t points_written = writeAscii(model, out);

            out.close();

            cout << "Wrote " << points_written << " points to file " << name << endl;
         }
        else if(options->getOutputFormat() == "SLAM")
        {
            std::cerr << "I am sorry! This is not implemented yet" << std::endl;
        }
        else
        {
            std::cerr << "I am sorry! This is not implemented yet" << std::endl;
        }
    }
}

    template <typename Iterator>
bool parse_filename(Iterator first, Iterator last, int& i)
{

    using qi::lit;
    using qi::uint_parser;
    using qi::parse;
    using boost::spirit::qi::_1;
    using boost::phoenix::ref;

    uint_parser<unsigned, 10, 3, 3> uint_3_d;

    bool r = parse(
            first,                          /*< start iterator >*/
            last,                           /*< end iterator >*/
            ((lit("scan")|lit("Scan")) >> uint_3_d[ref(i) = _1])   /*< the parser >*/
            );

    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

bool sortScans(boost::filesystem::path firstScan, boost::filesystem::path secScan)
{
    std::string firstStem = firstScan.stem().string();
    std::string secStem   = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_filename(firstStem.begin(), firstStem.end(), i);
    bool sec = parse_filename(secStem.begin(), secStem.end(), j);

    if(first && sec)
    {
        return (i < j);
    }
    else
    {
        // this causes non valid files being at the beginning of the vector.
        if(sec)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

int main(int argc, char** argv) {
    // Parse command line arguments
    options = new kaboom::Options(argc, argv);

    boost::filesystem::path inputDir(options->getInputDir());
    boost::filesystem::path outputDir(options->getOutputDir());

    // Check input directory
    if(!boost::filesystem::exists(inputDir))
    {
        cout << timestamp << "Error: Directory " << options->getInputDir() << " does not exist" << endl;
        exit(-1);
    }

    // Check if output dir exists
    if(!boost::filesystem::exists(outputDir))
    {
        cout << timestamp << "Creating directory " << options->getOutputDir() << endl;
        if(!boost::filesystem::create_directory(outputDir))
        {
            cout << timestamp << "Error: Unable to create " << options->getOutputDir() << endl;
            exit(-1);
        }
    }

    boost::filesystem::path abs_in = boost::filesystem::canonical(inputDir);
    boost::filesystem::path abs_out = boost::filesystem::canonical(outputDir);

    if(abs_in == abs_out)
    {
        cout << timestamp << "Error: We think it is not a good idea to write into the same directory. " << endl;
        exit(-1);
    }

    // Create director iterator and parse supported file formats
    boost::filesystem::directory_iterator end;
    vector<boost::filesystem::path> v;
    for(boost::filesystem::directory_iterator it(inputDir); it != end; ++it)
    {
        std::string ext =    it->path().extension().string();
        if(ext == ".3d" || ext == ".ply" || ext == ".dat" || ext == ".txt" )
        {
            v.push_back(it->path());
        }
    }

    // Sort entries
    sort(v.begin(), v.end(), sortScans);

    vector<float>             merge_points;
    vector<unsigned char>    merge_colors;

    int j = -1;
    for(vector<boost::filesystem::path>::iterator it = v.begin(); it != v.end(); ++it)
    {
        int i = 0;

        std::string currFile = (it->stem()).string();
        bool p = parse_filename(currFile.begin(), currFile.end(), i);

        //if parsing failed terminate, this should never happen.
        if(!p)
        {
            std::cerr << timestamp << "ERROR " << " " << *it << " does not match the naming convention" << std::endl;
            break;
        }

        // check if the current scan has the same numbering like the previous, this should not happen.
        if(i == j)
        {
            std::cerr << timestamp << "ERROR " << *std::prev(it) << " & " << *it << " have identical numbering" << std::endl;
            break;
        }

        // check if the scan is in the range which should be processed
        if(i >= options->getStart()){
            // when end is default(=0) process the complete vector
            if(0 == options->getEnd() || i <= options->getEnd())
            {
                try
                {
                    // This is dirty and bad designed.
                    // We need to know when we advanced to the last scan
                    // for ply merging. Which originally was not planned.
                    // Two cases end option set or not.
                    if((i  == options->getEnd()) || std::next(it, 1) == v.end())
                    {
                        lastScan = true;
                    }
                    processSingleFile(*it);
                    std::cout << " finished" << std::endl;
                }
                catch(const char* msg)
                {
                    std::cerr << timestamp << msg << *it << std::endl;
                    break;
                }
                j = i;
            }
            else
            {
                break;
            }
        }

    }

    cout << timestamp << "Program end." << endl;
    delete options;
    return 0;
}
