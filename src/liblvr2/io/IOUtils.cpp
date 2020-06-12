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

#include "lvr2/io/DataStruct.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/AsciiIO.hpp"
#include "lvr2/registration/TransformUtils.hpp"

#include <random>
#include <unordered_set>

namespace lvr2
{

void transformPointCloudAndAppend(PointBufferPtr& buffer,
        boost::filesystem::path& transfromFile,
        std::vector<float>& pts,
        std::vector<float>& nrm)
{
     std::cout << timestamp << "Transforming normals " << std::endl;

     char frames[2048];
     char pose[2014];

     sprintf(frames, "%s/%s.frames", transfromFile.parent_path().c_str(),
             transfromFile.stem().c_str());
     sprintf(pose, "%s/%s.pose", transfromFile.parent_path().c_str(), transfromFile.stem().c_str());

     boost::filesystem::path framesPath(frames);
     boost::filesystem::path posePath(pose);


     Transformd transform = Transformd::Identity();

     if(boost::filesystem::exists(framesPath))
     {
        std::cout << timestamp << "Transforming according to " << framesPath.filename() << std::endl;
        transform = getTransformationFromFrames<double>(framesPath);
     }
     else if(boost::filesystem::exists(posePath))
     {
        std::cout << timestamp << "Transforming according to " << posePath.filename() << std::endl;
        transform = getTransformationFromFrames<double>(posePath);
     }
     else
     {
        std::cout << timestamp << "Warning: found no transformation for "
            << transfromFile.filename() << std::endl;
     }

     size_t n_normals;
     size_t w_normals;
     size_t n_points = buffer->numPoints();

     floatArr normals = buffer->getFloatArray("normals", n_normals, w_normals); 
     floatArr points = buffer->getPointArray();

     if (w_normals != 3)
     {
        std::cout << timestamp << "Warning: width of normals is not 3" << std::endl;
        return;
     }
     if(n_normals != n_points)
     {
         std::cout << timestamp << "Warning: point and normal count mismatch" << std::endl;
         return;
     }

     for(size_t i = 0; i < n_points; i++)
     {

        float x = points[3 * i];
        float y = points[3 * i + 1];
        float z = points[3 * i + 2];

        Vector4d v(x,y,z,1);
        Vector4d tv = transform * v;

//        points[3 * i]     = tv[0];
//        points[3 * i + 1] = tv[1];
//        points[3 * i + 2] = tv[2];

        pts.push_back(tv[0]);
        pts.push_back(tv[1]);
        pts.push_back(tv[2]);

        Rotationd rotation = transform.block(0, 0, 3, 3);

        float nx = normals[3 * i];
        float ny = normals[3 * i + 1];
        float nz = normals[3 * i + 2];

        Vector3d normal(nx, ny, nz);
        Vector3d tn = rotation * normal;

//        normals[3 * i]     = tn[0];
//        normals[3 * i + 1] = tn[1];
//        normals[3 * i + 2] = tn[2];

        nrm.push_back(tn[0]);
        nrm.push_back(tn[1]);
        nrm.push_back(tn[2]);
     }

}

size_t countPointsInFile(const boost::filesystem::path& inFile)
{
    std::ifstream in(inFile.c_str());
    std::cout << timestamp << "Counting points in "
        << inFile.filename().string() << "..." << std::endl;

    // Count lines in file
    size_t n_points = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        n_points++;
    }
    in.close();

    std::cout << timestamp << "File " << inFile.filename().string()
        << " contains " << n_points << " points." << std::endl;

    return n_points;
}

void writePose(const BaseVector<float>& position, const BaseVector<float>& angles, const boost::filesystem::path& out)
{
    std::ofstream o(out.c_str());
    if(o.good())
    {
        o << position[0] << " " << position[1] << " " << position[2] << std::endl;
        o << angles[0] << " " << angles[1] << " " << angles[2];
    }
}

size_t writeModel(ModelPtr model, const boost::filesystem::path& outfile)
{
    size_t n_ip = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();

    ModelFactory::saveModel(model, outfile.string());

    return n_ip;
}

size_t writePointsToStream(ModelPtr model, std::ofstream& out, bool nocolor)
{
    size_t n_ip, n_colors;
    size_t w_colors;

    n_ip = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();

    ucharArr colors = model->m_pointCloud->getUCharArray("colors", n_colors, w_colors);

    for(int a = 0; a < n_ip; a++)
    {
        out << arr[a * 3] << " " << arr[a * 3 + 1] << " " << arr[a * 3 + 2];

        if(n_colors && !(nocolor))
        {
            for (unsigned i = 0; i < w_colors; i++)  
            {
                out << " " << (int)colors[a * w_colors + i];
            }
        }
        out << std::endl;

    }

    return n_ip;
}

size_t getReductionFactor(ModelPtr model, size_t reduction)
{
    size_t n_points = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();


    std::cout << timestamp << "Point cloud contains " << n_points << " points." << std::endl;

/*
     * If reduction is less than the number of points it will segfault
     * because the modulo operation is not defined for n mod 0
     * and we have to keep all points anyways.
     * Same if no targetSize was given.
     */
    if(reduction != 0)
    {
        if(reduction < n_points)
        {
            return (int)n_points / reduction;
        }
    }

    /* No reduction write all points */
    return 1;
}

size_t getReductionFactor(boost::filesystem::path& inFile, size_t targetSize)
{
    /*
     * If reduction is less than the number of points it will segfault
     * because the modulo operation is not defined for n mod 0
     * and we have to keep all points anyways.
     * Same if no targetSize was given.
     */
    if(targetSize != 0)
    {
        // Count lines in file
        size_t n_points = countPointsInFile(inFile);

        if(targetSize < n_points)
        {
            return (int)n_points / targetSize;
        }
    }

    /* No reduction write all points */
    return 1;

}

void writePointsAndNormals(std::vector<float>& p, std::vector<float>& n, std::string outfile)
{

    ModelPtr model(new Model);
    PointBufferPtr buffer(new PointBuffer);

    // Passing the raw data pointers from the vectors
    // to a shared array is a bad idea. Due to the PointBuffer
    // interface we have to copy the data :-(
    //    floatArr points(p.data());
    //    floatArr normals(n.data());

    floatArr points(new float[p.size()]);
    floatArr normals(new float[n.size()]);

    std::cout << timestamp << "Copying buffers for output." << std::endl;
    // Assuming p and n have the same size (which they should)
    for(size_t i = 0; i < p.size(); i++)
    {
        points[i] = p[i];
        normals[i] = n[i];
    }

    buffer->setPointArray(points, p.size() / 3);
    buffer->setNormalArray(normals, n.size() / 3);

    model->m_pointCloud = buffer;

    std::cout << timestamp << "Saving " << outfile << std::endl;
    ModelFactory::saveModel(model, outfile);
    std::cout << timestamp << "Done." << std::endl;
}

void getPoseFromFile(BaseVector<float>& position, BaseVector<float>& angles, const boost::filesystem::path file)
{
    ifstream in(file.c_str());
    if(in.good())
    {
        in >> position.x >> position.y >> position.z;
        in >> angles.y >> angles.y >> angles.z;
    }
}

size_t getNumberOfPointsInPLY(const std::string& filename)
{
    size_t n_points = 0;
    size_t n_vertices = 0;

    // Try to open file
    std::ifstream in(filename.c_str());
    if(in.good())
    {
        // Check for ply tag
        std::string tag;
        in >> tag;
        if(tag == "PLY" || tag == "ply")
        {
            // Parse header
            std::string token;
            while (in.good() && token != "end_header" && token != "END_HEADER")
            {
                in >> token;
              

                // Check for vertex field
                if(token == "vertex" || token == "VERTEX")
                {
                    in >> n_vertices;
                }

                // Check for point field
                if(token == "point" || token == "POINT")
                {
                    in >> n_points;
                }
            }
            if(n_points == 0 && n_vertices == 0)
            {
                std::cout << timestamp << "PLY contains neither vertices nor points." << std::endl;
                return 0;
            }
            
            // Prefer points over vertices
            if(n_points)
            {
                return n_points;
            }
            else
            {
                return n_vertices;
            }
        }
        else
        {
            std::cout << timestamp << filename << " is not a valid .ply file." << std::endl;
        }
        
    }
    return 0;
}

template<typename T>
typename Channel<T>::Ptr subSampleChannel(Channel<T>& src, std::vector<size_t> ids)
{
    // Create smaller channel of same type
    size_t width = src.width();
    typename Channel<T>::Ptr red(new Channel<T>(ids.size(), width));

    // Sample from original and insert into reduced 
    // channel
    boost::shared_array<T> a(red->dataPtr());
    boost::shared_array<T> b(src.dataPtr());
    for(size_t i = 0; i < ids.size(); i++)
    {
        for(size_t j = 0; j < red->width(); j++)
        {
            a[i * width + j] = b[ids[i] * width + j];
        }
    }
    return red;
}

template<typename T>
void subsample(PointBufferPtr src, PointBufferPtr dst, const vector<size_t>& indices)
{
    // Go over all supported channel types and sub-sample
    std::map<std::string, Channel<T>> channels;
    src->getAllChannelsOfType(channels);      
    for(auto i : channels)
    {
        std::cout << timestamp << "Subsampling channel " << i.first << std::endl;
        typename Channel<T>::Ptr c = subSampleChannel(i.second, indices);
        dst->addChannel<T>(c, i.first);
    }

}

PointBufferPtr subSamplePointBuffer(PointBufferPtr src, const std::vector<size_t>& indices)
{
    PointBufferPtr buffer(new PointBuffer);

    // Go over all supported channel types and sub-sample
    subsample<char>(src, buffer, indices);
    subsample<unsigned char>(src, buffer, indices);
    subsample<short>(src, buffer, indices);
    subsample<int>(src, buffer, indices);
    subsample<unsigned int>(src, buffer, indices);
    subsample<float>(src, buffer, indices);
    subsample<double>(src, buffer, indices);

    return buffer;
}

PointBufferPtr subSamplePointBuffer(PointBufferPtr src, const size_t& n)
{
    // Buffer for reduced points
    PointBufferPtr buffer(new PointBuffer);
    size_t numSrcPts = src->numPoints();
    
    // Setup random device and distribution
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, numSrcPts);

    // Check buffer size
    if(n <= numSrcPts)
    {
        // Create index buffer for sub-sampling, using set to avoid duplicates
        std::unordered_set<size_t> index_set;
        while(index_set.size() < n)
        {   
            index_set.insert(dist(rng));
        }

        // Copy indices into vector for faster access and []-operator support
        //.In c++14 this is the fastest way. In C++17 a better alternative 
        // would be to use extract().
        vector<size_t> indices;
        indices.insert(indices.end(), index_set.begin(), index_set.end());
        index_set.clear();

        // Go over all supported channel types and sub-sample
        subsample<char>(src, buffer, indices);
        subsample<unsigned char>(src, buffer, indices);
        subsample<short>(src, buffer, indices);
        subsample<int>(src, buffer, indices);
        subsample<unsigned int>(src, buffer, indices);
        subsample<float>(src, buffer, indices);
        subsample<double>(src, buffer, indices);
    }
    else
    {
        std::cout << timestamp << "Sub-sampling not possible. Number of sampling points is " << std::endl;
        std::cout << timestamp << "larger than number in src buffer. (" << n << " / " << numSrcPts << ")" << std::endl;
    }
    

    return buffer;
}

void slamToLVRInPlace(PointBufferPtr src)
{
    // Get point channel
    typename Channel<float>::Optional opt = src->getChannel<float>("points");
    if(opt)
    {
        size_t n = opt->numElements();
        floatArr points = opt->dataPtr();

        #pragma omp parallel for
        for(size_t i = 0; i < n; i++)
        {
            float x = points[3 * i];
            float y = points[3 * i + 1];
            float z = points[3 * i + 2];

            points[3 * i]       = z / 100.0f;
            points[3 * i + 1]   = -x / 100.0f;
            points[3 * i + 2]   = y / 100.0f;

        }
    }
}

void parseSLAMDirectory(std::string dir, vector<ScanPtr>& scans)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {

        boost::filesystem::directory_iterator lastFile;
        std::vector<boost::filesystem::path> scan_files;

        // First, look for .3d files
        for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
        {
            boost::filesystem::path p = it->path();
            if(p.extension().string() == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
                {
                    scan_files.push_back(p);
                }
            }
        }

        if(scan_files.size() > 0)
        {
            for(size_t i = 0; i < scan_files.size(); i++)
            {
                ScanPtr scan = ScanPtr(new Scan());

                std::string filename = (scan_files[i]).stem().string();
                boost::filesystem::path frame_file(filename + ".frames");
                boost::filesystem::path pose_file(filename + ".pose");

                boost::filesystem::path frame_path = directory/frame_file;
                boost::filesystem::path pose_path = directory/pose_file;

                std::cout << "Loading '" << filename << "'" << std::endl;
                AsciiIO io;
                ModelPtr model = io.read(scan_files[i].string());
                scan->points = model->m_pointCloud;

                size_t numPoints = scan->points->numPoints();
                floatArr pts = scan->points->getPointArray();

                for (size_t i = 0; i < numPoints; i++)
                {
                    BaseVector<float> pt(pts[i*3 + 0], pts[i*3 + 1], pts[i*3 + 2]);
                    scan->boundingBox.expand(pt);
                }

                Transformd pose_estimate = Transformd::Identity();
                Transformd registration = Transformd::Identity();

                if(boost::filesystem::exists(frame_path))
                {
                    std::cout << timestamp << "Loading frame information from " << frame_path << std::endl;
                    registration = getTransformationFromFrames<double>(frame_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a frame file for " << filename << std::endl;
                }

                if(boost::filesystem::exists(pose_path))
                {
                    std::cout << timestamp << "Loading pose estimation from " << pose_path << std::endl;
                    pose_estimate = getTransformationFromPose<double>(pose_path);
                }
                else
                {
                    std::cout << timestamp << "Did not find a pose file for " << filename << std::endl;
                }

                // transform points?
                scan->registration = registration;
                scan->poseEstimation = pose_estimate;

                scans.push_back(scan);
            }
        }
        else
        {
            std::cout << timestamp << "Error in parseSLAMDirectory(): '"
                      << "Directory does not contain any .3d files." << std::endl;
        }
    }
    else
    {
        std::cout << timestamp << "Error in parseSLAMDirectory(): '"
                  << dir << "' is nor a directory." << std::endl;
    }
}

} // namespace lvr2
