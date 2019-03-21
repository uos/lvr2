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

#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/ModelFactory.hpp"
namespace lvr2
{


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

void getPoseFromMatrix(BaseVector<float>& position, BaseVector<float>& angles, const Eigen::Matrix4d& mat)
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
    std::ifstream poseIn(pose.c_str());
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

Eigen::Matrix4d getTransformationFromDat(boost::filesystem::path& frames)
{
    double alignxf[16];
    int color;

    std::ifstream in(frames.c_str());
    if(in.good())
    {
        for(int i = 0; i < 16; i++)
        {
            in >> alignxf[i];
        }
    }

    return buildTransformation(alignxf);
}

size_t countPointsInFile(boost::filesystem::path& inFile)
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

void writeFrame(Eigen::Matrix4d transform, const boost::filesystem::path& framesOut)
{
    std::ofstream out(framesOut.c_str());

    // write the rotation matrix
    out << transform.col(0)(0) << " " << transform.col(0)(1) << " " << transform.col(0)(2)
        << " " << 0 << " "
        << transform.col(1)(0) << " " << transform.col(1)(1) << " " << transform.col(1)(2)
        << " " << 0 << " "
        << transform.col(2)(0) << " " << transform.col(2)(1) << " " << transform.col(2)(2)
        << " " << 0 << " ";

    // write the translation vector
    out << transform.col(3)(0) << " "
        << transform.col(3)(1) << " "
        << transform.col(3)(2) << " "
        << transform.col(3)(3);

    out.close();
}

size_t writeModel( ModelPtr model,const  boost::filesystem::path& outfile)
{
    size_t n_ip = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();

    ModelFactory::saveModel(model, outfile.string());

    return n_ip;
}

size_t writePointsToStream(ModelPtr model, std::ofstream& out, bool nocolor)
{
    size_t n_ip, n_colors;
    unsigned w_colors;

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


void transformPointCloud(ModelPtr model, Eigen::Matrix4d transformation)
{
    std::cout << timestamp << "Transforming model." << std::endl;

    size_t numPoints = model->m_pointCloud->numPoints();
    floatArr arr = model->m_pointCloud->getPointArray();

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


     Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();

     if(boost::filesystem::exists(framesPath))
     {
        std::cout << timestamp << "Transforming according to " << framesPath.filename() << std::endl;
        transform = getTransformationFromFrames(framesPath);
     }
     else if(boost::filesystem::exists(posePath))
     {
        std::cout << timestamp << "Transforming according to " << posePath.filename() << std::endl;
        transform = getTransformationFromFrames(posePath);
     }
     else
     {
        std::cout << timestamp << "Warning: found no transformation for "
            << transfromFile.filename() << std::endl;
     }

     size_t n_normals;
     unsigned w_normals;
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

        Eigen::Vector4d v(x,y,z,1);
        Eigen::Vector4d tv = transform * v;

//        points[3 * i]     = tv[0];
//        points[3 * i + 1] = tv[1];
//        points[3 * i + 2] = tv[2];

        pts.push_back(tv[0]);
        pts.push_back(tv[1]);
        pts.push_back(tv[2]);

        Eigen::Matrix3d rotation = transform.block(0, 0, 3, 3);

        float nx = normals[3 * i];
        float ny = normals[3 * i + 1];
        float nz = normals[3 * i + 2];

        Eigen::Vector3d normal(nx, ny, nz);
        Eigen::Vector3d tn = rotation * normal;

//        normals[3 * i]     = tn[0];
//        normals[3 * i + 1] = tn[1];
//        normals[3 * i + 2] = tn[2];

        nrm.push_back(tn[0]);
        nrm.push_back(tn[1]);
        nrm.push_back(tn[2]);
     }

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

void getPoseFromFile(BaseVector<float>& position, BaseVector<float>& angles, const boost::filesystem::path file)
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

} // namespace lvr2
