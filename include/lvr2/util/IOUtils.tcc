#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

template<typename T>
Transform<T> getTransformationFromPose(const boost::filesystem::path& pose)
{
    std::ifstream poseIn(pose.c_str());
    if(poseIn.good())
    {
        T rPosTheta[3];
        T rPos[3];
        T alignxf[16];

        poseIn >> rPos[0] >> rPos[1] >> rPos[2];
        poseIn >> rPosTheta[0] >> rPosTheta[1] >> rPosTheta[2];

        rPosTheta[0] *= 0.0174533;
        rPosTheta[1] *= 0.0174533;
        rPosTheta[2] *= 0.0174533;

        T sx = sin(rPosTheta[0]);
        T cx = cos(rPosTheta[0]);
        T sy = sin(rPosTheta[1]);
        T cy = cos(rPosTheta[1]);
        T sz = sin(rPosTheta[2]);
        T cz = cos(rPosTheta[2]);

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
        return Transform<T>::Identity();
    }
}

template<typename T>
Transform<T> getTransformationFromFrames(const boost::filesystem::path& frames)
{
    T alignxf[16];
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

template<typename T>
Transform<T> getTransformationFromDat(const boost::filesystem::path& frames)
{
    T alignxf[16];
    int color;

    std::ifstream in(frames.c_str());
    if(in.good())
    {
        for(int i = 0; i < 16; i++)
        {
            in >> alignxf[i];
        }
    }
    return Eigen::Map<Transform<T>>(alignxf).transpose();
}

template<typename T>
void writeFrame(const Transform<T>& transform, const boost::filesystem::path& framesOut)
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

template<typename T>
void writePose(const BaseVector<T>& position, const BaseVector<T>& angles, const boost::filesystem::path& out)
{
    std::ofstream o(out.c_str());
    if(o.good())
    {
        o << position[0] << " " << position[1] << " " << position[2] << std::endl;
        o << angles[0] << " " << angles[1] << " " << angles[2];
    }
}

template<typename T>
void writePose(const Vector3<T>& position, const Vector3<T>& angles, const boost::filesystem::path& out)
{
    std::ofstream o(out.c_str());
    if(o.good())
    {
        o << position(0) << " " << position(1) << " " << position(2) << std::endl;
        o << angles(0) << " " << angles(1) << " " << angles(2);
    }
}

template<typename T>
void writePose(const Transform<T>& transform, const boost::filesystem::path& poseOut)
{
    // block<1,3> was not working for the riegl projects, Alex said it worked before though
    Vector3<T> position = transform.template block<3, 1>(0,3);
    Vector3<T> angles = transform.template block<3,3>(0,0).eulerAngles(0, 1, 2) / 0.0174533;

    std::cout << "Writing: " << std::endl;
    std::cout << "- position: " << position.transpose() << std::endl;
    std::cout << "- rotation: " << angles.transpose() << std::endl;

    // to degree
    writePose(position, angles, poseOut);
}

template<typename T>
Transform<T> loadFromFile(const boost::filesystem::path& file)
{
    Transform<T> m;
    T arr[16];
    std::ifstream in(file.string());
    for(size_t i = 0; i < 16; i++)
    {
        in >> arr[i];
    }
    return Eigen::Map<Matrix4RM<T>>(arr);
}

template<typename T>
Transform<T> getTransformationFromFile(const boost::filesystem::path& file)
{
    boost::filesystem::path extension = file.extension();
    if(extension == ".dat")
    {
        return getTransformationFromDat<T>(file);
    }
    else if(extension == ".pose")
    {
        return getTransformationFromPose<T>(file);
    }
    else if(extension == ".frames")
    {
        return getTransformationFromFrames<T>(file);
    }
    else if(extension == ".matrix")
    {
        return loadFromFile<T>(file);
    }
    
    throw std::invalid_argument(string("Unknown Pose extension: ") + extension.string());
}

} // namespace lvr2