
namespace lvr2
{

template<typename T>
Eigen::Matrix<T,4,4> getTransformationFromPose(const boost::filesystem::path& pose)
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
        return Eigen::Matrix<T,4,4>::Identity();
    }
}

template<typename T>
Eigen::Matrix<T,4,4> getTransformationFromFrames(const boost::filesystem::path& frames)
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
Eigen::Matrix<T,4,4> getTransformationFromDat(const boost::filesystem::path& frames)
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
    return Eigen::Map<Eigen::Matrix<T,4,4>>(alignxf).transpose();
}

template<typename T>
void writeFrame(const Eigen::Matrix<T, 4, 4>& transform, const boost::filesystem::path& framesOut)
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

} // namespace lvr2