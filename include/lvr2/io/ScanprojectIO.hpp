#pragma once

#include <lvr2/io/BaseIO.hpp>
#include <lvr2/geometry/Matrix4.hpp>
#include <lvr2/geometry/BaseVector.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

namespace lvr2
{

using Vec = BaseVector<float>;

struct ImageFile {
    Matrix4<Vec>     orientation_transform;
    Matrix4<Vec>     extrinsic_transform;

    fs::path     image_file;

    // fx, fy, Cx, Cy
    float intrinsic_params[4];

    // params in this order: k1, k2, k3, k4, p1, p2
    float distortion_params[6];
};

struct ScanPosition {
    fs::path               scan_file;
    Matrix4<Vec>           transform;
    std::vector<ImageFile> images;
};

struct Scanproject {
    fs::path calib_dir;
    fs::path img_dir;
    fs::path scans_dir;
    std::vector<ScanPosition> scans;
};



class ScanprojectIO : public BaseIO {

public:
    ModelPtr read(std::string dir);
    void save(std::string dir);

    bool parse_project(const std::string &dir);

    Scanproject &get_project();

    // TODO maybe find a better place for these static methods
    template <typename BaseVecT>
    static Matrix4<BaseVecT> riegl_to_slam6d_transform(const Matrix4<BaseVecT> &in) {
        Matrix4<BaseVecT> ret;

        ret[0] = in[5];
        ret[1] = -in[9];
        ret[2] = -in[1];
        ret[3] = -in[13];
        ret[4] = -in[6];
        ret[5] = in[10];
        ret[6] = in[2];
        ret[7] = in[14];
        ret[8] = -in[4];
        ret[9] = in[8];
        ret[10] = in[0];
        ret[11] = in[12];
        ret[12] = -100*in[7];
        ret[13] = 100*in[11];
        ret[14] = 100*in[3];
        ret[15] = in[15];

        return ret;
    }

    template <typename BaseVecT>
    static Matrix4<BaseVecT> slam6d_to_riegl_transform(const Matrix4<BaseVecT> &in) {
        Matrix4<BaseVecT> ret;

        ret[0] = in[10];
        ret[1] = -in[2];
        ret[2] = in[6];
        ret[3] = in[14]/100.0;
        ret[4] = -in[8];
        ret[5] = in[0];
        ret[6] = -in[4];
        ret[7] = -in[12]/100.0;
        ret[8] = in[9];
        ret[9] = -in[1];
        ret[10] = in[5];
        ret[11] = in[13]/100.0;
        ret[12] = in[11];
        ret[13] = -in[3];
        ret[14] = in[7];
        ret[15] = in[15];

        return ret;
    }

    template <typename ValueType>
    static ValueType deg_to_rad(ValueType deg) {
        return M_PI / 180.0 * deg;
    }

    template <typename ValueType>
    static ValueType rad_to_deg(ValueType rad) {
        return rad * 180 / M_PI;
    }

private:
    template<typename ValueType>
    bool load_params_from_file(ValueType *buf, const fs::path &src, unsigned int count);
    bool exists_and_is_dir(const fs::path &dir);
    fs::path project_dir;
    Scanproject project;
};

} // namespace lvr2
