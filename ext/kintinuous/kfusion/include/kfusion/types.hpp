#pragma once

#include <kfusion/cuda/device_array.hpp>
#include <kfusion/Options.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iosfwd>

struct CUevent_st;

namespace kfusion
{
    typedef cv::Matx33f Mat3f;
    typedef cv::Vec3f Vec3f;
    typedef cv::Vec3i Vec3i;
    typedef cv::Affine3f Affine3f;

    struct KF_EXPORTS Intr
    {
        float fx, fy, cx, cy;

        Intr ();
        Intr (float fx, float fy, float cx, float cy);
        Intr operator()(int level_index) const;
    };

    struct KF_EXPORTS ImgPose
    {
		cv::Mat image;
		Affine3f pose;
		cv::Mat intrinsics;
		cv::Mat distortion;
    };

    struct KF_EXPORTS TSDFSlice
    {
		cv::Mat tsdf_values_;
		Vec3i offset_;
		Vec3i back_offset_;
		std::vector<ImgPose*> imgposes_;
    };

    KF_EXPORTS std::ostream& operator << (std::ostream& os, const Intr& intr);

    struct Point
    {
        union
        {
            float data[4];
            struct { float x, y, z, w; };
        };

        Point& operator+(cv::Vec<float, 3> vec)
		{
			this->x += vec[0];
			this->y += vec[1];
			this->z += vec[2];
			return *this;
		}
    };

    typedef Point Normal;

    KF_EXPORTS std::ostream& operator << (std::ostream& os, const kfusion::Point& p);

    struct RGB
    {
        union
        {
            struct { unsigned char b, g, r; };
            int bgra;
        };
    };

    struct PixelRGB
    {
        unsigned char r, g, b;
    };

    namespace cuda
    {
        typedef cuda::DeviceMemory CudaData;
        typedef cuda::DeviceArray2D<unsigned short> Depth;
        typedef cuda::DeviceArray2D<unsigned short> Dists;
        typedef cuda::DeviceArray2D<RGB> Image;
        typedef cuda::DeviceArray2D<Normal> Normals;
        typedef cuda::DeviceArray2D<Point> Cloud;

        struct Frame
        {
            bool use_points;

            std::vector<Depth> depth_pyr;
            std::vector<Cloud> points_pyr;
            std::vector<Normals> normals_pyr;
        };
    }

    inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

    struct KF_EXPORTS ScopeTime
    {
        const char* name;
        double start;
        ScopeTime(const char *name);
        ~ScopeTime();
        double getTime();
    };

    struct KF_EXPORTS SampledScopeTime
    {
    public:
        enum { EACH = 33 };
        SampledScopeTime(double& time_ms);
        ~SampledScopeTime();
    private:
        double getTime();
        SampledScopeTime(const SampledScopeTime&);
        SampledScopeTime& operator=(const SampledScopeTime&);

        double& time_ms_;
        double start;
    };

    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters
        //Intr intr_rgb;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float shifting_distance;
        double distance_camera_target;

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

		Options* cmd_options; // cmd_options

    };

}
