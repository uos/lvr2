#pragma once

#include <kfusion/types.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include "kfusion/cyclical_buffer.h"
#include <vector>
#include <string>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

    class KF_EXPORTS KinFu
    {
    public:
        typedef cv::Ptr<KinFu> Ptr;

        enum RenderMode
        {
          Scene = 0,
          Normals = 2,
          SceneAndNormals = 3
        };

        KinFu(const KinFuParams& params);

        const KinFuParams& params() const;
        KinFuParams& params();

        void performLastScan();

        void performShift() { perform_shift_ = true;}

        bool hasShifted()
			{return has_shifted_;}

        void triggerCheckForShift()
    			{checkForShift_ = !checkForShift_;}

		void triggerRecord()
			{record_mode_ = !record_mode_;}

		 bool isLastScan()
			{return perform_last_scan_;};

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cuda::CyclicalBuffer& cyclical() const;
        cuda::CyclicalBuffer& cyclical();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        void reset(Affine3f initialPose = Affine3f::Identity());

        bool operator()(const cuda::Depth& dpeth, const cuda::Image& image = cuda::Image());

        void renderImage(cuda::Image& image, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);
        void renderImage(cuda::Image& image, const Affine3f& pose, Intr cameraIntrinsics, cv::Size size, int flags = 0);

        Affine3f getCameraPose (int time = -1) const;
        std::vector<Affine3f>& getCameraPoses() {return poses_;}
    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

		bool has_shifted_;
		bool perform_last_scan_;
		bool perform_shift_;
		bool record_mode_;
        bool checkForShift_;
        cv::Ptr<cuda::TsdfVolume> volume_;
        /** \brief Cyclical buffer object */
        cuda::CyclicalBuffer cyclical_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
    };
}
