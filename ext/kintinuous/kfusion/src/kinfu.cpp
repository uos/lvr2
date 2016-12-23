#include <kfusion/kinfu.hpp>
#include <kfusion/precomp.hpp>
#include <kfusion/internal.hpp>

using namespace std;
using namespace kfusion;
using namespace kfusion::cuda;

kfusion::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params), has_shifted_(false), perform_last_scan_(false), perform_shift_(false)
                                                   , cyclical_(params), checkForShift_(true)

{
    CV_Assert(params.volume_dims[0] % 32 == 0);

    volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));

    volume_->setTruncDist(params_.tsdf_trunc_dist);
    volume_->setMaxWeight(params_.tsdf_max_weight);
    volume_->setSize(params_.volume_size);
    volume_->setPose(params_.volume_pose);
    volume_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

    // initialize cyclical buffer
    cyclical_.initBuffer(volume_);

    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);
    allocate_buffers();
    reset();
}

const kfusion::KinFuParams& kfusion::KinFu::params() const
{ return params_; }

kfusion::KinFuParams& kfusion::KinFu::params()
{ return params_; }

void kfusion::KinFu::performLastScan()
{
	perform_last_scan_ = true;
}

const kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf() const
{ return *volume_; }

kfusion::cuda::TsdfVolume& kfusion::KinFu::tsdf()
{ return *volume_; }

const kfusion::cuda::CyclicalBuffer& kfusion::KinFu::cyclical() const
{ return cyclical_; }

kfusion::cuda::CyclicalBuffer& kfusion::KinFu::cyclical()
{ return cyclical_; }

const kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp() const
{ return *icp_; }

kfusion::cuda::ProjectiveICP& kfusion::KinFu::icp()
{ return *icp_; }

void kfusion::KinFu::allocate_buffers()
{
    const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void kfusion::KinFu::reset(Affine3f initialPose)
{
    if (frame_counter_)
        cout << "Reset" << endl;
    //initialPose.translate(Vec3f(0, 0, params_.distance_camera_target));
    frame_counter_ = 0;
    poses_.clear();
    //poses_.reserve(60000);
    poses_.push_back(initialPose);
    volume_->clear();
    volume_->setPose(params_.volume_pose);
    cyclical_.resetBuffer (volume_);
    cyclical_.resetMesh();
    has_shifted_=false;
}

kfusion::Affine3f kfusion::KinFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}

bool kfusion::KinFu::operator()(const kfusion::cuda::Depth& depth, const kfusion::cuda::Image& /*image*/)
{
    const KinFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();

    cuda::computeDists(depth, dists_, p.intr);
    cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0)
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    for (int i = 1; i < LEVELS; ++i)
        cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);

    for (int i = 0; i < LEVELS; ++i)
#if defined USE_DEPTH
        cuda::computeNormalsAndMaskDepth(p.intr, curr_.depth_pyr[i], curr_.normals_pyr[i]);
#else
        cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
#endif

    cuda::waitAllDefaultStream();

    //can't perform more on first frame
    if (frame_counter_ == 0)
    {
        volume_->integrate(dists_, cyclical_.getBuffer(), poses_.back(), p.intr);
#if defined USE_DEPTH
        curr_.depth_pyr.swap(prev_.depth_pyr);
#else
        curr_.points_pyr.swap(prev_.points_pyr);
#endif
        curr_.normals_pyr.swap(prev_.normals_pyr);
        return ++frame_counter_, false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // ICP
    Affine3f affine; // cuur -> prev
    {
        //ScopeTime time("icp");
#if defined USE_DEPTH
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
#else
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);
#endif
        if (!ok)
            return reset(), false;
    }
    poses_.push_back(poses_.back() * affine); // curr -> global

    // check if we need to shift
    if(checkForShift_)
        has_shifted_ = cyclical_.checkForShift(volume_, getCameraPose(), params_.distance_camera_target , perform_shift_, perform_last_scan_, record_mode_);

    perform_shift_ = false;
    ///////////////////////////////////////////////////////////////////////////////////////////
    // Volume integration
    Affine3f local_pose = Affine3f().translate(poses_.back().translation() - volume_->getPose().translation());
			 local_pose.rotate(poses_.back().rotation());
    // We do not integrate volume if camera does not move.
    float rnorm = (float)cv::norm(affine.rvec());
    float tnorm = (float)cv::norm(affine.translation());
    bool integrate = (rnorm + tnorm)/2 >= p.tsdf_min_camera_movement;
    if (integrate)
    {
        //ScopeTime time("tsdf");
        volume_->integrate(dists_, cyclical_.getBuffer(), poses_.back(), p.intr);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting
    {
        //ScopeTime time("ray-cast-all");
#if defined USE_DEPTH
        volume_->raycast(poses_.back(), cyclical_.getBuffer(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizeDepthNormals(prev_.depth_pyr[i-1], prev_.normals_pyr[i-1], prev_.depth_pyr[i], prev_.normals_pyr[i]);
#else
        //volume_->raycast(local_pose, p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        volume_->raycast(poses_.back(), cyclical_.getBuffer(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
#endif
        cuda::waitAllDefaultStream();
    }

    return ++frame_counter_, true;
}

void kfusion::KinFu::renderImage(cuda::Image& image, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

#if defined USE_DEPTH
    #define PASS1 prev_.depth_pyr
#else
    #define PASS1 prev_.points_pyr
#endif

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(prev_.normals_pyr[0], image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(prev_.normals_pyr[0], i2);
    }
#undef PASS1
}


void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    depths_.create(p.rows, p.cols);
    normals_.create(p.rows, p.cols);
    points_.create(p.rows, p.cols);

#if defined USE_DEPTH
    #define PASS1 depths_
#else
    #define PASS1 points_
#endif

    volume_->raycast(pose, cyclical_.getBuffer(), p.intr, PASS1, normals_);

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
        cuda::renderTangentColors(normals_, i2);
    }
#undef PASS1
}


void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, Intr cameraIntrinsics, cv::Size size, int flag)
{
    int rows = size.height;
    int cols = size.width;
    image.create(rows, flag != 3 ? cols : cols * 2);
    depths_.create(rows, cols);
    normals_.create(rows, cols);
    points_.create(rows, cols);

#if defined USE_DEPTH
    #define PASS1 depths_
#else
    #define PASS1 points_
#endif

    volume_->raycast(pose, cyclical_.getBuffer(), cameraIntrinsics, PASS1, normals_);

    if (flag < 1 || flag > 3)
        cuda::renderImage(PASS1, normals_, cameraIntrinsics, params_.light_pose, image);
    else if (flag == 2)
        cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        DeviceArray2D<RGB> i1(rows, cols, image.ptr(), image.step());
        DeviceArray2D<RGB> i2(rows, cols, image.ptr() + cols, image.step());

        cuda::renderImage(PASS1, normals_, cameraIntrinsics, params_.light_pose, i1);
        cuda::renderTangentColors(normals_, i2);
    }
#undef PASS1
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
