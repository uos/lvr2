#include <iostream>

#include "lvr2/types/Scan.hpp"
#include "lvr2/io/HDF5IO.hpp"
#include "lvr2/io/ScanprojectIO.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/util/Util.hpp"

#include <opencv2/core.hpp>


using namespace lvr2;

ScanPtr toScanPtr(ScanPosition sp)
{
    ScanPtr scan_data = ScanPtr(new Scan());
    ModelPtr model = ModelFactory::readModel(sp.scan_file.string());
    scan_data->m_points = model->m_pointCloud;

    size_t numPoints = scan_data->m_points->numPoints();
    floatArr pts = scan_data->m_points->getPointArray();

    for (size_t i = 0; i < numPoints; i++)
    {
        BaseVector<float> pt(pts[i*3 + 0], pts[i*3 + 1], pts[i*3 + 2]);
        scan_data->m_boundingBox.expand(pt);
    }

    scan_data->m_registration = sp.transform;
    scan_data->m_poseEstimation = sp.transform;

    scan_data->m_registration.transpose();
    scan_data->m_poseEstimation.transpose();

    return scan_data;
}

void slamToLVR(ScanPtr& sd)
{
    // // registrations
    // sd.m_registration.transpose();
    // sd.m_registration = slam6DToRieglTansform(sd.m_registration);

    // // poseEstimation
    // sd.m_poseEstimation.transpose();
    // sd.m_poseEstimation = slam6DToRieglTansform(sd.m_poseEstimation);

    // // points
    // size_t num_points = sd.m_points->numPoints();
    // floatArr pts = sd.m_points->getPointArray();
    
    // BoundingBox<Vec> bb;

    // #pragma omp for
    // for(size_t j=0; j<num_points; j++)
    // {
    //     BaseVector<float>* mem_ptr = reinterpret_cast<BaseVector<float>*>(pts.get()+j*3);
    //     const BaseVector<float> p = Util::slam6d_to_riegl_point(*mem_ptr);
        
    //     bb.expand(p);
    //     mem_ptr->x = p.x;
    //     mem_ptr->y = p.y;
    //     mem_ptr->z = p.z;
    // }
    // sd.m_boundingBox = bb;

}

CameraData toCamData(ImageFile img_file)
{
    CameraData ret;
    ret.intrinsics = Transformd();
    ret.intrinsics(0) = img_file.intrinsic_params[0];
    ret.intrinsics(5) = img_file.intrinsic_params[1];
    ret.intrinsics(2) = img_file.intrinsic_params[2];
    ret.intrinsics(6) = img_file.intrinsic_params[3];

    ret.extrinsics = img_file.extrinsic_transform.inverse() * img_file.orientation_transform;

    ret.image = cv::imread(img_file.image_file.string(), CV_LOAD_IMAGE_COLOR);

    return ret;
}

void testRead(HDF5IO& hdf5)
{
    std::vector<std::vector<CameraData> > cam_data;
    cam_data = hdf5.getRawCamData();

    for(int scan_id=0; scan_id < cam_data.size(); scan_id++)
    {
        for(int cam_id=0; cam_id < cam_data[scan_id].size(); cam_id++)
        {
            CameraData cam = cam_data[scan_id][cam_id];
            std::cout << "scan " << scan_id << ", cam " << cam_id << std::endl;
            cv::imshow("test", cam.image);
            cv::waitKey(0);
        }
    }
}

int main(int argc, char** argv)
{
    std::cout << "Scanproject to HDF5 converter started." << std::endl;
    if(argc > 1) {
        ScanprojectIO project;
        project.parse_project(argv[1]);
        auto proj = project.get_project();

        HDF5IO hdf5("test.h5", true);

        std::cout << "Found " << proj.scans.size() << " scans." << std::endl;

        for(int scan_id = 0; scan_id < proj.scans.size(); scan_id++)
        {
            const ScanPosition &pos = proj.scans[scan_id];
            std::cout << "scan: " << scan_id << std::endl;

            ScanPtr sd = toScanPtr(pos);
            slamToLVR(sd);

            hdf5.addRawScan(scan_id, sd);
            
            for(int img_id = 0; img_id < pos.images.size(); img_id++)
            {
                const ImageFile& img = pos.images[img_id];
                std::cout << "\timage: " << img_id << std::endl;

                CameraData cd = toCamData(img);

                hdf5.addRawCamData(scan_id, img_id, cd);
            }
        }

        std::cout << "finished successfully" << std::endl;
    } else {
        std::cout << "Usage: " << argv[0] << " [ScanprojectDirectory]" << std::endl;
    }

    return 0;
}
