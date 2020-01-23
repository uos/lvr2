#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Synthetic.hpp"

using namespace lvr2;

int main(int argc, char** argv)
{
    // create ScanProjectPtr
    ScanProjectPtr scan_proj(new ScanProject());

    for (int i = 0; i < 4; i++)
    {
        // generate synthetic sphere
        MeshBufferPtr sphere = synthetic::genSphere(10, 10);

        // create scan object
        ScanPtr scan_ptr(new Scan);
        scan_ptr->positionNumber = i;

        // add points to scan object
        scan_ptr->points.reset(new PointBuffer(sphere->getVertices(), sphere->numVertices()));
        scan_ptr->phiMin = i + 0.5;
        scan_ptr->phiMax = i + 1.0;

        scan_ptr->numPoints = scan_ptr->points->numPoints();

        // create ScanPositionPtr
        ScanPositionPtr scan_pos(new ScanPosition);
        scan_pos->scans.push_back(scan_ptr);

        // add scans to scanProjectPtr
        scan_proj->positions.push_back(scan_pos);

        
        for(int cam_id=0; cam_id<5; cam_id++)
        {
            ScanCameraPtr cam(new ScanCamera);
            
            for(int img_id = 0; img_id < 5; img_id++)
            {
                ScanImagePtr img(new ScanImage);

                
                img->extrinsics = Extrinsicsd::Identity();
                img->extrinsicsEstimate = Extrinsicsd::Identity();
                img->image = cv::Mat(500, 1000, CV_8UC3, cv::Scalar(0, 0, 100));
                cam->images.push_back(img);
            }

            scan_pos->cams.push_back(cam);
        }
    }

    scan_proj->pose = Transformd::Identity();

    std::cout << "--- WRITE TEST ---" << std::endl;
    saveScanProject("scan_directory", *scan_proj);

    
    std::cout << "--- READ TEST ---" << std::endl;
    ScanProject in_scan_proj;
    loadScanProject("scan_directory", in_scan_proj);



    return 0;
}
