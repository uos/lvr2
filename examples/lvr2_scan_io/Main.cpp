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
        ScanOptional scan_opt = Scan();

        // extract points to a pointBufferPtr
        PointBufferPtr points(new PointBuffer(sphere->getVertices(), sphere->numVertices()));

        // add points to scan object
        scan_opt->m_points = points;
        scan_opt->m_phiMin = i + 0.5;
        scan_opt->m_phiMax = i + 1.0;

        // create ScanPositionPtr
        ScanPositionPtr scan_pos(new ScanPosition());
        scan_pos->scan = scan_opt;

        // add scans to scanProjectPtr
        scan_proj->positions.push_back(scan_pos);

        scan_pos->scan->m_positionNumber = i;
    }

    scan_proj->pose = Transformd::Identity();

    std::cout << "--- WRITE TEST ---" << std::endl;
    saveScanProjectToDirectory("example_sav", *scan_proj);

    
    ScanProject in_scan_proj;
    std::cout << "--- READ TEST ---" << std::endl;
    loadScanProjectFromDirectory("example_sav", in_scan_proj);

    std::cout << in_scan_proj.positions[0]->scan->m_phiMin << std::endl;
    std::cout << in_scan_proj.positions[0]->scan->m_phiMax << std::endl;


    // auto pos_ids = loadPositionIdsFromDirectory("example_sav");
    
    // for(auto pos_id : pos_ids)
    // {
    //     std::cout << "scan: " << pos_id << std::endl;
    //     auto cam_ids = loadCamIdsFromDirectory("example_sav", pos_id);
    //     for(auto cam_id : cam_ids)
    //     {
    //         std::cout << "\tcam: " << cam_id << std::endl;
    //         auto img_ids = loadImageIdsFromDirectory("example_sav", pos_id, cam_id);
    //         for(auto img_id : img_ids)
    //         {
    //             std::cout << "\t\timg: " << img_id << std::endl;
    //         }
    //     }
    // }



    return 0;
}
