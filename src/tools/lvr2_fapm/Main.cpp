#include <iostream>
// #include "lvr2/algorithm/fapm/FeatureProjector.hpp"
#include "lvr2/algorithm/fapm/PixelProjector.hpp"
#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/types/Channel.hpp"
#include "lvr2/io/descriptions/DirectoryIO.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include <boost/range/iterator_range.hpp>


#include "lvr2/types/ScanTypes.hpp"

#include "lvr2/io/yaml/ScanCamera.hpp"
#include "lvr2/io/yaml/ScanImage.hpp"

// 1. EMBREE
// 2. OpenCL
// 3. BVH

#if defined LVR2_USE_EMBREE
#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"
using RaycasterImpl = lvr2::EmbreeRaycaster; 
#elif defined LVR2_USE_OPENCL
#include "lvr2/algorithm/raycasting/CLRaycaster.hpp"
using RaycasterImpl = lvr2::CLRaycaster;
#else
#include "lvr2/algorithm/raycasting/BVHRaycaster.hpp"
using RaycasterImpl = lvr2::BVHRaycaster;
#endif

using namespace lvr2;


namespace bfs = boost::filesystem;

Channel<float> convertToChannel(const std::vector<Vector3f>& in)
{
    Channel<float> out(in.size(), 3);

    for(int i=0; i < in.size(); i++)
    {
        out[i][0] = in[i].x();
        out[i][1] = in[i].y();
        out[i][2] = in[i].z();
    }

    return out;
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " [mesh] [campath]" << std::endl;
        return 0;
    }

    ModelPtr model = ModelFactory::readModel(argv[1]);
    if(!model->m_mesh)
    {
        std::cout << "Model does not contain a mesh. " << std::endl;
        return 0;
    }

    std::cout << "Initialize Raycasters acceleration structure" << std::endl;
    RaycasterBasePtr rc(new RaycasterImpl(model->m_mesh));
   

    std::cout << "Opening reconstructed surface data from: " << argv[1] << std::endl;
    std::string directory = argv[2];
    std::cout << "Opening cam: " << directory << std::endl;


    bfs::path cam_path = {directory};
    bfs::path cam_meta_file = cam_path / "meta.yaml";
    bfs::path cam_image_path = cam_path / "data";
    
    YAML::Node meta = YAML::LoadFile(cam_meta_file.string());
    ScanCameraPtr scan_cam = std::make_shared<ScanCamera>(meta.as<ScanCamera>());
    

    std::cout << "Scan Camera" << std::endl;
    std::cout << "-- Intrinsics: " << std::endl;
    std::cout << "Center: " << scan_cam->camera.cx << " " << scan_cam->camera.cy << std::endl;
    std::cout << "Focal: " << scan_cam->camera.fx << " " << scan_cam->camera.fy << std::endl;
    std::cout << "Resolution: " << scan_cam->camera.width << " " << scan_cam->camera.height << std::endl;
    

    std::cout << "Loading images pathes..." << std::endl;

    for(auto& entry : boost::make_iterator_range(bfs::directory_iterator(cam_image_path), {}))
    {
        bfs::path p(entry);
        
        if(p.extension().string() == ".yaml")
        {
            bfs::path image_file = cam_image_path / (p.stem().string() + ".png");
            
            YAML::Node meta = YAML::LoadFile(p.string());
            ScanImage scan_image;

            
            if(!YAML::convert<lvr2::ScanImage>::decode(meta, scan_image) )
            {
                std::cout << "Could not load " << p << " as ScanImage. Skip" << std::endl;
                continue;
            }
            
            scan_image.imageFile = image_file;
            scan_cam->images.push_back(std::make_shared<ScanImage>(scan_image));
            // project Freatures
        }
    }

    
    // cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    // cv::Ptr<cv::Feature2D> descriptor = cv::xfeatures2d::SIFT::create();

    std::cout << "Initialize FeatureProjector" << std::endl;
    PixelProjectorPtr pp(new PixelProjector(rc, scan_cam));

    std::cout << "Project Features" << std::endl;

    std::vector<Vector3f> points_tmp;

    for(int i = 0; i<scan_cam->images.size(); i++)
    {
        std::cout << "Project image: " << i << std::endl; 
        ScanImagePtr scan_image = scan_cam->images[i];
        
        if(i != 2)
        {
            continue;
        }
        
        scan_image->image = cv::imread(scan_image->imageFile.string());
        cv::imshow("img", scan_image->image);
        cv::waitKey();

        cv::Point2f pixel(0.0, 0.0);

        std::vector<std::vector<uint8_t> > hits;
        std::vector<std::vector<Vector3f> > intersections;

        pp->project(scan_image, hits, intersections);
        // uint8_t hit;
        // Vector3f intersection;
        // pp->project(scan_image, pixel, hit, intersection);

        for(int i=0; i<hits.size(); i++)
        {
            for(int j=0; j<hits[i].size(); j++)
            {
                if(hits[i][j])
                {
                    points_tmp.push_back(intersections[i][j]);
                }
            }
        }

        
    }

    std::cout << "Got " << points_tmp.size() << " intersections" << std::endl;

    Channel<float> points = convertToChannel(points_tmp);

    std::cout << "Saving snapshot" << std::endl;
    PointBufferPtr pbuffer(new PointBuffer);
    pbuffer->add("points", points);

    ModelPtr m(new Model(pbuffer));

    ModelFactory::saveModel(m, "test.ply");

    return 0;
}
