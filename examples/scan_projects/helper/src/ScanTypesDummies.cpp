#include "ScanTypesDummies.hpp"

#include "lvr2/util/Synthetic.hpp"

namespace lvr2 {

PointBufferPtr operator*(Transformd T, PointBufferPtr points)
{
    PointBufferPtr ret(new PointBuffer);

    for(auto elem : *points)
    {
        (*ret)[elem.first] = elem.second;
    }

    if(ret->hasChannel<float>("points"))
    {
        Channel<float> points = ret->get<float>("points");
    
        for(size_t i=0; i<points.numElements(); i++)
        {
            Vector4d p(points[i][0], points[i][1], points[i][2], 1);
            Vector4d p_ = T * p;
            points[i][0] = p_(0);
            points[i][1] = p_(1);
            points[i][2] = p_(2);
        }

        ret->add<float>("points", points);
    }

    if(ret->hasChannel<float>("normals"))
    {
        Channel<float> normals = ret->get<float>("normals");

        for(size_t i=0; i<normals.numElements(); i++)
        {
            Vector3d n(normals[i][0], normals[i][1], normals[i][2]);
            Vector3d n_ = T.block<3,3>(0,0) * n;
            normals[i][0] = n_(0);
            normals[i][1] = n_(1);
            normals[i][2] = n_(2);
        }

        ret->add("normals", normals);
    }

    return ret;
}

ScanProjectPtr dummyScanProject()
{
    ScanProjectPtr ret(new ScanProject);

    PointBufferPtr points = synthetic::genSpherePoints(200, 200);
    size_t npoints = points->numPoints();
    Channel<float> normals(npoints, 3);

    for(size_t i=0; i<npoints; i++)
    {
        normals[i][0] = 1.0;
        normals[i][1] = 0.0;
        normals[i][2] = 0.0;
    }
    points->add("normals", normals);

    for(size_t i=0; i<10; i++)
    {
        ScanPositionPtr scan_pos(new ScanPosition);
         
        scan_pos->transformation = Transformd::Identity();
        scan_pos->transformation(0,3) = static_cast<double>(i);
        scan_pos->poseEstimation = scan_pos->transformation;

        for(size_t j=0; j<2; j++)
        {
            LIDARPtr lidar(new LIDAR);
            lidar->name = "Riegl";
            lidar->transformation = Transformd::Identity();
            lidar->transformation(1,3) = static_cast<double>(j);

            for(size_t k=0; k<10; k++)
            {
                ScanPtr scan(new Scan);

                SphericalModel model;
                model.principal = Eigen::Vector3d(0.0, 0.0, 0.0);
                model.phi[0] = -M_PI;
                model.phi[1] = M_PI;
                model.phi[2] = 0.2;

                model.theta[0] = -M_PI;
                model.theta[1] = M_PI;
                model.theta[2] = 1.0;

                model.range[0] = 0.5;
                model.range[1] = 600.0;
                model.range[2] = 0.001;

                scan->model = model;
                
                scan->poseEstimation = lvr2::Transformd::Identity();
                scan->transformation = lvr2::Transformd::Identity();
                scan->transformation(2,3) = static_cast<double>(k);

                Transformd T = Transformd::Identity();
                T(2,3) = static_cast<double>(i);
                scan->points = T * points;

                scan->numPoints = scan->points->numPoints();
                scan->startTime = 0.0;
                scan->endTime  = 100.0;

                lidar->scans.push_back(scan);
            }
            
            scan_pos->lidars.push_back(lidar);
        }

        for(size_t j=0; j<2; j++)
        {
            CameraPtr scan_cam(new Camera);
            scan_cam->transformation = Transformd::Identity();
            scan_cam->transformation(1,3) = -static_cast<double>(j);
            scan_cam->model.distortionModel = "opencv";
            scan_cam->model.distortionCoefficients.resize(10);
            scan_cam->model.cx = 100.2;
            scan_cam->model.cy = 50.5;
            scan_cam->model.fx = 120.99;
            scan_cam->model.fy = 90.72;

            for(size_t k=0; k<10; k++)
            {
                scan_cam->model.distortionCoefficients[k] = static_cast<double>(k) / 4.0;
            }
            
            // image groups
            for(size_t k=0; k<2; k++)
            {
                CameraImageGroupPtr g(new CameraImageGroup);
                g->transformation.setIdentity();
                
                for(size_t l=0; l<3; l++)
                {
                    CameraImagePtr si = synthetic::genLVRImage();
                    si->timestamp = 0.0;
                    si->transformation = Transformd::Identity();
                    si->transformation(2,3) = -static_cast<double>(l);
                    si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(l + 1);
                    g->images.push_back(si);
                }
                scan_cam->images.push_back(g);
            }

            // images at top level
            for(size_t k=0; k<3; k++)
            {
                CameraImagePtr si = synthetic::genLVRImage();
                si->timestamp = 0.0;
                si->transformation = Transformd::Identity();
                si->transformation(2,3) = -static_cast<double>(k);
                si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(k + 1);
                scan_cam->images.push_back(si);
            }

            scan_cam->name = "Canon";
            scan_pos->cameras.push_back(scan_cam);
        }

        for(size_t j=0; j<2; j++)
        {
            HyperspectralCameraPtr h_cam(new HyperspectralCamera);

            h_cam->transformation = Transformd::Identity();
            h_cam->transformation(1,3) = -static_cast<double>(j);

            h_cam->model.principal(0) =  5.5;
            h_cam->model.principal(1) = 4.4;

            h_cam->model.focalLength(0) = 10.1;
            h_cam->model.focalLength(1) = 10.2;

            h_cam->model.distortionCoefficients.resize(3);
            h_cam->model.distortionCoefficients[0] = 2.0;
            h_cam->model.distortionCoefficients[1] = 1.0;
            h_cam->model.distortionCoefficients[2] = 0.5;

            for(size_t k=0; k<3; k++)
            {
                HyperspectralPanoramaPtr pano(new HyperspectralPanorama);

                pano->framesResolution[0] = 900;
                pano->framesResolution[1] = 2;
                pano->framesResolution[2] = 150;
                
                pano->bandAxis = 2;
                pano->frameAxis = 1;
                pano->dataType = "uint8";
                
                pano->panoramaResolution[0] = 900;
                pano->panoramaResolution[1] = 7001;
                pano->panoramaResolution[2] = 150;

                // override sensor model
                pano->model = h_cam->model;

                for(size_t l=0; l<7; l++)
                {
                    HyperspectralPanoramaChannelPtr hchannel(new HyperspectralPanoramaChannel);

                    CameraImagePtr si = synthetic::genLVRImage();
                    hchannel->channel = si->image.clone();
                    hchannel->timestamp = 0.0;
                    pano->channels.push_back(hchannel);
                }

                h_cam->panoramas.push_back(pano);
            }

            scan_pos->hyperspectral_cameras.push_back(h_cam);
        }

        ret->positions.push_back(scan_pos);
    }

    ret->unit = "meter";
    ret->coordinateSystem = "right-handed";
    ret->transformation = Transformd::Identity();

    return ret;
}


ScanProjectPtr dummyScanProjectMinimal()
{
    ScanProjectPtr ret(new ScanProject);

    PointBufferPtr points = synthetic::genSpherePoints(200, 200);
    size_t npoints = points->numPoints();
    Channel<float> normals(npoints, 3);

    for(size_t i=0; i<npoints; i++)
    {
        normals[i][0] = 1.0;
        normals[i][1] = 0.0;
        normals[i][2] = 0.0;
    }
    points->add("normals", normals);

    for(size_t i=0; i<1; i++)
    {
        ScanPositionPtr scan_pos(new ScanPosition);
         
        scan_pos->transformation = Transformd::Identity();
        scan_pos->transformation(0,3) = static_cast<double>(i);
        scan_pos->poseEstimation = scan_pos->transformation;

        for(size_t j=0; j<1; j++)
        {
            LIDARPtr lidar(new LIDAR);
            lidar->name = "Riegl VZ400i";
            lidar->transformation = Transformd::Identity();
            lidar->transformation(1,3) = static_cast<double>(j);

            for(size_t k=0; k<1; k++)
            {
                ScanPtr scan(new Scan);

                SphericalModel model;
                model.principal = Eigen::Vector3d(0.0, 0.0, 0.0);
                model.phi[0] = -M_PI;
                model.phi[1] = M_PI;
                model.phi[2] = 0.2;

                model.theta[0] = -M_PI;
                model.theta[1] = M_PI;
                model.theta[2] = 1.0;

                model.range[0] = 0.5;
                model.range[1] = 600.0;
                model.range[2] = 0.001;

                scan->model = model;

                scan->poseEstimation = lvr2::Transformd::Identity();
                scan->transformation = lvr2::Transformd::Identity();
                scan->transformation(2,3) = static_cast<double>(k);

                Transformd T = Transformd::Identity();
                T(2,3) = static_cast<double>(i);
                scan->points = T * points;

                scan->numPoints = scan->points->numPoints();
                scan->startTime = 0.0;
                scan->endTime  = 100.0;

                lidar->scans.push_back(scan);
            }
            
            scan_pos->lidars.push_back(lidar);
        }

        for(size_t j=0; j<1; j++)
        {
            CameraPtr scan_cam(new Camera);
            scan_cam->transformation = Transformd::Identity();
            scan_cam->transformation(1,3) = -static_cast<double>(j);
            scan_cam->model.distortionModel = "opencv";
            scan_cam->model.distortionCoefficients.resize(5);
            scan_cam->model.cx = 100.2;
            scan_cam->model.cy = 50.5;
            scan_cam->model.fx = 120.99;
            scan_cam->model.fy = 90.72;

            for(size_t k=0; k<5; k++)
            {
                scan_cam->model.distortionCoefficients[k] = static_cast<double>(k) / 4.0;
            }
            
            // image groups
            for(size_t k=0; k<1; k++)
            {
                CameraImageGroupPtr g(new CameraImageGroup);
                g->transformation.setIdentity();
                
                for(size_t l=0; l<1; l++)
                {
                    CameraImagePtr si = synthetic::genLVRImage();
                    si->timestamp = 0.0;
                    si->transformation = Transformd::Identity();
                    si->transformation(2,3) = -static_cast<double>(l);
                    si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(l + 1);
                    g->images.push_back(si);
                }
                scan_cam->images.push_back(g);
            }

            // // images at top level
            // for(size_t k=0; k<3; k++)
            // {
            //     CameraImagePtr si = synthetic::genLVRImage();
            //     si->timestamp = 0.0;
            //     si->transformation = Transformd::Identity();
            //     si->transformation(2,3) = -static_cast<double>(k);
            //     si->extrinsicsEstimation = Extrinsicsd::Identity() / static_cast<double>(k + 1);
            //     scan_cam->images.push_back(si);
            // }

            scan_cam->name = "Nikkon610d";
            scan_pos->cameras.push_back(scan_cam);
        }

        for(size_t j=0; j<1; j++)
        {
            HyperspectralCameraPtr h_cam(new HyperspectralCamera);

            h_cam->name = "Resonon Pika L";

            h_cam->transformation = Transformd::Identity();
            h_cam->transformation(1,3) = -static_cast<double>(j);

            h_cam->model.principal(0) =  5.5;
            h_cam->model.principal(1) = 4.4;

            h_cam->model.focalLength(0) = 10.1;
            h_cam->model.focalLength(1) = 10.2;

            h_cam->model.distortionCoefficients.resize(3);
            h_cam->model.distortionCoefficients[0] = 2.0;
            h_cam->model.distortionCoefficients[1] = 1.0;
            h_cam->model.distortionCoefficients[2] = 0.5;

            for(size_t k=0; k<1; k++)
            {
                HyperspectralPanoramaPtr pano(new HyperspectralPanorama);

                pano->framesResolution[0] = 900;
                pano->framesResolution[1] = 2;
                pano->framesResolution[2] = 150;
                
                pano->bandAxis = 2;
                pano->frameAxis = 1;
                pano->dataType = "uint8";
                
                pano->panoramaResolution[0] = 900;
                pano->panoramaResolution[1] = 7001;
                pano->panoramaResolution[2] = 150;

                // override sensor model
                pano->model = h_cam->model;

                for(size_t l=0; l<2; l++)
                {
                    HyperspectralPanoramaChannelPtr hchannel(new HyperspectralPanoramaChannel);

                    CameraImagePtr si = synthetic::genLVRImage();
                    hchannel->channel = si->image.clone();
                    hchannel->timestamp = 0.0;
                    pano->channels.push_back(hchannel);
                }

                h_cam->panoramas.push_back(pano);
            }

            scan_pos->hyperspectral_cameras.push_back(h_cam);
        }

        ret->positions.push_back(scan_pos);
    }

    ret->unit = "meter";
    ret->coordinateSystem = "right-handed";
    ret->transformation = Transformd::Identity();

    return ret;
}


} // namespace lvr2
