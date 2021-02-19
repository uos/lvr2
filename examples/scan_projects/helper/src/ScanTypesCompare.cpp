#include "ScanTypesCompare.hpp"

namespace lvr2 {

bool equal(CameraImagePtr si1, CameraImagePtr si2)
{
    if(!si1 && si2){return false;}
    if(si1 && !si2){return false;}
    if(!si1 && !si2){return true;}

    if(!si1->transformation.isApprox(si2->transformation)){ 
        std::cout << "ScanImage transformation differ: " << std::endl;
        std::cout << si1->transformation << std::endl;
        std::cout << " != " << std::endl;
        std::cout << si2->transformation << std::endl;
        return false; 
    }

    if(!si1->extrinsicsEstimation.isApprox(si2->extrinsicsEstimation)){ 
        std::cout << "ScanImage extrinsicsEstimation differ: " << std::endl;
        std::cout << si1->extrinsicsEstimation << std::endl;
        std::cout << " != " << std::endl;
        std::cout << si2->extrinsicsEstimation << std::endl;
        return false; 
    }

    if(si1->image.rows != si2->image.rows)
    {
        std::cout << "Image rows differs" << std::endl;
        std::cout << si1->image.rows << " != " <<  si2->image.rows << std::endl;
        return false;
    }

    if(si1->image.cols != si2->image.cols)
    {
        std::cout << "Image cols differs" << std::endl;
        std::cout << si1->image.cols << " != " <<  si2->image.cols << std::endl;
        return false;
    }

    // if(cv::countNonZero(si1->image != si2->image) != 0){
    //     std::cout <<"ScanImage image data differ "  << std::endl;
    //     cv::imshow("ScanImage 1", si1->image);
    //     cv::imshow("ScanImage 2", si2->image);
    //     cv::waitKey(0);
    //     return false;
    // }

    return true;
}

bool equal(CameraPtr sc1, CameraPtr sc2)
{
    if(!sc1 && sc2){return false;}
    if(sc1 && !sc2){return false;}
    if(!sc1 && !sc2){return true;}

    if(!sc1->transformation.isApprox(sc2->transformation)){ 
        std::cout << "Camera transformation differ: " << std::endl;
        std::cout << sc1->transformation << std::endl;
        std::cout << " != " << std::endl;
        std::cout << sc2->transformation << std::endl;
        return false; 
    }

    if(!equal(sc1->model.cx, sc2->model.cx))
    {
        std::cout << "ScanCamera cx differ: " << sc1->model.cx << " != " << sc2->model.cx << std::endl;
        return false;
    }

    if(!equal(sc1->model.cy,sc2->model.cy) )
    {
        std::cout << "ScanCamera cy differ: "  << sc1->model.cy << " != " << sc2->model.cy << std::endl;
        return false;
    }

    if(!equal(sc1->model.fx,sc2->model.fx) )
    {
        std::cout << "ScanCamera fx differ: " << sc1->model.fx << " != " << sc2->model.fx <<  std::endl;
        return false;
    }

    if(!equal(sc1->model.fy, sc2->model.fy) )
    {
        std::cout << "ScanCamera fy differ: " << sc1->model.fy << " != " << sc2->model.fy << std::endl;
        return false;
    }

    if(sc1->name != sc2->name)
    {
        std::cout << "ScanCamera SensorName differ: " << sc1->name << " != " << sc2->name << std::endl;
        return false;
    }

    if(sc1->images.size() != sc2->images.size())
    {
        std::cout << "ScanCamera number of images differ: " << sc1->images.size() << " != "  << sc2->images.size() << std::endl;
        return false;
    }

    for(size_t i=0; i<sc1->images.size(); i++)
    {
        if(!equal(sc1->images[i], sc2->images[i]))
        {
            return false;
        }
    }

    return true;
}

bool equal(PointBufferPtr p1, PointBufferPtr p2)
{
    if(!p1 && p2){
        std::cout << "P1 is empty but not P2" << std::endl;
        return false;}
    if(p1 && !p2){
        std::cout << "P2 is empty but not P1" << std::endl;
        return false;}
    if(!p1 && !p2){return true;}

    for(auto elem : *p1)
    {
        auto it = p2->find(elem.first);
        if(it != p2->end())
        {
            // found channel in second pointbuffer
            if(elem.second.type() != it->second.type())
            {
                std::cout << "Type differ for " << elem.first << std::endl;
                return false;
            }

            if(elem.second.numElements() != it->second.numElements())
            {
                std::cout << "numElements differ for " << elem.first << std::endl;
                return false;
            }

            if(elem.second.width() != it->second.width())
            {
                std::cout << "width differ for " << elem.first << std::endl;
                return false;
            }

        } else {
            std::cout << "Could not find channel " << elem.first << " in p2" << std::endl;
            return false;
        }
    }

    return true;
}

bool equal(const float& a, const float& b)
{
    return abs(a - b) < std::numeric_limits<float>::epsilon();
}

bool equal(const double& a, const double& b)
{
    return abs(a - b) < std::numeric_limits<double>::epsilon();
}

bool equal(ScanPtr s1, ScanPtr s2)
{
    if(!s1 && s2){return false;}
    if(s1 && !s2){return false;}
    if(!s1 && !s2){return true;}

    if(!s1->transformation.isApprox(s2->transformation)){ 
        std::cout << "Scan transformation differ: " << std::endl;
        std::cout << s1->transformation << std::endl;
        std::cout << " != " << std::endl;
        std::cout << s2->transformation << std::endl;
        return false; 
    }

    if(!equal(s1->startTime, s2->startTime) ){
        std::cout << "Scan: startTime differs: " << s1->startTime << " <-> " << s2->startTime << std::endl;
        return false;}
    if(!equal(s1->endTime, s2->endTime) ){
        std::cout << "Scan: endTime differs: " << s1->endTime << " <-> " << s2->endTime << std::endl;
        return false;}
    if(s1->numPoints != s2->numPoints){
        std::cout << "Scan: numPoints differs: " << s1->numPoints << " <-> " << s2->numPoints << std::endl;
        return false;}
    if(!equal(s1->phiMin, s2->phiMin) ){
        std::cout << "Scan: phiMin differs: " << s1->phiMin << " <-> " << s2->phiMin << std::endl;
        return false;}
    if(!equal(s1->phiMax, s2->phiMax) ){
        std::cout << "Scan: phiMax differs: " << s1->phiMax << " <-> " << s2->phiMax << std::endl;
        return false;}
    if(!equal(s1->thetaMin, s2->thetaMin) ){
        std::cout << "Scan: thetaMin differs: " << s1->thetaMin << " <-> " << s2->thetaMin << std::endl;
        return false;}
    if(!equal(s1->thetaMax, s2->thetaMax) ){
        std::cout << "Scan: thetaMax differs: " << s1->thetaMax << " <-> " << s2->thetaMax << std::endl;
        return false;}
    
    if(s1->vResolution != s2->vResolution){
        std::cout << "Scan: vResolution differs: " << s1->vResolution << " <-> " << s2->vResolution << std::endl;
        return false;}
    
    if(s1->hResolution != s2->hResolution){
        std::cout << "Scan: hResolution differs: " << s1->hResolution << " <-> " << s2->hResolution << std::endl;
        return false;}

    if(!equal(s1->points, s2->points)){
        std::cout << "scan points differ" << std::endl;
        return false;}

    return true;
}

bool equal(LIDARPtr l1, LIDARPtr l2)
{
    if(!l1 && l2){return false;}
    if(l1 && !l2){return false;}
    if(!l1 && !l2){return true;}

    if(!l1->transformation.isApprox(l2->transformation)){ 
        std::cout << "LIDAR transformation differ: " << std::endl;
        std::cout << l1->transformation << std::endl;
        std::cout << " != " << std::endl;
        std::cout << l2->transformation << std::endl;
        return false; 
    }

    if(l1->scans.size() != l2->scans.size())
    {
        std::cout << "LIDAR nscans mismatch: " <<  l1->scans.size() << " != " << l2->scans.size() << std::endl;
        return false;
    }

    for(size_t i = 0; i<l1->scans.size(); i++)
    {
        if(!equal(l1->scans[i], l2->scans[i]))
        {
            return false;
        }
    }

    return true;
}

bool equal(ScanPositionPtr sp1, ScanPositionPtr sp2)
{
    if(!sp1 && sp2){return false;}
    if(sp1 && !sp2){return false;}
    if(!sp1 && !sp2){return true;}

    // gps
    // if(sp1->longitude != sp2->longitude){ return false; }
    // if(sp1->latitude != sp2->latitude){ return false; }
    // if(sp1->altitude != sp2->altitude){ return false; }
    // timestamp
    if(sp1->timestamp != sp2->timestamp){ 
        std::cout <<  "ScanPosition timestamp differs" << std::endl;
        return false; 
    }
    // pose
    if(!sp1->poseEstimation.isApprox(sp2->poseEstimation)){ 
        std::cout <<  "ScanPosition poseEstimation differs" << std::endl;
        return false; 
    }
    if(!sp1->transformation.isApprox(sp2->transformation)){ 
        std::cout <<  "ScanPosition transformation differs" << std::endl;
        return false; 
    }

    // vectors
    // if(sp1->scans.size() != sp2->scans.size()){return false;}
    if(sp1->cameras.size() != sp2->cameras.size()){
        std::cout <<  "ScanPosition Ncameras differs: " << sp1->cameras.size()  << " != " << sp2->cameras.size() << std::endl;
        return false;
    }

    // scans
    for(size_t i=0; i<sp1->lidars.size(); i++)
    {
        if(!equal(sp1->lidars[i], sp2->lidars[i]))
        {
            return false;
        }
    }

    // cams
    for(size_t i=0; i<sp1->cameras.size(); i++)
    {
        if(!equal(sp1->cameras[i], sp2->cameras[i]))
        {
            return false;
        }
    }

    return true;
}

bool equal(ScanProjectPtr sp1, ScanProjectPtr sp2)
{
    // equal shared ptr
    if(!sp1 && sp2){return false;}
    if(sp1 && !sp2){return false;}
    if(!sp1 && !sp2){return true;}

    if(sp1->coordinateSystem != sp2->coordinateSystem)
    {
        std::cout << "ScanProject coordinateSystem differ"  << std::endl;
        return false;
    }

    if(!sp1->transformation.isApprox(sp2->transformation) )
    {
        std::cout << "ScanProject transformation differ"  << std::endl;
        return false;
    }

    if(sp1->name != sp2->name)
    {
        std::cout << "ScanProject name differ"  << std::endl;
        return false;
    }

    if(sp1->positions.size() != sp2->positions.size())
    {
        std::cout << "ScanProject Npositions differ: "  << sp1->positions.size() << " != " << sp2->positions.size() << std::endl;
        return false;
    }

    for(size_t i=0; i<sp1->positions.size(); i++)
    {
        if(!equal(sp1->positions[i], sp2->positions[i]))
        {
            return false;
        }
    }

    return true;
}

} // namespace lvr2