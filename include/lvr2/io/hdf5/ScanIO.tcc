
#include "ScanIO.hpp"

namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanIO<Derived>::save(std::string name, const ScanPtr& buffer)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, true);
    save(g, buffer);
}

template <typename Derived>
void ScanIO<Derived>::save(HighFive::Group& group, const ScanPtr& scan)
{

    std::string id(ScanIO<Derived>::ID);
    std::string obj(ScanIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // save points
    std::vector<size_t> scanDim = {scan->m_points->numPoints(), 3};
    std::vector<hsize_t> scanChunk = {scan->m_points->numPoints(), 3};
    boost::shared_array<float> test = scan->m_points->getPointArray();
    m_arrayIO->template save<float>(group, "points", scanDim, scanChunk, test);
    // m_pointCloudIO->save(group, scan->m_points); maybe use PointCloudIO?

    m_matrixIO->save(group, "finalPose", scan->m_registration);
    m_matrixIO->save(group, "initialPose", scan->m_poseEstimation);

    boost::shared_array<float> bbox(new float[6]);
    bbox.get()[0] = scan->m_boundingBox.getMin()[0];
    bbox.get()[1] = scan->m_boundingBox.getMin()[1];
    bbox.get()[2] = scan->m_boundingBox.getMin()[2];
    bbox.get()[3] = scan->m_boundingBox.getMax()[0];
    bbox.get()[4] = scan->m_boundingBox.getMax()[1];
    bbox.get()[5] = scan->m_boundingBox.getMax()[2];
    std::vector<hsize_t> chunkBB{2, 3};
    std::vector<size_t> dimBB{2, 3};
    m_arrayIO->save(group, "boundingBox", dimBB, chunkBB, bbox);

    bbox.get()[0] = scan->m_globalBoundingBox.getMin()[0];
    bbox.get()[1] = scan->m_globalBoundingBox.getMin()[1];
    bbox.get()[2] = scan->m_globalBoundingBox.getMin()[2];
    bbox.get()[3] = scan->m_globalBoundingBox.getMax()[0];
    bbox.get()[4] = scan->m_globalBoundingBox.getMax()[1];
    bbox.get()[5] = scan->m_globalBoundingBox.getMax()[2];
    m_arrayIO->save(group, "globalBoundingBox", dimBB, chunkBB, bbox);

    std::vector<hsize_t> chunkTwo{2};
    std::vector<size_t> dimTwo{2};

    boost::shared_array<float> fovArr(new float[2]);
    fovArr.get()[0] = scan->m_hFieldOfView;
    fovArr.get()[1] = scan->m_vFieldOfView;
    m_arrayIO->save(group, "fov", dimTwo, chunkTwo, fovArr);

    boost::shared_array<float> resolution(new float[2]);
    resolution.get()[0] = scan->m_hResolution;
    resolution.get()[1] = scan->m_vResolution;
    m_arrayIO->save(group, "resolution", dimTwo, chunkTwo, resolution);
}


template <typename Derived>
ScanPtr ScanIO<Derived>::load(std::string name)
{
    ScanPtr ret;

    if (hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret               = load(g);
    }

    return ret;
}

template <typename Derived>
ScanPtr ScanIO<Derived>::loadScan(std::string name)
{
    return load(name);
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(HighFive::Group& group)
{
    ScanPtr ret;
    /*
    if (!isScan(group))
    {
        std::cout << "[Hdf5IO - ScanIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return ret;
    }
    */
    ret = ScanPtr(new Scan());

    std::vector<size_t> dimensionPoints;
    floatArr pointArr;
    if(group.exist("points"))
    {
        pointArr = m_arrayIO->template load<float>(group, "points", dimensionPoints);
        if(dimensionPoints.at(1) != 3)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong point dimensions. Points will not be loaded." << std::endl;
        }
        else
        {
            ret->m_points = PointBufferPtr(new PointBuffer(pointArr, dimensionPoints.at(0)));
        }
    }
    boost::optional<lvr2::Transformd> finalPose = m_matrixIO->template load<lvr2::Transformd>(group, "finalPose");
    if(finalPose)
    {
        ret->m_registration = finalPose.get();
    }
    boost::optional<lvr2::Transformd> initialPose = m_matrixIO->template load<lvr2::Transformd>(group, "initialPose");
    if(initialPose)
    {
        ret->m_poseEstimation = initialPose.get();
    }


    if(group.exist("boundingBox"))
    {
        std::vector<size_t> dimBB;
        floatArr bbox = m_arrayIO->template load<float>(group, "boundingBox", dimBB);
        if((dimBB.at(0) != 2 || dimBB.at(1) != 3) && dimBB.at(0) != 6)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong boundingBox dimensions. BoundingBox will not be loaded." << std::endl;
        }
        else
        {
            BaseVector<float> min(bbox.get()[0], bbox.get()[1], bbox.get()[2]);
            BaseVector<float> max(bbox.get()[3], bbox.get()[4], bbox.get()[5]);
            BoundingBox<BaseVector<float>> boundingBox(min, max);
            ret->m_boundingBox = boundingBox;
        }
    }

    if(group.exist("globalBoundingBox"))
    {
        std::vector<size_t> dimBB;
        floatArr bbox = m_arrayIO->template load<float>(group, "globalBoundingBox", dimBB);
        if((dimBB.at(0) != 2 || dimBB.at(1) != 3) && dimBB.at(0) != 6)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong globalBoundingBox dimensions. GlobalBoundingBox will not be loaded." << std::endl;
        }
        else
        {
            BaseVector<float> min(bbox.get()[0], bbox.get()[1], bbox.get()[2]);
            BaseVector<float> max(bbox.get()[3], bbox.get()[4], bbox.get()[5]);
            BoundingBox<BaseVector<float>> boundingBox(min, max);
            ret->m_globalBoundingBox = boundingBox;
        }
    }
    if(group.exist("fov"))
    {
        std::vector<size_t> dimTwo;
        floatArr fovArr = m_arrayIO->template load<float>(group, "fov", dimTwo);
        if(dimTwo.at(0) != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong fov dimensions. FOV will not be loaded." << std::endl;
        }
        else
        {
            ret->m_hFieldOfView = fovArr.get()[0];
            ret->m_vFieldOfView = fovArr.get()[1];
        }
    }
    if(group.exist("resolution"))
    {
        std::vector<size_t> dimTwo;
        floatArr resArr = m_arrayIO->template load<float>(group, "resolution", dimTwo);
        if(dimTwo.at(0) != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong resolution dimensions. Resolution will not be loaded." << std::endl;
        }
        else
        {
            ret->m_hResolution = resArr.get()[0];
            ret->m_vResolution = resArr.get()[1];
        }
    }

    return ret;
}

template <typename Derived>
bool ScanIO<Derived>::isScan(
        HighFive::Group& group)
{
    std::string id(ScanIO<Derived>::ID);
    std::string obj(ScanIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id)
           && hdf5util::checkAttribute(group, "CLASS", obj);
}

template<typename Derived>
std::vector<ScanPtr> ScanIO<Derived>::loadAllScans(std::string groupName) {
    std::vector<ScanPtr> scans = std::vector<ScanPtr>();
    if (hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ScanPtr tmp;
        for(auto scanName : g.listObjectNames())
        {
            HighFive::Group scan = g.getGroup(scanName);
            tmp = load(scan);
            if(tmp)
            {
                scans.push_back(tmp);
            }
        }
    }
    return scans;
}


} // namespace hdf5features

} // namespace lvr2
