#include "ScanIO.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"

namespace lvr2
{

    namespace hdf5features
    {

        template <typename Derived>
        void ScanIO<Derived>::save(std::string name, const Scan& buffer)
        {
            HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, true);
            save(g, buffer);
        }

        template <typename Derived>
        void ScanIO<Derived>::save(HighFive::Group& group, const Scan& scan)
        {

            std::string id(ScanIO<Derived>::ID);
            std::string obj(ScanIO<Derived>::OBJID);
            hdf5util::setAttribute(group, "IO", id);
            hdf5util::setAttribute(group, "CLASS", obj);

            // save points
            std::vector<size_t> scanDim = {scan.m_points->numPoints(), 3};
            std::vector<hsize_t> scanChunk = {scan.m_points->numPoints(), 3};
            boost::shared_array<float> points = scan.m_points->getPointArray();
            m_arrayIO->template save<float>(group, "points", scanDim, scanChunk, points);

            m_matrixIO->save(group, "finalPose", scan.m_registration);
            m_matrixIO->save(group, "initialPose", scan.m_poseEstimation);

            boost::shared_array<float> bbox(new float[6]);
            bbox.get()[0] = scan.m_boundingBox.getMin()[0];
            bbox.get()[1] = scan.m_boundingBox.getMin()[1];
            bbox.get()[2] = scan.m_boundingBox.getMin()[2];
            bbox.get()[3] = scan.m_boundingBox.getMax()[0];
            bbox.get()[4] = scan.m_boundingBox.getMax()[1];
            bbox.get()[5] = scan.m_boundingBox.getMax()[2];
            std::vector<hsize_t> chunkBB{2, 3};
            std::vector<size_t> dimBB{2, 3};
            m_arrayIO->save(group, "boundingBox", dimBB, chunkBB, bbox);

            std::vector<hsize_t> chunkTwo{2};
            std::vector<size_t> dimTwo{2};

            boost::shared_array<float> phiArr(new float[2]);
            phiArr.get()[0] = scan.m_phiMin;
            phiArr.get()[1] = scan.m_phiMax;
            m_arrayIO->save(group, "phi", dimTwo, chunkTwo, phiArr);

            boost::shared_array<float> thetaArr(new float[2]);
            phiArr.get()[0] = scan.m_thetaMin;
            phiArr.get()[1] = scan.m_thetaMax;
            m_arrayIO->save(group, "phi", dimTwo, chunkTwo, phiArr);

            boost::shared_array<float> resolution(new float[2]);
            resolution.get()[0] = scan.m_hResolution;
            resolution.get()[1] = scan.m_vResolution;
            m_arrayIO->save(group, "resolution", dimTwo, chunkTwo, resolution);

            boost::shared_array<float> timestamp(new float[2]);
            timestamp.get()[0] = scan.m_startTime;
            timestamp.get()[1] = scan.m_endTime;
            m_arrayIO->save(group, "timestamp", dimTwo, chunkTwo, timestamp);
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
            //HighFive::Group preview = hdf5util::getGroup(m_file_access->m_hdf5_file, "preview/position_00001", false);;

            ret = ScanPtr(new Scan());

            std::vector<size_t> dimensionPoints;
            floatArr pointArr;
            if(group.exist("points"))
            {
                //pointArr = m_arrayIO->template load<float>(preview, "points", dimensionPoints);
                pointArr = m_arrayIO->template load<float>(group, "points", dimensionPoints);
                if(dimensionPoints.at(1) != 3)
                {
                    std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong point dimensions. Points will not be loaded." << std::endl;
                }
                else
                {
                    ret->m_points = PointBufferPtr(new PointBuffer(pointArr, dimensionPoints.at(0)));
                    ret->m_numPoints = dimensionPoints.at(0);
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

            if(group.exist("phi"))
            {
                std::vector<size_t> dimTwo;
                floatArr phiArr = m_arrayIO->template load<float>(group, "phi", dimTwo);
                if(dimTwo.at(0) != 2)
                {
                    std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong phi dimensions. Phi will not be loaded." << std::endl;
                }
                else
                {
                    ret->m_phiMin = phiArr.get()[0];
                    ret->m_phiMax = phiArr.get()[1];
                }
            }
            if(group.exist("theta"))
            {
                std::vector<size_t> dimTwo;
                floatArr thetaArr = m_arrayIO->template load<float>(group, "theta", dimTwo);
                if(dimTwo.at(0) != 2)
                {
                    std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong theta dimensions. Theta will not be loaded." << std::endl;
                }
                else
                {
                    ret->m_thetaMin = thetaArr.get()[0];
                    ret->m_thetaMax = thetaArr.get()[1];
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
            if(group.exist("timestamp"))
            {
                std::vector<size_t> dimTwo;
                floatArr timestampArr = m_arrayIO->template load<float>(group, "timestamp", dimTwo);
                if(dimTwo.at(0) != 2)
                {
                    std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong timestamp dimensions. Timestamp will not be loaded." << std::endl;
                }
                else
                {
                    ret->m_startTime = timestampArr.get()[0];
                    ret->m_endTime = timestampArr.get()[1];
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

        template <typename Derived>
        ScanPtr ScanIO<Derived>::loadPreview(std::string name) {
            ScanPtr ret;
            if (hdf5util::exist(m_file_access->m_hdf5_file, name))
            {

                HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
                std::vector <std::string> splitGroup = hdf5util::splitGroupNames(name);
                std::string previewString = "/preview/" + splitGroup.back();

                ret = ScanPtr(new Scan());

                if (hdf5util::exist(m_file_access->m_hdf5_file, previewString))
                {
                    HighFive::Group previewGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, previewString, false);
                    std::vector <size_t> dimensionPoints;
                    floatArr pointArr;
                    if (previewGroup.exist("points"))
                    {
                        pointArr = m_arrayIO->template load<float>(previewGroup, "points", dimensionPoints);
                        if (dimensionPoints.at(1) != 3)
                        {
                            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong point dimensions. Points will not be loaded."
                                      << std::endl;
                        }
                        else
                        {
                            ret->m_points = PointBufferPtr(new PointBuffer(pointArr, dimensionPoints.at(0)));
                            ret->m_numPoints = dimensionPoints.at(0);
                        }
                    }
                }
                else // use normal points
                {
                    std::cout << "Didn't found the Preview-Group. Try to use normal Points." << std::endl;
                    std::vector <size_t> dimensionPoints;
                    floatArr pointArr;
                    if (group.exist("points"))
                    {
                        pointArr = m_arrayIO->template load<float>(group, "points", dimensionPoints);
                        if (dimensionPoints.at(1) != 3)
                        {
                            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong point dimensions. Points will not be loaded."
                                      << std::endl;
                        }
                        else
                        {
                            ret->m_points = PointBufferPtr(new PointBuffer(pointArr, dimensionPoints.at(0)));
                            ret->m_numPoints = dimensionPoints.at(0);
                        }
                    }
                }

                boost::optional <lvr2::Transformd> finalPose = m_matrixIO->template load<lvr2::Transformd>(group,
                                                                                                           "finalPose");
                if (finalPose)
                {
                    ret->m_registration = finalPose.get();
                }
                boost::optional <lvr2::Transformd> initialPose = m_matrixIO->template load<lvr2::Transformd>(group,
                                                                                                             "initialPose");
                if (initialPose)
                {
                    ret->m_poseEstimation = initialPose.get();
                }


                if (group.exist("boundingBox"))
                {
                    std::vector <size_t> dimBB;
                    floatArr bbox = m_arrayIO->template load<float>(group, "boundingBox", dimBB);
                    if ((dimBB.at(0) != 2 || dimBB.at(1) != 3) && dimBB.at(0) != 6)
                    {
                        std::cout
                                << "[Hdf5IO - ScanIO] WARNING: Wrong boundingBox dimensions. BoundingBox will not be loaded."
                                << std::endl;
                    }
                    else
                    {
                        BaseVector<float> min(bbox.get()[0], bbox.get()[1], bbox.get()[2]);
                        BaseVector<float> max(bbox.get()[3], bbox.get()[4], bbox.get()[5]);
                        BoundingBox <BaseVector<float>> boundingBox(min, max);
                        ret->m_boundingBox = boundingBox;
                    }
                }

                if (group.exist("phi"))
                {
                    std::vector <size_t> dimTwo;
                    floatArr phiArr = m_arrayIO->template load<float>(group, "phi", dimTwo);
                    if (dimTwo.at(0) != 2)
                    {
                        std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong phi dimensions. Phi will not be loaded."
                                  << std::endl;
                    }
                    else
                    {
                        ret->m_phiMin = phiArr.get()[0];
                        ret->m_phiMax = phiArr.get()[1];
                    }
                }
                if (group.exist("theta"))
                {
                    std::vector <size_t> dimTwo;
                    floatArr thetaArr = m_arrayIO->template load<float>(group, "theta", dimTwo);
                    if (dimTwo.at(0) != 2)
                    {
                        std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong theta dimensions. Theta will not be loaded."
                                  << std::endl;
                    }
                    else
                    {
                        ret->m_thetaMin = thetaArr.get()[0];
                        ret->m_thetaMax = thetaArr.get()[1];
                    }
                }
                if (group.exist("resolution"))
                {
                    std::vector <size_t> dimTwo;
                    floatArr resArr = m_arrayIO->template load<float>(group, "resolution", dimTwo);
                    if (dimTwo.at(0) != 2)
                    {
                        std::cout
                                << "[Hdf5IO - ScanIO] WARNING: Wrong resolution dimensions. Resolution will not be loaded."
                                << std::endl;
                    }
                    else
                    {
                        ret->m_hResolution = resArr.get()[0];
                        ret->m_vResolution = resArr.get()[1];
                    }
                }
                if (group.exist("timestamp"))
                {
                    std::vector <size_t> dimTwo;
                    floatArr timestampArr = m_arrayIO->template load<float>(group, "timestamp", dimTwo);
                    if (dimTwo.at(0) != 2)
                    {
                        std::cout
                                << "[Hdf5IO - ScanIO] WARNING: Wrong timestamp dimensions. Timestamp will not be loaded."
                                << std::endl;
                    }
                    else
                    {
                        ret->m_startTime = timestampArr.get()[0];
                        ret->m_endTime = timestampArr.get()[1];
                    }
                }
            }
            return ret;
        }

    } // namespace hdf5features

} // namespace lvr2
