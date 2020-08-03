#include "lvr2/io/yaml/Label.hpp"
namespace lvr2
{

template <typename Derived>
void LabelIO<Derived>::saveLabels(
    std::string& group,
    LabelRootPtr labelRootPtr)
{
    //TODO Maybe add Description conatining the group and datasetname
    boost::filesystem::path pointCloud("pointCloud");
    boost::filesystem::path groupPath = (boost::filesystem::path(group) / pointCloud);
    m_featureBase->m_kernel->savePointBuffer(groupPath.string(), "points", labelRootPtr->points);

    for(auto classPtr : labelRootPtr->labelClasses)
    {
        boost::filesystem::path classPath(classPtr->className);
        boost::filesystem::path totalPath(groupPath / classPath);
        for(auto instancePtr : classPtr->instances)
        {
            boost::filesystem::path finalPath(totalPath / boost::filesystem::path(instancePtr->instanceName));
            std::string groupName = finalPath.string();
            /*
            YAML::Node node;
            node = *instancePtr;
            */
            /*std::string metaName = instancePtr->instanceName + std::string(".yaml");
            std::string dataSetName = instancePtr->instanceName + std::string(".ids");

            Description d = m_featureBase->m_description->labelInstance(group, classPtr->className, instancePtr->instanceName);
            if(d.groupName)
            {
                groupName = *d.groupName;
            }

            if(d.dataSetName)
            {
                dataSetName = *d.dataSetName;
            }

            if(d.metaName)
            {
                metaName = *d.metaName;
            }

            if(d.metaData)
            {
                node = *d.metaData;
            }

            //save meta
            m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
*/
            //save IDS
            int* sharedArrayData = new int[instancePtr->labeledIDs.size()];
            std::memcpy(sharedArrayData, instancePtr->labeledIDs.data(), instancePtr->labeledIDs.size());
            std::vector<size_t>dim = {instancePtr->labeledIDs.size()};
            boost::shared_array<int>labeledIDs (sharedArrayData);
            m_arrayIO->saveIntArray(groupName, "IDs", dim, labeledIDs);

            //save Color
            int* sharedColorData = new int[3];
            dim = {3};
            sharedColorData[0] = instancePtr->color[0];
            sharedColorData[1] = instancePtr->color[1];
            sharedColorData[2] = instancePtr->color[2];
            boost::shared_array<int>colors (sharedColorData);
            m_arrayIO->saveIntArray(groupName, "Color", dim, colors);
        }
    }
}

template <typename Derived>
LabelRootPtr LabelIO<Derived>::loadLabels(const std::string& group)
{
    LabelRootPtr ret(new LabelRoot);


    std::string scanName("points");
    //read Pointbuffer 
    boost::shared_array<float> pointData;
    std::vector<size_t> pointDim;
    pointData = m_featureBase->m_kernel->loadFloatArray(group, "points", pointDim);
    PointBufferPtr pb = PointBufferPtr(new PointBuffer(pointData, pointDim[0]));
    ret->points = pb;


    std::vector<std::string> labelClasses;
    m_featureBase->m_kernel->subGroupNames(group, labelClasses);
    boost::filesystem::path groupPath(group);
    for (auto classGroup : labelClasses)
    {
        if(classGroup == "points")
        {
            //TODO make it less hacky
            continue;
        }
        std::cout << classGroup << std::endl;
        boost::filesystem::path classPath(groupPath / boost::filesystem::path(classGroup));
        LabelClassPtr classPtr(new LabelClass);
        classPtr->className = classGroup;

        std::vector<std::string> labelInstances;
        m_featureBase->m_kernel->subGroupNames(classPath.string(), labelInstances);
        for(auto instanceGroup : labelInstances)
        {
            std::cout << instanceGroup << std::endl;
            LabelInstancePtr instancePtr(new LabelInstance);
            instancePtr->instanceName = instanceGroup;
            //Get Color and IDs
            boost::filesystem::path instancePath(instanceGroup);
            boost::filesystem::path finalPath(classPath / instancePath);
            boost::shared_array<int> rgbData;
            std::vector<size_t> rgbDim;
            boost::shared_array<int> idData;
            std::vector<size_t> idDim;
            idData = m_arrayIO->loadIntArray(finalPath.string(), "IDs", idDim);
            rgbData = m_arrayIO->loadIntArray(finalPath.string(), "Color", rgbDim);

            instancePtr->color[0] = rgbData[0];
            instancePtr->color[1] = rgbData[1];
            instancePtr->color[2] = rgbData[2];

            std::vector<int> tmp(idData.get(), idData.get() + idDim[0]);
            instancePtr->labeledIDs = std::move(tmp);

            classPtr->instances.push_back(instancePtr);
        }
        ret->labelClasses.push_back(classPtr);
    }
    return ret;
}

} // namespace lvr2
