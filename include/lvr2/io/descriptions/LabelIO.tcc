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
    m_featureBase->m_kernel->savePointBuffer(groupPath.string(), "points.ply", labelRootPtr->points);
         
    //iterate over classes
    for(auto classPtr : labelRootPtr->labelClasses)
    {
        std::cout<< "clas" << std::endl;
        boost::filesystem::path classPath(classPtr->className);
        boost::filesystem::path totalPath(groupPath / classPath);
        std::string groupName = totalPath.string();
        for(auto instancePtr : classPtr->instances)
        {


            std::cout<< "instance" << std::endl;
            YAML::Node node;
            node = *instancePtr;
            
            std::string dataSetName = instancePtr->instanceName + std::string(".ids");
            std::string metaName = instancePtr->instanceName + std::string(".yaml");

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

            //save IDS
            std::vector<size_t>dim = {instancePtr->labeledIDs.size()};
            boost::shared_array<int>labeledIDs (new int[instancePtr->labeledIDs.size()]);
            std::memcpy(&labeledIDs, instancePtr->labeledIDs.data(), instancePtr->labeledIDs.size());
            m_arrayIO->saveIntArray(groupName, dataSetName, dim, labeledIDs);
        }
    }
}

template <typename Derived>
LabelRootPtr LabelIO<Derived>::loadLabels(const std::string& group)
{
    LabelRootPtr ret(new LabelRoot);
    std::vector<std::string> labelClasses;
    m_featureBase->m_kernel->subGroupNames(group, labelClasses);

    for (auto classGroup : labelClasses)
    {
    }
}

} // namespace lvr2
