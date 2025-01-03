#include "lvr2/io/scanio/yaml/Label.hpp"
namespace lvr2
{

namespace scanio
{

template <typename Derived>
void LabelIO<Derived>::saveLabels(
    std::string& groupName,
    LabelRootPtr labelRootPtr) const
{
    //TODO Maybe add Description conatining the group and datasetname
    boost::filesystem::path pointCloud("pointCloud");
    boost::filesystem::path groupPath = (boost::filesystem::path(groupName) / pointCloud);
    m_baseIO->m_kernel->savePointBuffer(groupPath.string(), "points", labelRootPtr->points);

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

            Description d = m_baseIO->m_description->labelInstance(group, classPtr->className, instancePtr->instanceName);
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
            m_baseIO->m_kernel->saveMetaYAML(groupName, metaName, node);
*/
            //save IDS
            int* sharedArrayData = new int[instancePtr->labeledIDs.size()];
            std::memcpy(sharedArrayData, instancePtr->labeledIDs.data(), instancePtr->labeledIDs.size()*sizeof(int));
            std::vector<size_t>dim = {instancePtr->labeledIDs.size()};
            boost::shared_array<int>labeledIDs (sharedArrayData);
            m_arrayIO->saveIntArray(groupName, "IDs", dim, labeledIDs);

            //save Color
            int* sharedColorData = new int[3];
            dim = {3};
            sharedColorData[0] = instancePtr->color[0];
            sharedColorData[1] = instancePtr->color[1];
            sharedColorData[2] = instancePtr->color[2];
            boost::shared_array<int> colors(sharedColorData);
            m_arrayIO->saveIntArray(groupName, "Color", dim, colors);
        }
    }
    //save Waveform
    if(labelRootPtr->waveform)
    {
        lvr2::logout::get() << lvr2::info << "[LabelIO] Saving Waveform" << lvr2::endl;
        m_fullWaveformIO->saveLabelWaveform(groupPath.string(),labelRootPtr->waveform);      
    }else
    {
        lvr2::logout::get() << lvr2::info << "[LabelIO] No Waveform" << lvr2::endl;
    }

}

template <typename Derived>
LabelRootPtr LabelIO<Derived>::loadLabels(const std::string& group) const
{
    LabelRootPtr ret(new LabelRoot);


    std::string scanName("points");
    //read Pointbuffer 
    boost::shared_array<float> pointData;
    std::vector<size_t> pointDim;
    pointData = m_baseIO->m_kernel->loadFloatArray(group, "points", pointDim);
    PointBufferPtr pb = PointBufferPtr(new PointBuffer(pointData, pointDim[0]));
    ret->points = pb;


    std::vector<std::string> labelClasses;
    m_baseIO->m_kernel->subGroupNames(group, labelClasses);
    boost::filesystem::path groupPath(group);
    for (auto classGroup : labelClasses)
    {
        if(classGroup == "points" || classGroup == "waveform")
        {
            //TODO make it less hacky
            continue;
        }
        boost::filesystem::path classPath(groupPath / boost::filesystem::path(classGroup));
        LabelClassPtr classPtr(new LabelClass);
        classPtr->className = classGroup;

        std::vector<std::string> labelInstances;
        m_baseIO->m_kernel->subGroupNames(classPath.string(), labelInstances);
        for(auto instanceGroup : labelInstances)
        {
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

    //read Waveform
    boost::filesystem::path waveformPath(groupPath / boost::filesystem::path("waveform"));
    if (m_baseIO->m_kernel->exists(waveformPath.string()))
    {
            lvr2::logout::get() << lvr2::info << "[LabelIO] Read Waveform" << lvr2::endl;
            WaveformPtr fwPtr = m_fullWaveformIO->loadLabelWaveform(groupPath.string());
            ret->waveform = fwPtr;
    }
    return ret;

}

} // namespace scanio

} // namespace lvr2
