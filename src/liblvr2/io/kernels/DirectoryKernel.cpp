#include "lvr2/io/kernels/DirectoryKernel.hpp"

#include <boost/range/iterator_range.hpp>

namespace lvr2
{

void DirectoryKernel::saveMeshBuffer(
    const std::string &group,
    const std::string &container,
    const MeshBufferPtr &buffer) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    if(!boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directories(p.parent_path());
    }

    ModelPtr model(new Model);
    model->m_mesh = buffer;
    // std::cout << timestamp << "Directory Kernel::saveMeshBuffer(): " << p.string() << std::endl;
    ModelFactory::saveModel(model, p.string());
}

void DirectoryKernel::savePointBuffer(
    const std::string &group,
    const std::string &container,
    const PointBufferPtr &buffer) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    if(!boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directories(p.parent_path());
    }
    ModelPtr model(new Model);
    model->m_pointCloud = buffer;
    // std::cout << timestamp << "Directory Kernel::savePointBuffer(): " << p.string() << std::endl;
    ModelFactory::saveModel(model, p.string());
}

void DirectoryKernel::saveImage(
    const std::string &group,
    const std::string &container,
    const cv::Mat &image) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    if(!boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directories(p.parent_path());
    }
    // std::cout << timestamp << "Directory Kernel::saveImage(): " << p.string() << std::endl;

    cv::imwrite(p.string(), image);
}

void DirectoryKernel::saveMetaYAML(
    const std::string &group,
    const std::string &container,
    const YAML::Node &node) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    if(!boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directories(p.parent_path());
    }
    // std::cout << timestamp << "Directory Kernel::saveMetaYAML(): " << p.string() << std::endl;
    saveMetaInformation(p.string(), node);
}

MeshBufferPtr DirectoryKernel::loadMeshBuffer(
    const std::string &group,
    const std::string container) const
{
    MeshBufferPtr ret;

    boost::filesystem::path p = getAbsolutePath(group, container);
    std::cout << timestamp << "Directory Kernel::loadMeshBuffer(): " << p.string() << std::endl;
    ModelPtr model = ModelFactory::readModel(p.string());
    if (model)
    {
        ret =  model->m_mesh;
    }
    return ret;
}

PointBufferPtr DirectoryKernel::loadPointBuffer(
    const std::string &group,
    const std::string &container) const
{
    PointBufferPtr ret;

    boost::filesystem::path p = getAbsolutePath(group, container);
    ModelPtr model = ModelFactory::readModel(p.string());
    if (model)
    {
        ret = model->m_pointCloud;
    }
    
    return ret;
}

boost::optional<cv::Mat> DirectoryKernel::loadImage(
    const std::string &group,
    const std::string &container) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    boost::optional<cv::Mat> opt;
    // std::cout << timestamp << "Directory Kernel::loadImage: " << p.string() << std::endl;
    if(boost::filesystem::exists(p))
    {
        opt = cv::imread(p.string());

    }
    else
    {
        opt = boost::none;
        std::cout << "[DirectoryKernel - loadImage] Loading image at " << group << " - " << container << " failed!" << std::endl;
    }
    return opt;
}

bool DirectoryKernel::loadMetaYAML(
    const std::string &group,
    const std::string &container,
    YAML::Node& n) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    // std::cout << timestamp << "Directory Kernel::loadMetaYAML: " << p.string() << std::endl;
    YAML::Node node = loadMetaInformation(p.string());
    n = node;

    return n.Type() != YAML::NodeType::Null && n.Type() != YAML::NodeType::Undefined;
}

bool DirectoryKernel::exists(const std::string &group) const
{
    return boost::filesystem::exists(getAbsolutePath(group, ""));
}

bool DirectoryKernel::exists(const std::string &group, const std::string &container) const
{
    // Check if container is not empty to prevent checking
    // against the root itself
    if(container != "")
    {
        return boost::filesystem::exists(getAbsolutePath(group, container));
    }
    return false;
}

void DirectoryKernel::subGroupNames(const std::string &group, std::vector<string> &subGroupNames) const
{
    boost::filesystem::path groupPath(getAbsolutePath(group, ""));
    boost::filesystem::directory_iterator it(groupPath);
    while (it != boost::filesystem::directory_iterator{})
    {
        if (boost::filesystem::is_directory(*it))
        {
            subGroupNames.push_back(it->path().string());
        }
    }
}

void DirectoryKernel::subGroupNames(const std::string &group, const std::regex &filter, std::vector<string> &subGroupNames) const
{
    boost::filesystem::path groupPath(getAbsolutePath(group, ""));
    boost::filesystem::directory_iterator it(groupPath);
    while (it != boost::filesystem::directory_iterator{})
    {
        if (boost::filesystem::is_directory(*it))
        {
            std::string currentName = it->path().string();
            if (std::regex_match(currentName, filter))
            {
                subGroupNames.push_back(currentName);
            }
        }
    }
}

boost::filesystem::path DirectoryKernel::getAbsolutePath(const std::string &group, const std::string &name) const
{
    boost::filesystem::path groupPath(group);
    boost::filesystem::path namePath(name);
    boost::filesystem::path rootPath(m_fileResourceName);
    boost::filesystem::path ret = rootPath / groupPath / namePath;
    return ret;
}

std::vector<std::string> DirectoryKernel::listDatasets(const std::string& group) const
{
    // using namespace boost::filesystem;
    namespace bfs = boost::filesystem;

    std::vector<std::string> ret;
    bfs::path pg = getAbsolutePath(group, "");

    if(bfs::exists(pg))
    {
        bfs::directory_iterator end_itr;
        for(bfs::directory_iterator itr(pg); itr != end_itr; ++itr)
        {
            if (bfs::is_regular_file(itr->path())) 
            {
                if(!isMeta(itr->path().filename().string()))
                {
                    ret.push_back(itr->path().filename().string());
                }
            }
        }
    }       

    return ret;
}

charArr DirectoryKernel::loadCharArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<char>(group, container, dims);   
}

ucharArr DirectoryKernel::loadUCharArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<unsigned char>(group, container, dims);   
}

shortArr DirectoryKernel::loadShortArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<short>(group, container, dims);   
}

ushortArr DirectoryKernel::loadUShortArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<unsigned short>(group, container, dims);   
}

uint16Arr DirectoryKernel::loadUInt16Array(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<uint16_t>(group, container, dims);   
}

intArr DirectoryKernel::loadIntArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<int>(group, container, dims);   
}

uintArr DirectoryKernel::loadUIntArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<unsigned int>(group, container, dims);   
}

lintArr DirectoryKernel::loadLIntArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<long int>(group, container, dims);   
}

ulintArr DirectoryKernel::loadULIntArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<unsigned long int>(group, container, dims);   
}

floatArr DirectoryKernel::loadFloatArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<float>(group, container, dims);   
}

doubleArr DirectoryKernel::loadDoubleArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<double>(group, container, dims);   
}

boolArr DirectoryKernel::loadBoolArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<bool>(group, container, dims);   
}

void DirectoryKernel::saveCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<char>& data) const
{
    saveArray<char>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned char>& data) const
{
    saveArray<unsigned char>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveShortArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<short>& data) const
{
    saveArray<short>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveUShortArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned short>& data) const
{
    saveArray<unsigned short>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveUInt16Array(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<uint16_t>& data) const
{
    saveArray<uint16_t>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<int>& data) const
{
    saveArray<int>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveUIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned int>& data) const
{
    saveArray<unsigned int>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveLIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<long int>& data) const
{
    saveArray<long int>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveULIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned long int>& data) const
{
    saveArray<unsigned long int>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<float>& data) const
{
    saveArray<float>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<double>& data) const
{
    saveArray<double>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveBoolArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<bool>& data) const
{
    saveArray<bool>(groupName, datasetName, dimensions, data);
}

std::unordered_map<std::string, YAML::Node> DirectoryKernel::metas(
        const std::string& group) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    boost::filesystem::path groupPath(getAbsolutePath(group, ""));
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(groupPath), {}))
    {
        if(isMeta(entry.path().string()))
        {
            ret[entry.path().stem().string()] = loadMetaInformation(
                entry.path().string());
        }
    }

    return ret;
}

std::unordered_map<std::string, YAML::Node> DirectoryKernel::metas(
    const std::string& group, const std::string& entity) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    boost::filesystem::path groupPath(getAbsolutePath(group, ""));
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(groupPath), {}))
    {
        if(isMeta(entry.path().string()))
        {
            YAML::Node meta = loadMetaInformation(
                entry.path().string());

            if (meta["entity"])
            {
                if(meta["entity"].as<std::string>() == entity)
                {
                    ret[entry.path().stem().string()] = meta;
                }
            }
        }
    }

    return ret;
}

bool DirectoryKernel::isMeta(const std::string& path) const
{
    return isMetaFile(path);
}

} // namespace lvr2