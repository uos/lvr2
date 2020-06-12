#include "lvr2/io/descriptions/DirectoryKernel.hpp"

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
    std::cout << timestamp << "Directory Kernel::saveMeshBuffer(): " << p.string() << std::endl;
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
    std::cout << timestamp << "Directory Kernel::savePointBuffer(): " << p.string() << std::endl;
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
    std::cout << timestamp << "Directory Kernel::saveImage(): " << p.string() << std::endl;

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
    std::cout << timestamp << "Directory Kernel::saveMetaYAML(): " << p.string() << std::endl;
    saveMetaInformation(p.string(), node);
}



MeshBufferPtr DirectoryKernel::loadMeshBuffer(
    const std::string &group,
    const std::string container) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    std::cout << timestamp << "Directory Kernel::loadMeshBuffer(): " << p.string() << std::endl;
    ModelPtr model = ModelFactory::readModel(p.string());
    if (model)
    {
        return model->m_mesh;
    }
}

PointBufferPtr DirectoryKernel::loadPointBuffer(
    const std::string &group,
    const std::string &container) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    std::cout << timestamp << "Directory Kernel::loadPointBuffer(): " << p.string() << std::endl;
    ModelPtr model = ModelFactory::readModel(p.string());
    if (model)
    {
        std::cout << model->m_pointCloud << std::endl;
        std::cout << model->m_pointCloud->numPoints() << std::endl;
        return model->m_pointCloud;
    }
}

boost::optional<cv::Mat> DirectoryKernel::loadImage(
    const std::string &group,
    const std::string &container) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    boost::optional<cv::Mat> opt;
    std::cout << timestamp << "Directory Kernel::loadImage: " << p.string() << std::endl;
    if(boost::filesystem::exists(p))
    {
        opt = cv::imread(p.string());

    }
    else
    {
        opt = boost::none;
    }
    return opt;
}

void DirectoryKernel::loadMetaYAML(
    const std::string &group,
    const std::string &container,
    YAML::Node& n) const
{
    boost::filesystem::path p = getAbsolutePath(group, container);
    std::cout << timestamp << "Directory Kernel::loadMetaYAML: " << p.string() << std::endl;
    YAML::Node node = loadMetaInformation(p.string());
    n = node;
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



ucharArr DirectoryKernel::loadUCharArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<unsigned char>(group, container, dims);   
}

floatArr DirectoryKernel::loadFloatArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<float>(group, container, dims);   
}

doubleArr DirectoryKernel::loadDoubleArray(const std::string& group, const std::string& container, std::vector<size_t>& dims) const
{
    return loadArray<double>(group, container, dims);   
}

void DirectoryKernel::saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<float>& data) const
{
    saveArray<float>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<double>& data) const
{
    saveArray<double>(groupName, datasetName, dimensions, data);
}

void DirectoryKernel::saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t>& dimensions, const boost::shared_array<unsigned char>& data) const
{
    saveArray<unsigned char>(groupName, datasetName, dimensions, data);
}

} // namespace lvr2