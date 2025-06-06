#include "lvr2/util/Hdf5Util.hpp"
#include "lvr2/types/Channel.hpp"

namespace lvr2
{

namespace hdf5util
{

std::vector<std::string> splitGroupNames(const std::string& groupName)
{
    std::vector<std::string> ret;

    std::string remainder = groupName;
    size_t delimiter_pos = 0;

    while ((delimiter_pos = remainder.find('/', delimiter_pos)) != std::string::npos)
    {
        if (delimiter_pos > 0)
        {
            ret.push_back(remainder.substr(0, delimiter_pos));
        }

        remainder = remainder.substr(delimiter_pos + 1);

        delimiter_pos = 0;
    }

    if (remainder.size() > 0)
    {
        ret.push_back(remainder);
    }

    return ret;
}

std::pair<std::string, std::string> validateGroupDataset(
    const std::string& groupName, 
    const std::string& datasetName)
{
    // std::cout << "Validate " << groupName << ", " << datasetName << std::endl;

    std::vector<std::string> groupNameSplit = splitGroupNames(groupName);
    std::vector<std::string> datasetSplit = splitGroupNames(datasetName);
    
    std::vector<std::string> totalPath;

    for(auto groupName : groupNameSplit)
    {
        totalPath.push_back(groupName);
    }

    for(auto containerName : datasetSplit)
    {
        totalPath.push_back(containerName);
    }

    std::string groupPath = "";
    if(totalPath.size() > 1)
    {
        groupPath = totalPath[0];
        for(size_t i=1; i<totalPath.size()-1; i++)
        {
            groupPath += "/" + totalPath[i];
        }
    }
    
    std::string container = totalPath.back();

    // std::cout << "To " << groupPath << ", " << container << std::endl;

    return {groupPath, container};
}

void writeBaseStructure(std::shared_ptr<HighFive::File> hdf5_file)
{
    int version = 1;
    hdf5_file->createDataSet<int>("version", HighFive::DataSpace::From(version)).write(version);
    // HighFive::Group raw_data_group = hdf5_file->createGroup("raw");

    // Create string with current time
    // std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    // std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    // std::string time(ctime(&t_now));

    // // Add current time to raw data group
    // raw_data_group.createDataSet<std::string>("created", HighFive::DataSpace::From(time))
    //     .write(time);
    // raw_data_group.createDataSet<std::string>("changed", HighFive::DataSpace::From(time))
    //     .write(time);

    // // Create empty reference frame
    // std::vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    // raw_data_group.createDataSet<float>("position", HighFive::DataSpace::From(frame)).write(frame);
}

HighFive::Group getGroup(std::shared_ptr<HighFive::File> hdf5_file,
                                const std::string& groupName,
                                bool create)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);


    try
    {
        HighFive::Group cur_grp = hdf5_file->getGroup("/");

        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (cur_grp.exist(groupNames[i]))
            {
                cur_grp = cur_grp.getGroup(groupNames[i]);
            }
            else if (create)
            {
                cur_grp = cur_grp.createGroup(groupNames[i]);
            }
            else
            {
                // Throw exception because a group we searched
                // for doesn't exist and create flag was false
                throw std::runtime_error("HDF5IO - getGroup(): Groupname '" + groupNames[i] +
                                         "' doesn't exist and create flag is false");
            }
        }
        return cur_grp;
    }
    catch (HighFive::Exception& e)
    {
        std::cout << "Error in getGroup (with group name '" << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;

    }


}

HighFive::Group getGroup(HighFive::Group& g, const std::string& groupName, bool create)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group cur_grp = g;

    try
    {

        for (size_t i = 0; i < groupNames.size(); i++)
        {

            if (cur_grp.exist(groupNames[i]))
            {
                cur_grp = cur_grp.getGroup(groupNames[i]);
            }
            else if (create)
            {
                cur_grp = cur_grp.createGroup(groupNames[i]);
            }
            else
            {
                // Throw exception because a group we searched
                // for doesn't exist and create flag was false
                throw std::runtime_error("HDF5IO - getGroup(): Groupname '" + groupNames[i] +
                                         "' doesn't exist and create flag is false");
            }
        }
    }
    catch (HighFive::Exception& e)
    {
        std::cout << timestamp << "Error in HDF5Util::getGroup '" << groupName << "': " << std::endl;
        std::cout << timestamp << e.what() << std::endl;
        throw e;
    }

    return cur_grp;
}

HighFiveSplit split(HighFive::Group g)
{
    HighFiveSplit ret;

    for(std::string key : g.listObjectNames())
    {
        HighFive::ObjectType obj_type = g.getObjectType(key);
        if(obj_type == HighFive::ObjectType::Group)
        {
            ret.groups.push_back(key);
        } 
        else if(obj_type == HighFive::ObjectType::Dataset) 
        {
            ret.datasets.push_back(key);
        }
    }

    return ret;
}

bool exist(const std::shared_ptr<HighFive::File>& hdf5_file, const std::string& groupName)
{
    HighFive::Group cur_grp = hdf5_file->getGroup("/");
    bool ret = exist(cur_grp, groupName);
    return ret;
}

bool exist(const HighFive::Group& group, const std::string& groupName)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group group_iter = group;

    try
    {
        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (group_iter.exist(groupNames[i]))
            {
                if (i+1 < groupNames.size())
                {
                    group_iter = group_iter.getGroup(groupNames[i]);
                }
            }
            else
            {
                return false;
            }
        }

        return true;
    }
    catch (HighFive::Exception& e)
    {
        std::cout << "Error in exist (with group name '" << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return false;
}

std::shared_ptr<HighFive::File> open(const boost::filesystem::path& filename, unsigned int flag)
{
    std::shared_ptr<HighFive::File> hdf5_file;

    if (!boost::filesystem::exists(filename))
    {
        hdf5_file.reset(
            new HighFive::File(filename.string(), flag | HighFive::File::Create));
        hdf5util::writeBaseStructure(hdf5_file);
    }
    else
    {
        hdf5_file.reset(new HighFive::File(filename.string(), flag));
    }

    return hdf5_file;
}

boost::optional<std::string> highFiveTypeToLvr(std::string h5type)
{
    boost::optional<std::string> ret;

    if(HighFive::AtomicType<char>().string() == h5type)
    {
        ret = Channel<char>::typeName();
    } else
    if(HighFive::AtomicType<unsigned char>().string() == h5type)
    {
        ret = Channel<unsigned char>::typeName();
    } else
    if(HighFive::AtomicType<short>().string() == h5type)
    {
        ret = Channel<short>::typeName();
    } else
    if(HighFive::AtomicType<unsigned short>().string() == h5type)
    {
        ret = Channel<unsigned short>::typeName();
    } else
    if(HighFive::AtomicType<int>().string() == h5type)
    {
        ret = Channel<int>::typeName();
    } else
    if(HighFive::AtomicType<long int>().string() == h5type)
    {
        ret = Channel<long int>::typeName();
    } else
    if(HighFive::AtomicType<unsigned int>().string() == h5type)
    {
        ret = Channel<unsigned int>::typeName();
    } else
    if(HighFive::AtomicType<size_t>().string() == h5type)
    {
        ret = Channel<size_t>::typeName();
    } else
    if(HighFive::AtomicType<float>().string() == h5type)
    {
        ret = Channel<float>::typeName();
    } else
    if(HighFive::AtomicType<double>().string() == h5type)
    {
        ret = Channel<double>::typeName();
    } else
    if(HighFive::AtomicType<bool>().string() == h5type)
    {
        ret = Channel<bool>::typeName();
    }

    return ret;
}

} // namespace hdf5util

} // namespace lvr2
