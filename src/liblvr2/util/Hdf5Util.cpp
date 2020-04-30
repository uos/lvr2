#include "lvr2/io/hdf5/Hdf5Util.hpp"

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

void writeBaseStructure(std::shared_ptr<HighFive::File> hdf5_file)
{
    int version = 1;
    hdf5_file->createDataSet<int>("version", HighFive::DataSpace::From(version)).write(version);
    HighFive::Group raw_data_group = hdf5_file->createGroup("raw");

    // Create string with current time
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    std::string time(ctime(&t_now));

    // Add current time to raw data group
    raw_data_group.createDataSet<std::string>("created", HighFive::DataSpace::From(time))
        .write(time);
    raw_data_group.createDataSet<std::string>("changed", HighFive::DataSpace::From(time))
        .write(time);

    // Create empty reference frame
    std::vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    raw_data_group.createDataSet<float>("position", HighFive::DataSpace::From(frame)).write(frame);
}

HighFive::Group getGroup(std::shared_ptr<HighFive::File> hdf5_file,
                                const std::string& groupName,
                                bool create)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {
        cur_grp = hdf5_file->getGroup("/");

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
        std::cout << "Error in getGroup (with group name '" << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return cur_grp;
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

bool exist(std::shared_ptr<HighFive::File> hdf5_file, const std::string& groupName)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {
        cur_grp = hdf5_file->getGroup("/");

        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (cur_grp.exist(groupNames[i]))
            {
                if (i < groupNames.size() - 1)
                {
                    cur_grp = cur_grp.getGroup(groupNames[i]);
                }
            }
            else
            {
                return false;
            }
        }
    }
    catch (HighFive::Exception& e)
    {
        std::cout << "Error in exist (with group name '" << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return true;
}

bool exist(HighFive::Group& group, const std::string& groupName)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group cur_grp = group;

    try
    {
        for (size_t i = 0; i < groupNames.size(); i++)
        {
            if (cur_grp.exist(groupNames[i]))
            {
                if (i < groupNames.size() - 1)
                {
                    cur_grp = cur_grp.getGroup(groupNames[i]);
                }
            }
            else
            {
                return false;
            }
        }
    }
    catch (HighFive::Exception& e)
    {
        std::cout << "Error in exist (with group name '" << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return true;
}

std::shared_ptr<HighFive::File> open(const std::string& filename)
{
    std::shared_ptr<HighFive::File> hdf5_file;
    boost::filesystem::path path(filename);

    if (!boost::filesystem::exists(path))
    {
        hdf5_file.reset(
            new HighFive::File(filename, HighFive::File::ReadWrite | HighFive::File::Create));
        hdf5util::writeBaseStructure(hdf5_file);
    }
    else
    {
        hdf5_file.reset(new HighFive::File(filename, HighFive::File::ReadWrite));
    }

    return hdf5_file;
}

} // namespace hdf5util

} // namespace lvr2