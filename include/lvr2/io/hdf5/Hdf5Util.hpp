#ifndef HDF5UTIL_HPP
#define HDF5UTIL_HPP

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include <H5Tpublic.h>
#include <hdf5_hl.h>

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include <boost/filesystem.hpp>

#include <lvr2/geometry/Matrix4.hpp>

namespace lvr2 {

namespace hdf5util {

static std::vector<std::string> splitGroupNames(const std::string &groupName)
{
    std::vector<std::string> ret;

    std::string remainder = groupName;
    size_t delimiter_pos = 0;

    while ( (delimiter_pos = remainder.find('/', delimiter_pos)) != std::string::npos)
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

static void writeBaseStructure(std::shared_ptr<HighFive::File> hdf5_file)
{
    int version = 1;
    hdf5_file->createDataSet<int>("version", HighFive::DataSpace::From(version)).write(version);
    HighFive::Group raw_data_group = hdf5_file->createGroup("raw");

    // Create string with current time
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t t_now= std::chrono::system_clock::to_time_t(now);
    std::string time(ctime(&t_now));

    // Add current time to raw data group
    raw_data_group.createDataSet<std::string>("created", HighFive::DataSpace::From(time)).write(time);
    raw_data_group.createDataSet<std::string>("changed", HighFive::DataSpace::From(time)).write(time);

    // Create empty reference frame
    std::vector<float> frame = Matrix4<BaseVector<float>>().getVector();
    raw_data_group.createDataSet<float>("position", HighFive::DataSpace::From(frame)).write(frame);
}


static HighFive::Group getGroup(
    std::shared_ptr<HighFive::File> hdf5_file,
    const std::string& groupName,
    bool create = true)
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
                throw std::runtime_error("HDF5IO - getGroup(): Groupname '"
                    + groupNames[i] + "' doesn't exist and create flag is false");
            }
        }
    }
    catch(HighFive::Exception& e)
    {
        std::cout << "Error in getGroup (with group name '"
                  << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return cur_grp;
}

static HighFive::Group getGroup(
    HighFive::Group& g,
    const std::string& groupName,
    bool create = true)
{
    std::vector<std::string> groupNames = hdf5util::splitGroupNames(groupName);
    HighFive::Group cur_grp;

    try
    {

        for (size_t i = 0; i < groupNames.size(); i++)
        {

            if (g.exist(groupNames[i]))
            {
                cur_grp = g.getGroup(groupNames[i]);
            }
            else if (create)
            {
                cur_grp = g.createGroup(groupNames[i]);
            }
            else
            {
                // Throw exception because a group we searched
                // for doesn't exist and create flag was false
                throw std::runtime_error("HDF5IO - getGroup(): Groupname '"
                    + groupNames[i] + "' doesn't exist and create flag is false");
            }
        }
    }
    catch(HighFive::Exception& e)
    {
        std::cout << "Error in getGroup (with group name '"
                  << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return cur_grp;
}

static bool exist(
    std::shared_ptr<HighFive::File> hdf5_file, 
    const std::string &groupName)
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
                if (i < groupNames.size() -1)
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
        std::cout << "Error in exist (with group name '"
                  << groupName << "': " << std::endl;
        std::cout << e.what() << std::endl;
        throw e;
    }

    return true;
}

static std::shared_ptr<HighFive::File> open(std::string filename)
{
    std::shared_ptr<HighFive::File> hdf5_file;
    boost::filesystem::path path(filename);

    if(!boost::filesystem::exists(path))
    {
        hdf5_file.reset(new HighFive::File(filename, HighFive::File::ReadWrite | HighFive::File::Create));
        hdf5util::writeBaseStructure(hdf5_file);
    } else {
        hdf5_file.reset(new HighFive::File(filename, HighFive::File::ReadWrite));
    }

    return hdf5_file;
}

template<typename T>
std::unique_ptr<HighFive::DataSet> createDataset(
    HighFive::Group& g,
    std::string datasetName,
    const HighFive::DataSpace& dataSpace,
    const HighFive::DataSetCreateProps& properties)
{
    std::unique_ptr<HighFive::DataSet> dataset;

    if(g.exist(datasetName))
    {
        try {
            dataset = std::make_unique<HighFive::DataSet>(
                g.getDataSet(datasetName)
            );
        } catch(HighFive::DataSetException& ex) {
            std::cout << "[Hdf5Util - createDataset] " << datasetName << " is not a dataset" << std::endl;
        }

        // check existing dimensions
        const std::vector<size_t> dims_old = dataset->getSpace().getDimensions();
        const std::vector<size_t> dims_new = dataSpace.getDimensions();
        
        
        if(dataset->getDataType() != HighFive::AtomicType<T>() )
        {
            // different datatype -> delete
            int result = H5Ldelete(g.getId(), datasetName.data(), H5P_DEFAULT);
            dataset = std::make_unique<HighFive::DataSet>(
                g.createDataSet<T>(datasetName, dataSpace, properties)
            );
        } else if(dims_old[0] != dims_new[0] || dims_old[1] != dims_new[1]) {
            // same datatype but different size -> resize
            
            std::cout << "[Hdf5Util - createDataset] WARNING: size has changed. resizing dataset " << std::endl;
            
            // 
            try {
                dataset->resize(dims_new);
            } catch(HighFive::DataSetException& ex) {
                std::cout << "[Hdf5Util - createDataset] WARNING: could not resize. Generating new space..." << std::endl;
                int result = H5Ldelete(g.getId(), datasetName.data(), H5P_DEFAULT);

                dataset = std::make_unique<HighFive::DataSet>(
                    g.createDataSet<T>(datasetName, dataSpace, properties)
                );
            }
        }
    } else {
        dataset = std::make_unique<HighFive::DataSet>(
            g.createDataSet<T>(datasetName, dataSpace, properties)
        );
    }

    return std::move(dataset);
}

template<typename T>
void setAttribute(HighFive::Group& g,
    const std::string& attr_name,
    T& data)
{
    bool use_existing_attribute = false;
    bool overwrite = false;

    if(g.hasAttribute(attr_name))
    {
        // check if attribute is the same
        HighFive::Attribute attr = g.getAttribute(attr_name);
        if(attr.getDataType() == HighFive::AtomicType<T>() )
        {
            T value;
            attr.read(value);

            use_existing_attribute = true;
            if(value != data)
            {
                overwrite = true;
            }
        }
    }

    if(!use_existing_attribute) {
        g.createAttribute<T>(attr_name, data);
    } else if(overwrite) {
        g.getAttribute(attr_name).write<T>(data);
    }
}

template<typename T>
bool checkAttribute(HighFive::Group& g,
    const std::string& attr_name,
    T& data)
{
    // check if attribute exists
    if(!g.hasAttribute(attr_name))
    {
        return false;
    }

    // check if attribute type is the same
    HighFive::Attribute attr = g.getAttribute(attr_name);
    if(attr.getDataType() != HighFive::AtomicType<T>() )
    {
        return false;
    }

    // check if attribute value is the same
    T value;
    attr.read(value);
    if(value != data)
    {
        return false;
    }

    return true;
}

} // namespace hdf5util

} // namespace lvr2

#endif