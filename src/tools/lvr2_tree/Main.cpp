#include <iostream>

#include <lvr2/io/descriptions/HDF5IO.hpp>
#include <lvr2/io/hdf5/Hdf5Util.hpp>
#include <boost/filesystem.hpp>

namespace bfs = boost::filesystem;

void printHDF5(const HighFive::DataSet& g, int depth=0)
{
    
}

void printHDF5(const HighFive::Group& g, int depth=0)
{
    for(auto groupName : g.listObjectNames())
    {
        HighFive::ObjectType h5type = g.getObjectType(groupName);

        if(h5type == HighFive::ObjectType::Group)
        {
            std::cout << std::string(depth, ' ') << groupName << " (Group)" << std::endl;
            HighFive::Group nextGroup = g.getGroup(groupName);
            printHDF5(nextGroup, depth+1);
        } else if(h5type == HighFive::ObjectType::Dataset) {
            HighFive::DataSet ds = g.getDataSet(groupName);

            std::cout << std::string(depth, ' ') << groupName << " (Dataset, ";
            HighFive::DataType dtype = ds.getDataType();
            auto lvrTypeName = lvr2::hdf5util::highFiveTypeToLvr(dtype.string());
            
            if(lvrTypeName)
            {
                std::cout << "type: " << *lvrTypeName;
            } else {
                std::cout << "type: unknown";
            }
            
            
            std::vector<size_t> dims = ds.getSpace().getDimensions();
            std::cout << ", dims: ";
            for(auto dim : dims)
            {
                std::cout << dim << " ";
            }
            std::cout << ")" << std::endl;
        } else {
            std::cout << std::string(depth, ' ') << groupName << " (Unknown)" << std::endl;
        }
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "lvr2_tree [.h5/Directory]" << std::endl;
        return 0;
    }

    bfs::path input_file(argv[1]);
    std::cout << "Open " << input_file << std::endl;

    if(input_file.extension() == ".h5")
    {
        std::cout << "Parsing hdf5" << std::endl;
        auto h5file = lvr2::hdf5util::open(input_file.string());
        
        HighFive::Group root_group = h5file->getGroup("/");
        
        printHDF5(root_group);

    } else if(input_file.extension() == "") {
        std::cout << "Parsing directory" << std::endl;
        std::cout << "TODO: implement" << std::endl;
    } else {
        std::cout << "Extension " << input_file.extension() << " unknown." << std::endl;
    }

    return 0;
}