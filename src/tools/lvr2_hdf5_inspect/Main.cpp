#include <iostream>

#include "Options.hpp"

#include <lvr2/io/scanio/HDF5IO.hpp>
#include <lvr2/util/Hdf5Util.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>


namespace bfs = boost::filesystem;

void printHDF5(const HighFive::Group& g, bool recursive=true, int depth=0)
{
    for(auto groupName : g.listObjectNames())
    {
        HighFive::ObjectType h5type = g.getObjectType(groupName);

        if(h5type == HighFive::ObjectType::Group)
        {
            std::cout << std::string(depth, ' ') << groupName << " (Group)" << std::endl;
            HighFive::Group nextGroup = g.getGroup(groupName);
            if(recursive)
            {
                printHDF5(nextGroup, recursive, depth+1);
            }
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

struct Command {
    std::string command;
    std::vector<std::string> arguments;
};

Command userInput()
{
    std::string line;
    std::getline(std::cin, line);

    std::vector<std::string> strs;
    boost::split(strs, line, boost::is_any_of(" "));

    std::vector<std::string> filtered;
    for(auto str : strs)
    {
        if(str != "")
        {
            filtered.push_back(str);
        }
    }

    Command command;

    if(filtered.size() > 0)
    {
        command.command = filtered[0];

        for(size_t i=1; i<filtered.size(); i++)
        {
            command.arguments.push_back(filtered[i]);
        }
    }

    return command;
}

void printHelp()
{
    std::cout << "help: \t\t prints this message" << std::endl;
    std::cout << "ls: \t\t lists groups and datasets" << std::endl;
    std::cout << "tree: \t\t lists groups and datasets recursively" << std::endl;
    std::cout << "cd: \t\t change the group" << std::endl;
    std::cout << "q,quit: \t close hdf5 inspector" << std::endl;
}

int main(int argc, char** argv)
{
    hdf5_inspect::Options opt(argc, argv);

    std::string filename = opt.inputFile();
    bfs::path input_file(filename);

    if(input_file.extension() != ".h5")
    {
        std::cout << "Specified File has not .h5 extension" << std::endl;
        return 0;
    }

    auto h5file = lvr2::hdf5util::open(input_file.string());
    
    bfs::path root_path = "";
    HighFive::Group root_group = h5file->getGroup("/");


    while(true)
    {
        std::cout << root_path.string() << "$ " << std::flush;
        Command c = userInput();

        if(c.command == "ls")
        {
            if(c.arguments.size() == 0)
            {
                printHDF5(root_group, false);
            } else {
                // TODO
            }
            // printHDF5(root_group, )
        } else if(c.command == "tree")
        {
            printHDF5(root_group, true);
        } else if(c.command == "cd")
        {
            if(c.arguments.size() == 0)
            {
                // go back to root
                root_path = "";
                root_group = h5file->getGroup("/");
            } else {
                std::string group_name = c.arguments[0];

                if(group_name == "..")
                {
                    // go one group to the top
                    if(root_path.has_parent_path())
                    {
                        root_path = root_path.parent_path();
                        root_group = h5file->getGroup(root_path.string());
                    }
                    
                } else if(group_name == ".") {
                    // do nothing
                } else {
                    if(!root_group.exist(group_name))
                    {
                        std::cout << group_name << " not found" << std::endl;
                    } else {
                        root_path /= group_name;
                        root_group = root_group.getGroup(c.arguments[0]);
                    }
                }
            }
        } else if(c.command == "help") {
            printHelp();
        } else if(c.command == "q" || c.command == "quit") {
            break;
        } else {
            std::cout << "Unknown command '" << c.command << "'. Enter 'help' to list the available commands." << std::endl;
        }
    }

    return 0;
}