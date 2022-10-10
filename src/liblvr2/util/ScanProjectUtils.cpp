#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/util/ScanProjectUtils.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/scanio/HDF5IO.hpp"
#include "lvr2/io/scanio/DirectoryIO.hpp"
#include "lvr2/io/kernels/DirectoryKernel.hpp"
#include "lvr2/io/kernels/HDF5Kernel.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

std::pair<ScanPtr, Transformd> scanFromProject(ScanProjectPtr project, size_t scanPositionNo, size_t lidarNo, size_t scanNo)
{
    Transformd transform = Transformd::Identity();

    if(project && scanPositionNo < project->positions.size())
    {
        transform = transform * project->transformation;
        ScanPositionPtr pos = project->positions[scanPositionNo];
        if(pos && lidarNo < pos->lidars.size())
        {
            transform = transform * pos->transformation;
            LIDARPtr lidar =  pos->lidars[lidarNo];
            if(lidar && scanNo < lidar->scans.size())
            {
                transform = transform * lidar->transformation;
                ScanPtr scan = lidar->scans[scanNo];
                return std::make_pair(scan, transform);
            }
        }
    }

    // Something went wrong during access...
    return std::make_pair(nullptr, Transformd::Identity());
}

ScanProjectPtr scanProjectFromHDF5(std::string file, const std::string& schemaName)
{
    HDF5KernelPtr kernel(new HDF5Kernel(file));
    HDF5SchemaPtr schema = hdf5SchemafromName(schemaName);

    lvr2::scanio::HDF5IO hdf5io(kernel, schema);
    return hdf5io.ScanProjectIO::load();
}

ScanProjectPtr scanProjectFromFile(const std::string& file)
{
    std::cout << timestamp << "Creating scan project from single file: " << file << std::endl;
    ScanProjectPtr project(new ScanProject);
    ModelPtr model = ModelFactory::readModel(file);

    if(model)
    {

        // Create new scan object and mark scan data as
        // loaded
        ScanPtr scan(new Scan);
        scan->points = model->m_pointCloud;

        // Create new lidar object
        LIDARPtr lidar(new LIDAR);

        // Create new scan position
        ScanPositionPtr scanPosPtr = ScanPositionPtr(new ScanPosition());

        // Buildup scan project structure
        project->positions.push_back(scanPosPtr);
        project->positions[0]->lidars.push_back(lidar);
        project->positions[0]->lidars[0]->scans.push_back(scan);

        return project;
    }
    else
    {
        std::cout << timestamp << "Unable to open file '" << file << "' for reading-" << std::endl;
    }
    return nullptr;
}

ScanProjectPtr scanProjectFromPLYFiles(const std::string &dir)
{
    std::cout << timestamp << "Creating scan project from a directory of .ply files..." << std::endl;
    ScanProjectPtr scanProject(new ScanProject);
    boost::filesystem::directory_iterator it{dir};
    while (it != boost::filesystem::directory_iterator{})
    {
        string ext = it->path().extension().string();
        if (ext == ".ply")
        {
            ModelPtr model = ModelFactory::readModel(it->path().string());

            // Create new Scan
            ScanPtr scan(new Scan);
            scan->points = model->m_pointCloud;

            // Wrap scan into lidar object
            LIDARPtr lidar(new LIDAR);
            lidar->scans.push_back(scan);

            // Put lidar into new scan position
            ScanPositionPtr position(new ScanPosition);
            position->lidars.push_back(lidar);

            // Add new scan position to scan project
            scanProject->positions.push_back(position);
        }
        it++;
    }
    if(scanProject->positions.size())
    {
        return scanProject;
    }
    else
    {
        std::cout << timestamp << "Warning: scan project is empty." << std::endl;
        return nullptr;
    }
}

ScanProjectPtr loadScanProject(const std::string& schema, const std::string source)
{
    boost::filesystem::path sourcePath(source);

    // Check if we try to load from a directory
    if(boost::filesystem::is_directory(source) else
        {
            std::cout << timestamp << "Could not create schema or kernel." << std::endl;
            std::cout << timestamp << "Schema name: " << schema << std::endl;
            std::cout << timestamp << "Source: " << source << std::endl;
        }yKernel(source));

        if(dirSchema && kernel)
        {
            DirectoryIO dirio_in(kernel, dirSchema);
            return dirio_in.ScanProjectIO::load();
        }
    }
    // Check if we try to load a HDF5 file
    else if(sourcePath.extension() == ".h5")
    {
        HDF5SchemaPtr hdf5Schema = hdf5SchemaFromName(schema);
        HDF5KernelPtr kernel(new HDF5Kernel(source));

        if(hdf5Schema && kernel)
        {
            HDF5IO hdf5io(kernel, hdf5Schema);
            return hdf5io.ScanProjectIO::load();
        }
    }

    // Loading failed. 
    std::cout << timestamp << "Could not create schema or kernel." << std::endl;
    std::cout << timestamp << "Schema name: " << schema << std::endl;
    std::cout << timestamp << "Source: " << source << std::endl;

    return nullptr;
}

} // namespace lvr2 else
     