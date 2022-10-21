#include "lvr2/util/ScanProjectUtils.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/util/ScanSchemaUtils.hpp"
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
    HDF5SchemaPtr schema = hdf5SchemaFromName(schemaName);

    lvr2::scanio::HDF5IO hdf5io(kernel, schema);
    return hdf5io.ScanProjectIO::load();
}

ScanProjectPtr scanProjectFromFile(const std::string& file)
{
    std::cout << timestamp << "[Load Scan Project from File] Creating scan project from single file: " << file << std::endl;
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
        std::cout << timestamp << "[Load Scan Project from file] Unable to open file '" << file << "' for reading-" << std::endl;
    }
    return nullptr;
}

ScanProjectPtr scanProjectFromPLYFiles(const std::string &dir)
{
    std::cout << timestamp << "[Load Scan Project from PLY] Creating scan project from a directory of .ply files..." << std::endl;
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
        std::cout << timestamp << "[Load Scan Project from PLY] Warning: scan project is empty." << std::endl;
        return nullptr;
    }
}

ScanProjectPtr loadScanProject(const std::string& schema, const std::string& source, bool loadData)
{
    boost::filesystem::path sourcePath(source);

    // Check if we try to load from a directory
    if(boost::filesystem::is_directory(sourcePath))
    {
        DirectorySchemaPtr dirSchema = directorySchemaFromName(schema, source);
        DirectoryKernelPtr kernel(new DirectoryKernel(source));

        if(dirSchema && kernel)
        {
            lvr2::scanio::DirectoryIOPtr dirio_in(new lvr2::scanio::DirectoryIO(kernel, dirSchema, loadData));
            return dirio_in->ScanProjectIO::load();
        }
    }
    // Check if we try to load a HDF5 file
    else if(sourcePath.extension() == ".h5")
    {
        HDF5SchemaPtr hdf5Schema = hdf5SchemaFromName(schema);
        HDF5KernelPtr kernel(new HDF5Kernel(source));

        if(hdf5Schema && kernel)
        {
            lvr2::scanio::HDF5IO hdf5io(kernel, hdf5Schema, loadData);
            return hdf5io.ScanProjectIO::load();
        }
    }

    // Loading failed. 
    std::cout << timestamp << "[Load Scan Project] Could not create schema or kernel for loading." << std::endl;
    std::cout << timestamp << "[Load Scan Project] Schema name: " << schema << std::endl;
    std::cout << timestamp << "[Load Scan Project] Source: " << source << std::endl;

    return nullptr;
}

ScanProjectPtr getSubProject(ScanProjectPtr project, std::vector<size_t> positions)
{
    ScanProjectPtr tmp = std::make_shared<ScanProject>();
    
    // Copy meta data
    tmp->boundingBox = project->boundingBox;
    tmp->crs = project->crs;
    tmp->name = project->name;
    tmp->transformation = project->transformation;
    tmp->unit = project->unit;

    // Copy only selected scan position into new project
    for(size_t i : positions)
    {
        if(i < project->positions.size())
        {
            tmp->positions.push_back(project->positions[i]);
        }
        else
        {
            std::cout   << timestamp << "[GetSubProject] Warning: Index" 
                        << i << " out of range, size is " 
                        << project->positions.size() << std::endl;
        }
    }

    // Correct bounding box
    std::cout << timestamp << "[GetSubProject] Correcting bounding box" << std::endl;

    BoundingBox<BaseVector<float>> bb;
    for(ScanPositionPtr p : tmp->positions)
    {
        if(p->boundingBox)
        {
            bb.expand(*(p->boundingBox));
        }
    }

    std::cout << timestamp << "[GetSubProject] New bounding box is: " << bb << std::endl;

    return tmp;
}

void saveScanProject(
    ScanProjectPtr& project,
    const std::vector<size_t>& positions,
    const std::string& schema,
    const std::string& target)
{
    // Create tmp scan project containing only the given positions
    ScanProjectPtr tmp = getSubProject(project, positions);
    
    // Save scan project to destination
    saveScanProject(tmp, schema, target);
}

void saveScanProject(ScanProjectPtr& project, const std::string& schema, const std::string& target)
{
    if(project)
    {
        boost::filesystem::path targetPath(target);
        if(boost::filesystem::is_directory(target))
        {
            DirectorySchemaPtr dirSchema = directorySchemaFromName(schema, target);
            DirectoryKernelPtr kernel(new DirectoryKernel(target));

            if (dirSchema && kernel)
            {
                lvr2::scanio::DirectoryIOPtr dirio (new lvr2::scanio::DirectoryIO(kernel, dirSchema));
                dirio->ScanProjectIO::save(project);
            }

        }
        else if(targetPath.extension() == ".h5")
        {
            HDF5SchemaPtr hdf5Schema = hdf5SchemaFromName(schema);
            HDF5KernelPtr kernel(new HDF5Kernel(target));

            if (hdf5Schema && kernel)
            {
                lvr2::scanio::HDF5IO hdf5io(kernel, hdf5Schema);
                hdf5io.ScanProjectIO::save(project);
            }
        }

        // Saving failed.
        std::cout << timestamp << "[Save Scan Project] Could not create schema or kernel for saving." << std::endl;
        std::cout << timestamp << "[Save Scan Project] Schema name: " << schema << std::endl;
        std::cout << timestamp << "[Save Scan Project] Target: " << target << std::endl;
    }
    else
    {
        std::cout << timestamp << "[Save Scan Project] Cannot save scan project from null pointer" << std::endl;
    }
}

void printScanProjectStructure(const ScanProjectPtr project)
{
    std::cout << project << std::endl;

    for(size_t i = 0; i < project->positions.size(); i++)
    {
        std::cout << timestamp << "Scan Position: " << i << " / " << project->positions.size() << std::endl;
        printScanPositionStructure(project->positions[i]);
    }
}

void printScanPositionStructure(const ScanPositionPtr p) 
{
    std::cout << p;
    for(size_t i = 0; i < p->lidars.size(); i++)
    {
        std::cout << timestamp << "LiDAR " << i << " / " << p->lidars.size() << std::endl;
        printLIDARStructure(p->lidars[i]);
    }
    for(size_t i = 0; i < p->cameras.size(); i++)
    {
        std::cout << timestamp << "Camera " << i << " / " << p->cameras.size() << std::endl;
        printCameraStructure(p->cameras[i]);
    }
    for(size_t i = 0; i < p->hyperspectral_cameras.size(); i++)
    {
        std::cout << timestamp << "Hyperspectral Camera " << i << " / " << p->hyperspectral_cameras.size() << std::endl;
        printHyperspectralCameraStructure(p->hyperspectral_cameras[i]);
    }

}

void printScanStructure(const ScanPtr p) 
{
    std::cout << p;
    // TODO: Implement output for point buffer
}

void printLIDARStructure(const LIDARPtr p) 
{
    std::cout << p;
    for(size_t i = 0; i < p->scans.size(); i++)
    {
        std::cout << timestamp << "Scan " << i << " / " << p->scans.size() << std::endl;
        printScanStructure(p->scans[i]);
    }
}

void printCameraStructure(const CameraPtr p) 
{
    std::cout << p;

    for(size_t i = 0; i < p->groups.size(); i++)
    {
        std::cout << timestamp << "Camera group " << i << " / " << p->groups.size() << std::endl;
        CameraImageGroupPtr g = p->groups[i]; 
        std::cout << timestamp << "Transformation: " << g->transformation << std::endl;
        std::cout << timestamp << "Type: " << g->type << std::endl;
        std::cout << timestamp << "Number of images: " << g->images.size() << std::endl;
    }
}

void printCameraImageGroupStructure(const CameraImageGroupPtr p) 
{
    std::cout << p;
    for(size_t i = 0; i < p->images.size(); i++)
    {
        std::cout << timestamp << "Image " << i << " / " << p->images.size() << std::endl;
        std::cout << p->images[i];
    }
}

void printHyperspectralCameraStructure(const HyperspectralCameraPtr p) 
{
    std::cout << p;
    for(size_t i = 0; i < p->panoramas.size(); i++)
    {
        std::cout << timestamp << "Panorama " << i << " / " << p->panoramas.size() << std::endl;
        printHyperspectralPanoramaStructure(p->panoramas[i]);
    }
}

void printHyperspectralPanoramaStructure(const HyperspectralPanoramaPtr p) 
{
    std::cout << p;
    for(size_t i = 0; i < p->channels.size(); i++)
    {
        std::cout << timestamp << "Channel " << i << " / " << p->channels.size() << std::endl;
        std::cout << p->channels[i];
    }
}

void printCameraImageStructure(const CameraImagePtr p)
{
    std::cout << p;
}

void estimateProjectNormals(ScanProjectPtr p, size_t kn, size_t ki)
{
    for(size_t positionNr = 0; positionNr < p->positions.size(); positionNr++)
    {
        ScanPositionPtr position = p->positions[positionNr];
        if(position)
        {
            for(size_t lidarNr = 0; lidarNr < position->lidars.size(); lidarNr++)
            {
                LIDARPtr lidar = position->lidars[lidarNr];
                if(lidar)
                {
                    for(size_t scanNr = 0; scanNr < lidar->scans.size(); scanNr++)
                    {
                        ScanPtr scan = lidar->scans[scanNr];
                        if(scan)
                        {
                            std::cout << timestamp << "[Project Normal Estimation]: Loading scan " << scanNr 
                                      << " from lidar " << lidarNr << " of scan position " << positionNr << std::endl;
                        
                            scan->load();
                            PointBufferPtr ptBuffer = scan->points;
                            if(ptBuffer)
                            {
                                size_t n = ptBuffer->numPoints();
                                if(n)
                                {
                                    std::cout << timestamp << "[Project Normal Estimation]: Loaded " << n << " points" << std::endl;
                                    std::cout << timestamp << "[Project Normal Estimation]: Building search tree..." << std::endl;

                                    AdaptiveKSearchSurfacePtr<BaseVector<float>> surface(new AdaptiveKSearchSurface<BaseVector<float>>(ptBuffer, "flann", kn, ki));
                                    surface->calculateSurfaceNormals();
                                    surface->interpolateSurfaceNormals();

                                    // Save data back to original project
                                    scan->save();

                                    // Free payload data
                                    scan->release();
                                }
                                else
                                {
                                    std::cout << timestamp << "[Project Normal Estimation]: No points in scan" << std::endl;
                                }
       
                            } 
                            else
                            {
                                std::cout << timestamp << "[Project Normal Estimation]:Unable to load point cloud data." << std::endl;
                            }
                        }
                        else
                        {
                            std::cout << timestamp << "[Project Normal Estimation]: "
                                      << "Unable to load scan " << scanNr << " of "
                                      << "lidar " << lidarNr << std::endl; 
                        }
                    }
                }
                else
                {
                    std::cout << timestamp << "[Project Normal Estimation]: Unable to load lidar " 
                                           << lidarNr << " of scan position " 
                                           << positionNr << std::endl;
                }
            }
        }
        else
        {
            std::cout << timestamp 
                      << "[Project Normal Estimation]: Unable to load scan position " 
                      << positionNr << std::endl;
        }
    }
}

ScanProjectPtr loadScanPositionsExplicitly(
    const std::string& schema,
    const std::string& root,
    const std::vector<size_t>& positions)
{
    
    boost::filesystem::path targetPath(root);
    if (boost::filesystem::is_directory(targetPath))
    {
        DirectorySchemaPtr dirSchema = directorySchemaFromName(schema, root);
        DirectoryKernelPtr kernel(new DirectoryKernel(root));

        if (dirSchema && kernel)
        {
            lvr2::scanio::DirectoryIOPtr dirio(new lvr2::scanio::DirectoryIO(kernel, dirSchema));
            ScanProjectPtr p = dirio->ScanProjectIO::load();
            
            // Clear scan positions
            p->positions.clear();

            // Iterator through given positions indices 
            // and try to load them
            for(size_t i : positions)
            {
                ScanPositionPtr pos = dirio->ScanPositionIO::load(i);
                if(pos)
                {
                    std::cout << timestamp << "[Load Positions Explicitly] : Loading scan position " << i << std::endl;
                    p->positions.push_back(pos);
                }
                else
                {
                    std::cout << timestamp 
                              << "[Load Positions Explicitly] : Position with index " 
                              << i << " cannot be loaded from directory." << std::endl;
                }
            }
            return p;
        }
        else
        {
            if(!kernel)
            {
                std::cout << timestamp 
                          << "[Load Positions Explicitly] : Could not create directory kernel from root " 
                          << root << std::endl; 
            }
            if(!dirSchema)
            {
                std::cout << timestamp 
                          << "[Load Positions Explicitly] : Could not create directory schema from name " 
                          << schema << std::endl; 
            }
        }
    }
    else if (targetPath.extension() == ".h5")
    {
        HDF5SchemaPtr hdf5Schema = hdf5SchemaFromName(schema);
        HDF5KernelPtr kernel(new HDF5Kernel(root));

        if (hdf5Schema && kernel)
        {
            lvr2::scanio::HDF5IOPtr hdf5io(new lvr2::scanio::HDF5IO(kernel, hdf5Schema));

            ScanProjectPtr p = hdf5io->ScanProjectIO::load();

            // Clear scan positions
            p->positions.clear();

            // Iterator through given positions indices
            // and try to load them
            for (size_t i : positions)
            {
                ScanPositionPtr pos = hdf5io->ScanPositionIO::load(i);
                if (pos)
                {
                    std::cout << timestamp << "[Load Positions Explicitly] : Loading scan position " << i << std::endl;
                    p->positions.push_back(pos);
                }
                else
                {
                    std::cout << timestamp
                              << "[Load Positions Explicitly] : Position with index "
                              << i << " cannot be loaded from HDF5 file." << std::endl;
                }
            }
            return p;
        }
        else
        {
            if(!kernel)
            {
                std::cout << timestamp 
                          << "[Load Positions Explicitly] : Could not create HDF5 kernel from root " 
                          << root << std::endl; 
            }
            if(!hdf5Schema)
            {
                std::cout << timestamp 
                          << "[Load Positions Explicitly] : Could not create HDF5 schema from name " 
                          << schema << std::endl; 
            }
        }
    }
    std::cout << timestamp << "[Load Positions Explicitly] : Could not load any data." << std::endl;
    return nullptr;
}

} // namespace lvr2 
     