#include "lvr2/util/ScanProjectUtils.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Factories.hpp"
#include "lvr2/util/ScanSchemaUtils.hpp"
#include "lvr2/util/TransformUtils.hpp"
#include "lvr2/util/Logging.hpp"
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
    lvr2::logout::get() << lvr2::info << "[Load Scan Project from File] Creating scan project from single file: " << file << lvr2::endl;
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
        lvr2::logout::get() << lvr2::info << "[Load Scan Project from file] Unable to open file '" << file << "' for reading-" << lvr2::endl;
    }
    return nullptr;
}

ScanProjectPtr scanProjectFromPLYFiles(const std::string &dir)
{
    lvr2::logout::get() << lvr2::info <<  "[Load Scan Project from PLY] Creating scan project from a directory of .ply files..." << lvr2::endl;
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
        lvr2::logout::get() << lvr2::warning << "[Load Scan Project from PLY] Warning: scan project is empty." << lvr2::endl;
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
            lvr2::scanio::HDF5IOPtr hdf5io(new lvr2::scanio::HDF5IO(kernel, hdf5Schema, loadData));
            return hdf5io->ScanProjectIO::load();
        }
    }

    // Loading failed. 
    lvr2::logout::get() << lvr2::error << "[Load Scan Project] Could not create schema or kernel for loading." << lvr2::endl;
    lvr2::logout::get() << lvr2::error << "[Load Scan Project] Schema name: " << schema << lvr2::endl;
    lvr2::logout::get() << lvr2::error << "[Load Scan Project] Source: " << source << lvr2::endl;

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
            lvr2::logout::get() << lvr2::warning << "[GetSubProject] Warning: Index" 
                        << i << " out of range, size is " 
                        << project->positions.size() << lvr2::endl;
        }
    }

    // Correct bounding box
    lvr2::logout::get() << lvr2::debug << "[GetSubProject] Correcting bounding box" << lvr2::endl;

    BoundingBox<BaseVector<float>> bb;
    for(ScanPositionPtr p : tmp->positions)
    {
        if(p->boundingBox)
        {
            bb.expand(*(p->boundingBox));
        }
    }

    lvr2::logout::get() << lvr2::info << "[GetSubProject] New bounding box is: " << bb << lvr2::endl;

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
        else
        {
            // Saving failed.
            lvr2::logout::get() << lvr2::error << "[Save Scan Project] Could not create schema or kernel for saving." << lvr2::endl;
            lvr2::logout::get() << lvr2::error << "[Save Scan Project] Schema name: " << schema << lvr2::endl;
            lvr2::logout::get() << lvr2::error << "[Save Scan Project] Target: " << target << lvr2::endl;
        }
    }
    else
    {
        lvr2::logout::get() << lvr2::error << "[Save Scan Project] Cannot save scan project from null pointer" << lvr2::endl;
    }
}

void printScanProjectStructure(const ScanProjectPtr project)
{
    lvr2::logout::get() << lvr2::info << project << lvr2::endl;

    for(size_t i = 0; i < project->positions.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[Scan Project] Position" << i << " / " << project->positions.size() << lvr2::endl;
        printScanPositionStructure(project->positions[i]);
    }
}

void printScanPositionStructure(const ScanPositionPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    for(size_t i = 0; i < p->lidars.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[Scan Position] LiDAR " << i << " / " << p->lidars.size() << lvr2::endl;
        printLIDARStructure(p->lidars[i]);
    }
    for(size_t i = 0; i < p->cameras.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[Scan Position] Camera " << i << " / " << p->cameras.size() << lvr2::endl;
        printCameraStructure(p->cameras[i]);
    }
    for(size_t i = 0; i < p->hyperspectral_cameras.size(); i++)
    {
        printHyperspectralCameraStructure(p->hyperspectral_cameras[i]);
    }

}

void printScanStructure(const ScanPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    // TODO: Implement output for point buffer
}

void printLIDARStructure(const LIDARPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    for(size_t i = 0; i < p->scans.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[LiDAR] Scan " << i << " / " << p->scans.size() << lvr2::endl;
        printScanStructure(p->scans[i]);
    }
}

void printCameraStructure(const CameraPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;

    for(size_t i = 0; i < p->groups.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[Camera] Camera group " << i << " / " << p->groups.size() << lvr2::endl;
        CameraImageGroupPtr g = p->groups[i]; 
        lvr2::logout::get() << lvr2::info << "[Camera] Transformation: " << g->transformation << lvr2::endl;
        lvr2::logout::get() << lvr2::info << "[Camera] Type: " << g->type << lvr2::endl;
        lvr2::logout::get() << lvr2::info << "[Camera] Number of images: " << g->images.size() << lvr2::endl;
    }
}

void printCameraImageGroupStructure(const CameraImageGroupPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    for(size_t i = 0; i < p->images.size(); i++)
    {
        lvr2::logout::get() << lvr2::info << "[Image Group] Image " << i << " / " << p->images.size() << lvr2::endl;
        lvr2::logout::get() << lvr2::info << p->images[i];
    }
}

void printHyperspectralCameraStructure(const HyperspectralCameraPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    for(size_t i = 0; i < p->panoramas.size(); i++)
    {
        lvr2::logout::get() << lvr2::info <<  "[Hyperspectral Camera] Panorama " << i << " / " << p->panoramas.size() << lvr2::endl;
        printHyperspectralPanoramaStructure(p->panoramas[i]);
    }
}

void printHyperspectralPanoramaStructure(const HyperspectralPanoramaPtr p) 
{
    lvr2::logout::get() << lvr2::info << p;
    for(size_t i = 0; i < p->channels.size(); i++)
    {
       lvr2::logout::get() << lvr2::info <<  "[Panorama Structure] Channel " << i << " / " << p->channels.size() << lvr2::endl;
       lvr2::logout::get() << lvr2::info <<  p->channels[i];
    }
}

void printCameraImageStructure(const CameraImagePtr p)
{
    lvr2::logout::get() << lvr2::info << p;
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
                            lvr2::logout::get() << lvr2::info << "[Project Normal Estimation]: Loading scan " << scanNr 
                                      << " from lidar " << lidarNr << " of scan position " << positionNr << lvr2::endl;
                        
                            scan->load();
                            PointBufferPtr ptBuffer = scan->points;
                            if(ptBuffer)
                            {
                                size_t n = ptBuffer->numPoints();
                                if(n)
                                {
                                    lvr2::logout::get() << lvr2::info << "[Project Normal Estimation]: Loaded " << n << " points" << lvr2::endl;
                                    lvr2::logout::get() << lvr2::info << "[Project Normal Estimation]: Building search tree..." << lvr2::endl;

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
                                    lvr2::logout::get() << lvr2::warning << "[Project Normal Estimation]: No points in scan" << lvr2::endl;
                                }
       
                            } 
                            else
                            {
                                lvr2::logout::get() << lvr2::warning << "[Project Normal Estimation]:Unable to load point cloud data." << lvr2::endl;
                            }
                        }
                        else
                        {
                            lvr2::logout::get() << lvr2::warning << "[Project Normal Estimation]: "
                                      << "Unable to load scan " << scanNr << " of "
                                      << "lidar " << lidarNr << lvr2::endl; 
                        }
                    }
                }
                else
                {
                    lvr2::logout::get() << lvr2::warning << "[Project Normal Estimation]: Unable to load lidar " 
                                           << lidarNr << " of scan position " 
                                           << positionNr << lvr2::endl;
                }
            }
        }
        else
        {
            lvr2::logout::get() << lvr2::warning 
                      << "[Project Normal Estimation]: Unable to load scan position " 
                      << positionNr << lvr2::endl;
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
                    lvr2::logout::get() << lvr2::info << "[Load Positions Explicitly] : Loading scan position " << i << lvr2::endl;
                    p->positions.push_back(pos);
                }
                else
                {
                    lvr2::logout::get() << lvr2::info 
                              << "[Load Positions Explicitly] : Position with index " 
                              << i << " cannot be loaded from directory." << lvr2::endl;
                }
            }
            return p;
        }
        else
        {
            if(!kernel)
            {
                lvr2::logout::get() << lvr2::warning 
                          << "[Load Positions Explicitly] : Could not create directory kernel from root " 
                          << root << lvr2::endl; 
            }
            if(!dirSchema)
            {
                lvr2::logout::get() << lvr2::warning 
                          << "[Load Positions Explicitly] : Could not create directory schema from name " 
                          << schema << lvr2::endl; 
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
                    lvr2::logout::get() << lvr2::info << "[Load Positions Explicitly] : Loading scan position " << i << lvr2::endl;
                    p->positions.push_back(pos);
                }
                else
                {
                    lvr2::logout::get() << lvr2::warning 
                              << "[Load Positions Explicitly] : Position with index "
                              << i << " cannot be loaded from HDF5 file." << lvr2::endl;
                }
            }
            return p;
        }
        else
        {
            if(!kernel)
            {
                lvr2::logout::get() << lvr2::warning 
                          << "[Load Positions Explicitly] : Could not create HDF5 kernel from root " 
                          << root << lvr2::endl; 
            }
            if(!hdf5Schema)
            {
                lvr2::logout::get() << lvr2::warning 
                          << "[Load Positions Explicitly] : Could not create HDF5 schema from name " 
                          << schema << lvr2::endl; 
            }
        }
    }
    lvr2::logout::get() << lvr2::error << "[Load Positions Explicitly] : Could not load any data." << lvr2::endl;
    return nullptr;
}

void writeScanProjectToPLY(ScanProjectPtr project, const std::string plyFile, bool firstScanOnly)
{
    // Step 0: Check if output file is valid
    std::ofstream outfile;
    outfile.open(plyFile.c_str(), std::ios::binary);
    
    if(!outfile.good())
    {
        lvr2::logout::get() << lvr2::warning << "[WriteScanProjectToPLY]: Unable to open file '" << plyFile << "' for writing." << lvr2::endl;
    }

    // Step 1: Manually count all points in project, search for normal and color information.
    size_t numPointsInProject = 0;
    size_t scansWithNormals = 0;
    size_t scansWithColors = 0;
    size_t totalScans = 0;
    for(size_t positionNo = 0; positionNo < project->positions.size(); positionNo++)
    {
        for(size_t lidarNo = 0; lidarNo < project->positions[positionNo]->lidars.size(); lidarNo++)
        {
            LIDARPtr lidar = project->positions[positionNo]->lidars[lidarNo];
            if(lidar->scans.size() > 0)
            {
                for(size_t scanNo = 0; scanNo < lidar->scans.size(); scanNo++)
                {
                    // Stop of more data is present
                    if (scanNo > 1 && firstScanOnly)
                    {
                        break;
                    }

                    // Get current scan
                    ScanPtr scan = lidar->scans[scanNo];

                    // Load payload data 
                    scan->load();
                    PointBufferPtr points = scan->points;

                    if(points)
                    {
                        totalScans++;
                        numPointsInProject += points->numPoints();
                        if(points->hasColors())
                        {
                            scansWithColors++;
                        }

                        if(points->hasNormals())
                        {
                            scansWithNormals++;
                        }
                    }

                    // Release data. This causes IO overhead, but
                    // we do not want to store all data in RAM
                    scan->release();
                }
            }
        }
    }
    lvr2::logout::get() << lvr2::info << "[WriteScanProjectToPLY]: Scan project has " << numPointsInProject << "points." << lvr2::endl;
    lvr2::logout::get() << "[WriteScanProjectToPLY]: Found " << scansWithNormals << " scans with normals." << lvr2::endl;
    lvr2::logout::get() << "[WriteScanProjectToPLY]: Found " << scansWithColors << " scans with colors." << lvr2::endl;

    // Check color / normal consistency
    bool exportColors = (scansWithColors == totalScans);
    bool exportNormals = (scansWithNormals == totalScans);

    if(exportNormals)
    {
        lvr2::logout::get() << "[WriteScanProjectToPLY]: Exporting normals." << lvr2::endl;
    }

    if(exportColors)
    {
        lvr2::logout::get() << "[WriteScanProjectToPLY]: Exporting colors." << lvr2::endl;
    }
    
    // Step 2: Write PLY header
    outfile << "ply" << std::endl;
    outfile << "format binary_little_endian 1.0" << std::endl;
    outfile << "element vertex " << numPointsInProject << std::endl;
    outfile << "property float x" << std::endl;
    outfile << "property float y" << std::endl;
    outfile << "property float z" << std::endl;

    if(exportColors)
    {
        outfile << "property uchar red" << std::endl;
        outfile << "property uchar green" << std::endl;
        outfile << "property uchar blue" << std::endl;
    }

    if(exportNormals)
    {
        outfile << "property float nx" << std::endl;
        outfile << "property float ny" << std::endl;
        outfile << "property float nz" << std::endl;
    }
    outfile << "end_header" << std::endl;

    // Step 3: Write data chunks from scans
    size_t c = 0;
    for (size_t positionNo = 0; positionNo < project->positions.size(); positionNo++)
    {
        for (size_t lidarNo = 0; lidarNo < project->positions[positionNo]->lidars.size(); lidarNo++)
        {
            LIDARPtr lidar = project->positions[positionNo]->lidars[lidarNo];
            if (lidar->scans.size() > 0)
            {
                for (size_t scanNo = 0; scanNo < lidar->scans.size(); scanNo++)
                {
                    // Stop of more data is present
                    if (scanNo > 1 && firstScanOnly)
                    {
                        break;
                    }

                    // Get current scan
                    ScanPtr scan = lidar->scans[scanNo];

                    lvr2::logout::get() 
                        << lvr2::info << "[WriteScanProjectToPLY]: Exporting scan " 
                        << c + 1 << " of " << totalScans << lvr2::endl;
                    c++;
                    // Load payload data
                    scan->load();

                    PointBufferPtr pointBuffer = scan->points;
                    if (pointBuffer)
                    {
                        // Transform scan data
                        Transformd positionPose = project->positions[positionNo]->transformation;
                        Transformd lidarPose = project->positions[positionNo]->lidars[lidarNo]->transformation;
                        Transformd transformation = transformRegistration(positionPose, lidarPose);
                        
                        // lvr2::logout::get() << "Pose: " << positionPose << lvr2::endl;
                        // lvr2::logout::get() << "Lidar: " << lidarPose << lvr2::endl;
                        // lvr2::logout::get() << "Transform: " << transformation << lvr2::endl;

                        transformPointBuffer(pointBuffer, transformation);

                        size_t np, nn, nc;
                        size_t w_color;
                        np = pointBuffer->numPoints();
                        nc = nn = np;

                        floatArr points = pointBuffer->getPointArray();
                        ucharArr colors = pointBuffer->getColorArray(w_color);
                        floatArr normals = pointBuffer->getNormalArray();

                        // Determine size of single point
                        size_t buffer_size = 3 * sizeof(float);

                        if (colors)
                        {
                            buffer_size += w_color * sizeof(unsigned char);
                        }

                        if (normals)
                        {
                            buffer_size += 3 * sizeof(float);
                        }

                        char *buffer = new char[buffer_size];

                        for (size_t i = 0; i < np; i++)
                        {
                            char *ptr = &buffer[0];

                            // Write coordinates to buffer
                            *((float *)ptr) = points[3 * i];
                            ptr += sizeof(float);
                            *((float *)ptr) = points[3 * i + 1];
                            ptr += sizeof(float);
                            *((float *)ptr) = points[3 * i + 2];

                            // Write colors to buffer
                            if (colors)
                            {
                                ptr += sizeof(float);
                                *((unsigned char *)ptr) = colors[3 * i];
                                ptr += sizeof(unsigned char);
                                *((unsigned char *)ptr) = colors[3 * i + 1];
                                ptr += sizeof(unsigned char);
                                *((unsigned char *)ptr) = colors[3 * i + 2];
                            }

                            if (normals)
                            {
                                ptr += sizeof(unsigned char);
                                *((float *)ptr) = normals[3 * i];
                                ptr += sizeof(float);
                                *((float *)ptr) = normals[3 * i + 1];
                                ptr += sizeof(float);
                                *((float *)ptr) = normals[3 * i + 2];
                            }
                            outfile.write((const char *)buffer, buffer_size);
                        }

                        delete[] buffer;

                        // Release data again.
                        scan->release();
                    }
                }
            }
        }
    }
    outfile.close();
}

} // namespace lvr2 
     