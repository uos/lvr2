/**
 * MultiPointCloud.cpp
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#include <fstream>
using std::ofstream;

#include "MultiPointCloud.h"
#include <boost/filesystem.hpp>

MultiPointCloud::MultiPointCloud(string dir)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {
        // Iterate through directory, count relevant objects
        int n3dFiles = 0;

        // First and last scan to load
        int firstScan = -1;
        int lastScan =  -1;

        boost::filesystem::directory_iterator lastFile;

        // First, look for .3d files
        for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
        {
            boost::filesystem::path p = it->path();
            if(string(p.extension().c_str()) == ".3d")
            {
                // Check for naming convention "scanxxx.3d"
                int num = 0;
                if(sscanf(p.filename().c_str(), "scan%3d", &num))
                {
                    n3dFiles++;
                    if(firstScan == -1) firstScan = num;
                    if(lastScan == -1) lastScan = num;

                    if(num > lastScan) lastScan = num;
                    if(num < firstScan) firstScan = num;
                }
            }
        }

        // Check if given directory contains scans in new format.
        // If so, read them and return result. Otherwise try to
        // read new format.
        if(n3dFiles > 0)
        {
            cout << "Reading " << n3dFiles << " scans in UOS format "
                 << "(From " << firstScan << " to " << lastScan << ")." << endl;
            readNewUOSFormat(dir, firstScan, lastScan);
        }
        else
        {
            // Count numbered sub directories, ignore others
            int nDirs = 0;
            for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
            {
                boost::filesystem::path p = it->path();
                int num = 0;

                // Only count numbered dirs
                if(sscanf(p.filename().c_str(), "%d", &num))
                {
                    if(firstScan == -1) firstScan = num;
                    if(lastScan == -1) lastScan = num;


                    if(num > lastScan) lastScan = num;
                    if(num < firstScan) firstScan = num;

                    nDirs++;
                }
            }

            // Check is dirs were found and try to read old format
            if(nDirs)
            {
                cout << "Reading " << nDirs << " scans in old UOS format "
                     << "(From " << firstScan << " to " << lastScan << ")." << endl;
                readOldUOSFormat(dir, firstScan, lastScan);
            }
            else
            {
                return;
            }
        }

    }
    else
    {
        cout << "MultiPointCloud: " << dir << " is not a directory." << endl;
    }
}

void MultiPointCloud::readNewUOSFormat(string dir, int first, int last)
{

    for(int fileCounter = first; fileCounter <= last; fileCounter++)
    {
        // New (unit) transformation matrix
        Matrix4 tf;

        // Input file streams for scan data, poses and frames
        ifstream scan_in, pose_in, frame_in;

        // Create scan file name
        boost::filesystem::path scan_path(
                boost::filesystem::path(dir) /
                boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".3d" ) );
        string scanFileName = "/" + scan_path.relative_path().string();

        // Read scan data
        scan_in.open(scanFileName.c_str());
        if(!scan_in.good())
        {
            // Continue with next file if the expected file couldn't be read
            cout << "MultiPointCloud: Unable to read scan " << scanFileName << endl;
            scan_in.close();
            scan_in.clear();
            continue;
        }
        else
        {
            // Create new point cloud
            PointCloud* pc = new PointCloud;


            // Try to get fransformation from .frames file
            boost::filesystem::path frame_path(
                    boost::filesystem::path(dir) /
                    boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".frames" ) );
            string frameFileName = "/" + frame_path.relative_path().string();
            cout << frameFileName << endl;

            frame_in.open(frameFileName.c_str());

            if(!frame_in.good())
            {
                // Try to parse .pose file
                boost::filesystem::path pose_path(
                        boost::filesystem::path(dir) /
                        boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".pose" ) );
                string poseFileName = "/" + pose_path.relative_path().string();

                pose_in.open(poseFileName.c_str());
                if(pose_in.good())
                {
                    float euler[6];
                    for(int i = 0; i < 6; i++) pose_in >> euler[i];
                    Vertex position(euler[0], euler[1], euler[2]);
                    Vertex angle(euler[3], euler[4], euler[5]);
                    tf = Matrix4(position, angle);
                }
                else
                {
                    cout << "MultiPointCloud: Warning: No position information found." << endl;
                    tf = Matrix4();
                }

            }
            else
            {
                // Use transformation from .frame files
                tf = parseFrameFile(frame_in);

            }

            // Print pose information
            float euler[6];
            tf.toPostionAngle(euler);

            cout << "Processing " << scanFileName << " @ "
                 << euler[0] << " " << euler[1] << " " << euler[2] << " "
                 << euler[3] << " " << euler[4] << " " << euler[5] << endl;

            // Skip first line in scan file (maybe metadata)
            char dummy[1024];
            scan_in.getline(dummy, 1024);

            // Read all points
            while(scan_in.good())
            {
                /// TODO: Check for intensity and/or color values in file
                float x, y, z;
                scan_in >> x >> y >> z;

                Vertex point(x, y, z);
                cout << "ORIG:  " << point;
                point.transform(tf);
                cout << "TRANS: " << point << endl;;
                pc->addPoint(x, y, z, 255, 0, 0);
            }

            // Update display list of point cloud to
            // compile new points into a new OpenGL display list
            pc->updateDisplayLists();
            pc->setName(scan_path.filename().c_str());
            addCloud(pc);
        }

    }

}

Matrix4 MultiPointCloud::parseFrameFile(ifstream& frameFile)
{
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }
    return Matrix4(m);
}

void MultiPointCloud::readOldUOSFormat(string dir, int first, int last)
{
    cout << "MultiPointCloud: Old UOS Format currently not supported" << endl;
}

MultiPointCloud::~MultiPointCloud()
{
    // TODO Auto-generated destructor stub
}

void MultiPointCloud::addCloud(PointCloud* pc)
{
    PointCloudAttribute* a = new PointCloudAttribute;
    a->cloud = pc;
    m_clouds[pc] = a;
    m_boundingBox->expand(*(pc->boundingBox()));
}

void MultiPointCloud::removeCloud(PointCloud* pc)
{
    m_clouds.erase(pc);
}

void MultiPointCloud::exportAllPoints(string filename)
{
    ofstream out(filename.c_str());
    if(out.good())
    {

        pc_attr_it it;
        for(it = m_clouds.begin(); it != m_clouds.end(); it++)
        {
            PointCloud* pc = it->second->cloud;
            if(pc->isActive())
            {
                cout << "Exporting points from " << pc->Name() << " to " << filename << endl;
                vector<ColorVertex>::iterator p_it;
                for(p_it = pc->points.begin(); p_it != pc->points.end(); p_it++)
                {
                    ColorVertex v = *p_it;
                    out << v.x << " " << v.y << " " << v.z <<  " "
                        << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
                }
            }
        }
        out.close();
    }

}
