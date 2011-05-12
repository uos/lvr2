/**
 * UosIO.tcc
 *
 *  @date 11.05.2011
 *  @author Thomas Wiemann
 */

#include "UosIO.hpp"


#include <list>
#include <vector>
using std::list;
using std::vector;
using std::ifstream;

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

#include "../geometry/Vertex.hpp"
#include "../geometry/Matrix4.hpp"
#include "Progress.hpp"
#include "Timestamp.hpp"

namespace lssr
{

template<typename T>
T** UosIO<T>::read(string dir, size_t &n)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {
        // Iterate through directory, count relevant objects
        int n3dFiles = 0;

        // First and last scan to load
        int firstScan = -1;
        int lastScan =  -1;

        directory_iterator lastFile;

        // First, look for .3d files
        for(directory_iterator it(directory); it != lastFile; it++ )
        {
            path p = it->path();
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
            cout << timestamp << "Reading " << n3dFiles << " scans in UOS format "
                 << "(From " << firstScan << " to " << lastScan << ")." << endl;
            return readNewFormat(dir, firstScan, lastScan, n);
        }
        else
        {
            // Count numbered sub directories, ignore others
            int nDirs = 0;
            for(directory_iterator it(directory); it != lastFile; it++ )
            {
                path p = it->path();
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
                cout << timestamp << "Reading " << nDirs << " scans in old UOS format "
                     << "(From " << firstScan << " to " << lastScan << ")." << endl;
                return readOldFormat(dir, firstScan, lastScan, n);
            }
            else
            {
                return 0;
            }
        }

    }
    else
    {
        cout << timestamp << "UOSReader: " << dir << " is not a directory." << endl;
    }
}

template<typename T>
T** UosIO<T>::readNewFormat(string dir, int first, int last, size_t &n)
{
    return 0;
}

template<typename T>
T** UosIO<T>::readOldFormat(string dir, int first, int last, size_t &n)
{
    Matrix4<float> m_tf;

    list<Vertex<float> > ptss;
    list<Vertex<float> > allPoints;
    for(int fileCounter = first; fileCounter <= last; fileCounter++)
    {
        float euler[6];
        ifstream scan_in, pose_in, frame_in;

        // Code imported from slam6d! Don't blame me..
        string scanFileName;
        string poseFileName;

        // Create correct path
        path p(path(dir)/path(to_string(fileCounter,3))/path("position.dat"));

        // Get file name (if some knows a more elegant way to
        // extract the pull path let me know
        poseFileName = "/" + p.relative_path().string();

        // Try to open file
        pose_in.open(poseFileName.c_str());

        // Abort if opening failed and try with next die
        if (!pose_in.good()) continue;
        cout << timestamp << "Processing Scan " << dir << "/" << to_string(fileCounter, 3) << endl;

        // Extract pose information
        for (unsigned int i = 0; i < 6; pose_in >> euler[i++]);

        // Convert mm to cm
        for (unsigned int i = 0; i < 3; i++) euler[i] = euler[i] * 0.1;

        // Convert angles from deg to rad
        for (unsigned int i = 3; i <= 5; i++) {
            euler[i] *= 0.01;
            //   if (euler[i] < 0.0) euler[i] += 360;
            euler[i] = rad(euler[i]);
        }

        // Read and convert scan
        for (int i = 1; ; i++) {
            //scanFileName = dir + to_string(fileCounter, 3) + "/scan" + to_string(i,3) + ".dat";

            path sfile(path(dir)/path(to_string(fileCounter, 3))/path("scan" + to_string(i) + ".dat"));
            scanFileName = "/" + sfile.relative_path().string();

            scan_in.open(scanFileName.c_str());
            if (!scan_in.good()) {
                scan_in.close();
                scan_in.clear();
                break;
            }


            int    Nr = 0, intensity_flag = 0;
            int    D;
            double current_angle;
            double X, Z, I;                     // x,z coordinate and intensity

            char firstLine[81];
            scan_in.getline(firstLine, 80);

            char cNr[4];
            cNr[0] = firstLine[2];
            cNr[1] = firstLine[3];
            cNr[2] = firstLine[4];
            cNr[3] = 0;
            Nr = atoi(cNr);

            // determine weather we have the new files with intensity information
            if (firstLine[16] != 'i') {
                intensity_flag = 1;
                char cAngle[8];
                cAngle[0] = firstLine[35];
                cAngle[1] = firstLine[36];
                cAngle[2] = firstLine[37];
                cAngle[3] = firstLine[38];
                cAngle[4] = firstLine[39];
                cAngle[5] = firstLine[40];
                cAngle[6] = firstLine[41];
                cAngle[7] = 0;
                current_angle = atof(cAngle);
                cout << current_angle << endl;
            } else {
                intensity_flag = 0;
                char cAngle[8];
                cAngle[0] = firstLine[54];
                cAngle[1] = firstLine[55];
                cAngle[2] = firstLine[56];
                cAngle[3] = firstLine[57];
                cAngle[4] = firstLine[58];
                cAngle[5] = firstLine[59];
                cAngle[6] = firstLine[60];
                cAngle[7] = 0;
                current_angle = atof(cAngle);
            }

            double cos_currentAngle = cos(rad(current_angle));
            double sin_currentAngle = sin(rad(current_angle));

            for (int j = 0; j < Nr; j++) {
                if (!intensity_flag) {
                    scan_in >> X >> Z >> D >> I;
                } else {
                    scan_in >> X >> Z;
                    I = 1.0;
                }

                // calculate 3D coordinates (local coordinates)
                Vertex<float> p;
                p[0] = X;
                p[1] = Z * sin_currentAngle;
                p[2] = Z * cos_currentAngle;

                ptss.push_back(p);
            }
            scan_in.close();
            scan_in.clear();
        }

        pose_in.close();
        pose_in.clear();

        // Create path to frame file
        path framePath(path(dir)/path("scan" + to_string(fileCounter,3) + ".frames"));
        string frameFileName = "/" + framePath.relative_path().string();

        // Try to open frame file
        frame_in.open(frameFileName.c_str());
        if(frame_in.good())
        {
            // Transform scan data according to frame file
            m_tf = parseFrameFile(frame_in);
        }
        else
        {
            // Transform scan data using information from 'position.dat'
            Vertex<float> position(euler[0], euler[1], euler[2]);
            Vertex<float> angle(euler[3], euler[4], euler[5]);
            m_tf = Matrix4<float>(position, angle);
        }

        // Transform points and insert in to global vector
        list<Vertex<float> >::iterator it;
        for(it = ptss.begin(); it != ptss.end(); it++)
        {
            Vertex<float> v = *it;
            v.transformCM(m_tf);
            allPoints.push_back(v);
        }

        // Clear scan
        ptss.clear();
    }

    // Convert into indexed array
    if(allPoints.size() > 0)
    {
        cout << timestamp << "Read " << allPoints.size() << " points." << endl;
        n = allPoints.size();
        T** out_pts = new T*[allPoints.size()];
        list<Vertex<float> >::iterator p_it;
        int i = 0;
        for(p_it = allPoints.begin(); p_it != allPoints.end(); p_it++)
        {
            out_pts[i] = new T[3];
            Vertex<float> v = *p_it;
            out_pts[i][0] = v[0];
            out_pts[i][1] = v[1];
            out_pts[i][2] = v[2];
            i++;
        }
        return out_pts;
    }
    else
    {
        // If everything else failed return a null pointer
        return 0;
    }


}

template<typename T>
Matrix4<float> UosIO<T>::parseFrameFile(ifstream& frameFile)
{
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }

    return Matrix4<float>(m);
}

} // namespace lssr
