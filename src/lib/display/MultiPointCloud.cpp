/**
 * MultiPointCloud.cpp
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#include <fstream>
using std::ofstream;

#include "MultiPointCloud.hpp"
#include <boost/filesystem.hpp>

#include "../io/UosIO.hpp"

namespace lssr
{

MultiPointCloud::MultiPointCloud(string dir)
{
    boost::filesystem::path directory(dir);
    if(is_directory(directory))
    {
        UosIO io;
        io.read(dir);
        int n = io.getNumScans();
        for(int i = 0; i < n; i++)
        {
            indexPair p = io.getScanRange(i);

            // Create new point cloud from scan
            PointCloud* pc = new PointCloud;
            for(int a = p.first; a <= p.second; a++)
            {
                float** points = io.getPointArray();
                unsigned char** colors = io.getPointColorArray();
                if(colors)
                {
                    pc->addPoint(points[a][0], points[a][1], points[a][2], colors[a][0], colors[a][1], colors[a][2]);
                }
                else
                {
                    pc->addPoint(points[a][0], points[a][1], points[a][2], 255, 0, 0);
                }
            }
            pc->updateDisplayLists();
            pc->setName(dir);
            addCloud(pc);
        }
    }
    else
    {
        cout << "MultiPointCloud: " << dir << " is not a directory." << endl;
    }
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
                vector<uColorVertex>::iterator p_it;
                for(p_it = pc->m_points.begin(); p_it != pc->m_points.end(); p_it++)
                {
                    uColorVertex v = *p_it;
                    out << v.x << " " << v.y << " " << v.z <<  " "
                        << (int)v.r << " " << (int)v.g << " " << (int)v.b << endl;
                }
            }
        }
        out.close();
    }

}

}
