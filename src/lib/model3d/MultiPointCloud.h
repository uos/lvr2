/**
 * MultiPointCloud.h
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#ifndef MULTIPOINTCLOUD_H_
#define MULTIPOINTCLOUD_H_

#include "PointCloud.h"

#include <map>
#include <string>
#include <sstream>

using std::stringstream;
using std::map;
using std::string;

struct PointCloudAttribute
{
    PointCloudAttribute() : cloud(0), marked(false), active(true) {}
    PointCloud*  cloud;
    bool         marked;
    bool         active;
};

typedef map<PointCloud*, PointCloudAttribute*> pc_attr_map;
typedef map<PointCloud*, PointCloudAttribute*>::iterator pc_attr_it;

class MultiPointCloud : public Renderable
{
public:
    MultiPointCloud(string pathname);
    virtual ~MultiPointCloud();
    virtual inline void render();

    void addCloud(PointCloud* pc);
    void removeCloud(PointCloud* pc);

    pc_attr_it first() { return m_clouds.begin();}
    pc_attr_it last()  { return m_clouds.end();}

    void exportAllPoints(string filename);

private:

    map<PointCloud*, PointCloudAttribute*>    m_clouds;
};

void MultiPointCloud::render()
{
    if(!active) return;
    map<PointCloud*, PointCloudAttribute*>::iterator it;
    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        it->second->cloud->render();
    }
}

#endif /* MULTIPOINTCLOUD_H_ */
