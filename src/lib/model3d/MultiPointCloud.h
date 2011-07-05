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
using std::map;

struct PointCloudAttribute
{
    PointCloudAttribute() : cloud(0), marked(false), active(true) {}
    PointCloud*  cloud;
    bool         marked;
    bool         active;
};

class MultiPointCloud : public Renderable
{
public:
    MultiPointCloud();
    virtual ~MultiPointCloud();
    virtual inline void render();

    void addCloud(PointCloud* pc);
    void removeCloud(PointCloud* pc);

private:
    map<PointCloud*, PointCloudAttribute*>    m_clouds;
};

void MultiPointCloud::render()
{
    map<PointCloud*, PointCloudAttribute*>::iterator it;
    for(it = m_clouds.begin(); it != m_clouds.end(); it++)
    {
        it->second->cloud->render();
    }
}

#endif /* MULTIPOINTCLOUD_H_ */
