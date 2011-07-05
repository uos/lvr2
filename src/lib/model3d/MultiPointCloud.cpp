/**
 * MultiPointCloud.cpp
 *
 *  @date 04.07.2011
 *  @author Thomas Wiemann
 */

#include "MultiPointCloud.h"

MultiPointCloud::MultiPointCloud()
{
    // TODO Auto-generated constructor stub

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
}

void MultiPointCloud::removeCloud(PointCloud* pc)
{
    m_clouds.erase(pc);
}

