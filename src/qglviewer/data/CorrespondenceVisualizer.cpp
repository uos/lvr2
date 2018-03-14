#include "CorrespondenceVisualizer.hpp"

#include <lvr/display/PointCorrespondences.hpp>

#include "../widgets/PointCloudTreeWidgetItem.h"

#include <vector>
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

CorrespondenceVisualizer::CorrespondenceVisualizer(string filename)
{
    ifstream in(filename.c_str());

    float f11, f12, f13;
    float f21, f22, f23;

    vector<float> p1;
    vector<float> p2;

    if(in.good())
    {
        while(in.good())
        {
            in >> f11 >> f12 >> f13;
            in >> f21 >> f22 >> f23;
            p1.push_back(f11);
            p1.push_back(f12);
            p1.push_back(f13);

            p2.push_back(f21);
            p2.push_back(f22);
            p2.push_back(f23);
        }
    }
    else
    {
        std::cout << "CorrespondenceVisualizer: Could not open file " << filename << endl;
    }

    lvr::PointCorrespondences* corr = new lvr::PointCorrespondences(p1, p2);

    m_renderable = corr;

    int modes = 0;
    PointCloudTreeWidgetItem* item = new PointCloudTreeWidgetItem(PointCloudItem);
    item->setSupportedRenderModes(modes);
    item->setViewCentering(false);
    item->setName("Correspondences");
    item->setRenderable(corr);
    m_treeItem = item;
}
