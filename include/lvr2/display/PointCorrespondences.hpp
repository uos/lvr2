#ifndef POINTCORRESPONDENCES_H
#define POINTCORRESPONDENCES_H

#include <lvr2/display/Renderable.hpp>

#include <vector>

namespace lvr2
{

class PointCorrespondences : public Renderable
{
public:
    PointCorrespondences(std::vector<float> points1, std::vector<float> points2);
    virtual void render();
private:
    GLuint m_displayList;
};

} // namespace lvr2

#include "PointCorrespondences.cpp"

#endif // POINTCORRESPONDENCES_H
