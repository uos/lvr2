#ifndef POINTCORRESPONDENCES_H
#define POINTCORRESPONDENCES_H

#include "Renderable.hpp"

#include <vector>

namespace lvr
{

class PointCorrespondences : public Renderable
{
public:
    PointCorrespondences(std::vector<float> points1, std::vector<float> points2);
    virtual void render();
private:
    GLuint m_displayList;
};

} // namespace lvr

#endif // POINTCORRESPONDENCES_H
