#ifndef POINTCORRESPONDENCES_H
#define POINTCORRESPONDENCES_H

#include "Renderable.hpp"

#include <vector>

class PointCorrespondences
{
public:
    PointCorrespondences(std::vector<float> points1, std::vector<float> points2);
    virtual void render();
private:
    GLuint m_displayList;
};

#endif // POINTCORRESPONDENCES_H
