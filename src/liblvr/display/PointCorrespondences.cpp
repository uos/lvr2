#include <lvr/display/PointCorrespondences.hpp>

namespace lvr
{

PointCorrespondences::PointCorrespondences(std::vector<float> points1, std::vector<float> points2)
{
    m_displayList = glGenLists(1);
    glNewList(m_displayList, GL_COMPILE);

    glBegin(GL_POINTS);
    // Render points
    for(size_t i = 0; i < points1.size() / 3; i++)
    {
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(points1[i * 3], points1[i * 3 + 1], points1[i * 3 + 2]);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(points2[i * 3], points2[i * 3 + 1], points2[i * 3 + 2]);
    }
    glEnd();
    for(size_t i = 0; i < points1.size() / 3; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(points1[i * 3], points1[i * 3 + 1], points1[i * 3 + 2]);
        glVertex3f(points2[i * 3], points2[i * 3 + 1], points2[i * 3 + 2]);
        glEnd();
    }
    glEndList();


}

void PointCorrespondences::PointCorrespondences::render()
{
    glLineWidth(1.0);
    glPointSize(1.0);
    glCallList(m_displayList);
}

} // namespace lvr
