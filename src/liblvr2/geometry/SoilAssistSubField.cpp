//
// Created by imitschke on 13.11.20.
//
#include "lvr2/geometry/SoilAssistSubField.hpp"

namespace lvr2
{

    SoilAssistSubField::SoilAssistSubField(){}

    void SoilAssistSubField::setHeadlands(std::vector<PolygonBufferPtr>& headlands)
    {
        m_headlands = headlands;
    }
    void SoilAssistSubField::addHeadland(PolygonBufferPtr  headland)
    {
        m_headlands.push_back(headland);
    }

    void SoilAssistSubField::setReferenceLines(std::vector<PolygonBufferPtr>& lines)
    {
        m_reference_lines = lines;
    }
    void SoilAssistSubField::addReferenceLine(PolygonBufferPtr line)
    {
        m_reference_lines.push_back(line);
    }

    void SoilAssistSubField::setBoundary(PolygonBufferPtr boundary)
    {
        m_boundary = boundary;
    }
    void SoilAssistSubField::setAccessPoints(std::vector<floatArr>& pts)
    {
        m_access_points = pts;
    }
    void SoilAssistSubField::addAccessPoint(floatArr point)
    {
        m_access_points.push_back(point);
    }
    void SoilAssistSubField::addAccessPoint(float* p)
    {
        floatArr fa(new float[3]);
        fa[0] = p[0];
        fa[1] = p[1];
        fa[2] = p[2];
        addAccessPoint(fa);
    }

    std::vector<PolygonBufferPtr> SoilAssistSubField::getHeadlands()
    {
        return m_headlands;
    }
    std::vector<PolygonBufferPtr> SoilAssistSubField::getReferenceLines()
    {
        return m_reference_lines;
    }
    PolygonBufferPtr SoilAssistSubField::getBoundary()
    {
        return m_boundary;
    }
    std::vector<floatArr> SoilAssistSubField::getAccessPoints()
    {
        return m_access_points;
    }
}
