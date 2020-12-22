//
// Created by imitschke on 13.11.20.
//
#include "lvr2/io/SoilAssistSubField.hpp"

namespace lvr2
{

    SoilAssistSubField::SoilAssistSubField(){}

    void SoilAssistSubField::setHeadlands(std::vector<PolygonPtr>& headlands)
    {
        m_headlands = headlands;
    }
    void SoilAssistSubField::addHeadland(PolygonPtr  headland)
    {
        m_headlands.push_back(headland);
    }

    void SoilAssistSubField::setReferenceLines(std::vector<PolygonPtr>& lines)
    {
        m_reference_lines = lines;
    }
    void SoilAssistSubField::addReferenceLine(PolygonPtr line)
    {
        m_reference_lines.push_back(line);
    }

    void SoilAssistSubField::setBoundary(PolygonPtr boundary)
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

    std::vector<PolygonPtr> SoilAssistSubField::getHeadlands()
    {
        return m_headlands;
    }
    std::vector<PolygonPtr> SoilAssistSubField::getReferenceLines()
    {
        return m_reference_lines;
    }
    PolygonPtr SoilAssistSubField::getBoundary()
    {
        return m_boundary;
    }
    std::vector<floatArr> SoilAssistSubField::getAccessPoints()
    {
        return m_access_points;
    }
}
