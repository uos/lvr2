//
// Created by imitschke on 13.11.20.
//

#ifndef LAS_VEGAS_SOILASSISTSUBFIELD_HPP
#define LAS_VEGAS_SOILASSISTSUBFIELD_HPP
#include "Polygon.hpp"

namespace lvr2
{
class SoilAssistSubField{
public:
    SoilAssistSubField();

    void setHeadlands(std::vector<PolygonPtr>& headlands);
    void addHeadland(PolygonPtr  headland);

    void setReferenceLines(std::vector<PolygonPtr>& lines);
    void addReferenceLine(PolygonPtr line);

    void setBoundary(PolygonPtr boundary);
    void setAccessPoints(std::vector<floatArr>& pts);
    void addAccessPoint(floatArr point);
    void addAccessPoint(float* p);

    std::vector<PolygonPtr> getHeadlands();
    std::vector<PolygonPtr> getReferenceLines();
    PolygonPtr getBoundary();
    std::vector<floatArr> getAccessPoints();

private:
    std::vector<PolygonPtr> m_headlands;
    std::vector<PolygonPtr>  m_reference_lines;
    PolygonPtr m_boundary;
    std::vector<floatArr> m_access_points;


};

    using SoilAssistSubFieldPtr = std::shared_ptr<SoilAssistSubField>;
}



#endif //LAS_VEGAS_SOILASSISTSUBFIELD_HPP
