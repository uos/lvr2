//
// Created by imitschke on 13.11.20.
//

#ifndef LAS_VEGAS_SOILASSISTSUBFIELD_HPP
#define LAS_VEGAS_SOILASSISTSUBFIELD_HPP
#include "lvr2/io/PolygonBuffer.hpp"

namespace lvr2
{
class SoilAssistSubField{
public:
    SoilAssistSubField();

    void setHeadlands(std::vector<PolygonBufferPtr>& headlands);
    void addHeadland(PolygonBufferPtr  headland);

    void setReferenceLines(std::vector<PolygonBufferPtr>& lines);
    void addReferenceLine(PolygonBufferPtr line);

    void setBoundary(PolygonBufferPtr boundary);
    void setAccessPoints(std::vector<floatArr>& pts);
    void addAccessPoint(floatArr point);
    void addAccessPoint(float* p);

    std::vector<PolygonBufferPtr> getHeadlands();
    std::vector<PolygonBufferPtr> getReferenceLines();
    PolygonBufferPtr getBoundary();
    std::vector<floatArr> getAccessPoints();

private:
    std::vector<PolygonBufferPtr> m_headlands;
    std::vector<PolygonBufferPtr>  m_reference_lines;
    PolygonBufferPtr m_boundary;
    std::vector<floatArr> m_access_points;


};

    using SoilAssistSubFieldPtr = std::shared_ptr<SoilAssistSubField>;
}



#endif //LAS_VEGAS_SOILASSISTSUBFIELD_HPP
