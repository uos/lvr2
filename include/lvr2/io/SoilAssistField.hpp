//
// Created by imitschke on 13.11.20.
//

#ifndef LAS_VEGAS_SOILASSISTFIELD_HPP
#define LAS_VEGAS_SOILASSISTFIELD_HPP
#include "Polygon.hpp"
#include "SoilAssistSubField.hpp"

namespace lvr2
{
class SoilAssistField {
public:
    SoilAssistField();
    SoilAssistField(std::string name, PolygonPtr m_boundary, std::vector<SoilAssistSubFieldPtr>& subfields);
    std::vector<SoilAssistSubFieldPtr> getSubFields();
    PolygonPtr getBoundary();
    std::string getName();

    void setSubFields(std::vector<SoilAssistSubFieldPtr>& fields);
    void addSubField(SoilAssistSubFieldPtr field);

    void setBoundary(PolygonPtr boundary);
    void setName(std::string name);

    void fromH5File(std::string path);
private:

    std::vector<SoilAssistSubFieldPtr> m_subfields;
    PolygonPtr m_boundary;
    std::string m_name;

    inline floatArr vec2Arr(std::vector<std::vector<float>> in)
    {
        floatArr arr(new float[in.size()*3]);
        for(size_t i = 0 ; i < in.size() ; i++)
        {
            arr[i*3] = in[i][0];
            arr[i*3+1] = in[i][1];
            arr[i*3+2] = in[i][2];
        }
        return arr;
    }
};

    using SoilAssistFieldPtr = std::shared_ptr<SoilAssistField>;

}




#endif //LAS_VEGAS_SOILASSISTFIELD_HPP
