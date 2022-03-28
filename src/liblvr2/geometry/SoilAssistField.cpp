//
// Created by imitschke on 13.11.20.
//
#include "lvr2/geometry/SoilAssistField.hpp"
#include "highfive/H5Easy.hpp"
#include "highfive/H5Group.hpp"
namespace lvr2
{
    SoilAssistField::SoilAssistField()
    {

    }
    SoilAssistField::SoilAssistField(std::string name, PolygonBufferPtr boundary, std::vector<SoilAssistSubFieldPtr>& subfields) : m_name(name), m_boundary(boundary), m_subfields(subfields)
    {
    }
    std::vector<SoilAssistSubFieldPtr> SoilAssistField::getSubFields()
    {
        return m_subfields;
    }
    PolygonBufferPtr SoilAssistField::getBoundary()
    {
        return m_boundary;
    }
    std::string SoilAssistField::getName()
    {
        return m_name;
    }

    void SoilAssistField::setSubFields(std::vector<SoilAssistSubFieldPtr>& fields)
    {
        m_subfields = fields;
    }
    void SoilAssistField::addSubField(SoilAssistSubFieldPtr field)
    {
        m_subfields.push_back(field);
    }

    void SoilAssistField::setBoundary(PolygonBufferPtr boundary)
    {
        m_boundary = boundary;
    }
    void SoilAssistField::setName(std::string name)
    {
        m_name = name;
    }

    void SoilAssistField::fromH5File(std::string path)
    {
        HighFive::File hdf5_file(path, HighFive::File::ReadOnly);
        std::vector<std::vector<float> > hborder = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file, "/field/outer_boundary/coordinates");
        this->setBoundary(PolygonBufferPtr(new PolygonBuffer(hborder)));
        if(hdf5_file.exist("/field/subfields"))
        {
            auto subfields_group = hdf5_file.getGroup("/field/subfields");
            auto subFieldNames = subfields_group.listObjectNames();
            for(auto subFieldName : subFieldNames)
            {
                SoilAssistSubFieldPtr subfield(new SoilAssistSubField);
                std::string boundary_path = "/field/subfields/" + subFieldName + "/boundary_outer/coordinates";
                if(hdf5_file.exist(boundary_path))
                {
                    std::vector<std::vector<float> > hborder = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file,boundary_path );
                    subfield->setBoundary(PolygonBufferPtr(new PolygonBuffer(hborder)));
                }

                std::string access_points_path = "/field/subfields/" + subFieldName + "/access_points/coordinates";
                if(hdf5_file.exist(access_points_path))
                {
                    std::vector<std::vector<float> > hpoints = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file,access_points_path );
                    std::vector<floatArr> acc_points;
                    for(auto&& p : hpoints)
                    {
                        floatArr tmpp(new float[3]);
                        tmpp[0] = p[0];
                        tmpp[1] = p[1];
                        tmpp[2] = p[2];
                        acc_points.push_back(tmpp);
                    }
                    subfield->setAccessPoints(acc_points);
                }
                // ReferenceLines:
                if(hdf5_file.exist("/field/subfields/" + subFieldName + "/reference_lines"))
                {
                    auto refLineGroup = hdf5_file.getGroup("/field/subfields/" + subFieldName + "/reference_lines");
                    auto refLineNames = refLineGroup.listObjectNames();
                    for(auto && refLineName : refLineNames)
                    {
                        auto ref_path = "/field/subfields/" + subFieldName + "/reference_lines/" + refLineName + "/coordinates";
                        if(hdf5_file.exist(ref_path))
                        {
                            std::vector<std::vector<float> > hpoints = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file,ref_path );
                            subfield->addReferenceLine(PolygonBufferPtr(new PolygonBuffer(hpoints)));
                        }
                    }
                }
                // headlands:
                if(hdf5_file.exist("/field/subfields/" + subFieldName + "/headlands"))
                {
                    auto headlandGroup = hdf5_file.getGroup("/field/subfields/" + subFieldName + "/headlands");
                    auto headlandNames = headlandGroup.listObjectNames();
                    for(auto && headlandName : headlandNames)
                    {
                        auto ref_path = "/field/subfields/" + subFieldName + "/headlands/" + headlandName + "/coordinates";
                        if(hdf5_file.exist(ref_path))
                        {
                            std::vector<std::vector<float> > hpoints = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file,ref_path );
                            subfield->addHeadland(PolygonBufferPtr(new PolygonBuffer(hpoints)));
                        }
                    }
                }
                this->addSubField(subfield);
            }
        }
    }
}