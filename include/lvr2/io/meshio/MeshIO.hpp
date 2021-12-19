#pragma once

#include "MeshSchema.hpp"
#include <lvr2/io/scanio/FileKernel.hpp>
#include <lvr2/io/meshio/MeshSchemaDirectory.hpp>
#include <lvr2/io/meshio/MeshSchemaHDF5.hpp>
#include <lvr2/io/meshio/FeatureBase.hpp>

namespace lvr2
{

template <typename FeatureBase>
class MeshIO
{
public:
    void saveMesh(
        std::string mesh_name, 
        MeshBufferPtr mesh
        ) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};

template <typename FeatureBase>
struct meshio::FeatureConstruct<MeshIO, FeatureBase>
{
    using type = typename FeatureBase::template add_features<MeshIO>::type;
};

} // namespace lvr2

#include "MeshIO.tcc"