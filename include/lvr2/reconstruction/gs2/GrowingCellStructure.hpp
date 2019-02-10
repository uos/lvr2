//
// Created by patrick on 10.02.19.
//

#ifndef LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
#define LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP

//#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/config/BaseOption.hpp>

namespace lvr2{

    template <typename BaseVecT, typename NormalT>
    class GrowingCellStructure {
    public:
        GrowingCellStructure(PointsetSurfacePtr<BaseVecT> surface, BaseOption &option){
            //m_surface = surface;
        }

    private:
        PointsetSurfacePtr<BaseVecT> m_surface;
    };
}




#endif //LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
