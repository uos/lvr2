//
// Created by patrick on 10.02.19.
//

#include <lvr2/reconstruction/gs2/GrowingCellStructure.hpp>
#include <lvr2/util/Debug.hpp>

namespace lvr2 {

    /*template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeBasicStep(){

    }

    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeVertexSplit(HalfEdgeHandle handle) {

    }

    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeEdgeCollapse(VertexHandle handle) {

    }*/


    /*template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getInitialMesh(HalfEdgeMesh<BaseVecT> &mesh){
        auto bounding_box = m_surface.get()->getBoundingBox();

        Vector<BaseVecT> centroid = bounding_box.getCentroid();
        Vector<BaseVecT> min = bounding_box.getMin();
        Vector<BaseVecT> max = bounding_box.getMax();

        float xdiff = (max.x - min.x) / 2;
        float ydiff = (max.y - max.x) / 2;
        float zdiff = (max.z - min.z) / 2;

        //scale diff acc to the box factor
        xdiff *= (1 - m_boxFactor);
        ydiff *= (1 - m_boxFactor);
        zdiff *= (1 - m_boxFactor);

        float minx, miny, minz, maxx, maxy, maxz;
        minx = min.x + xdiff;
        miny = min.y + ydiff;
        minz = min.z + zdiff;
        maxx = max.x - xdiff;
        maxy = max.y - ydiff;
        maxz = max.z - zdiff;

        Vector<BaseVecT> top(BaseVecT(centroid.x, maxy, centroid.z));
        Vector<BaseVecT> left(BaseVecT(minx, miny, minz));
        Vector<BaseVecT> right(BaseVecT(maxx,miny,minz));
        Vector<BaseVecT> back(BaseVecT(centroid.x, miny, maxz));

        mesh.addVertex(top);
        mesh.addVertex(left);
        mesh.addVertex(right);
        mesh.addVertex(back);

        mesh.addFace(top, left, right);
        mesh.addFace(top, left, back);
        mesh.addFace(top,right,back);
        mesh.addFace(left, right, back);
    }*/


    /*template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getMesh(HalfEdgeMesh<BaseVecT> &mesh){
        getInitialMesh(mesh);
    }*/

}