//
// Created by patrick on 10.02.19.
//

//#include <lvr2/reconstruction/gs2/GrowingCellStructure.hpp>
#include <lvr2/util/Debug.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/config/BaseOption.hpp>

namespace lvr2 {

    template <typename BaseVecT, typename NormalT>
    GrowingCellStructure<BaseVecT, NormalT>::GrowingCellStructure(PointsetSurfacePtr<BaseVecT> surface){
        m_surface = surface;
        m_mesh = 0;
    }



    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeBasicStep(){

        //TODO: get random point of the pointcloud

        auto pointer = m_surface.get()->pointBuffer();
        auto p_arr = pointer.get()->getPointArray(); //x,y,z,x,y,z .... 3*random, 3*random+2,3*random+3
        auto num_points = pointer.get()->numPoints();

        size_t random = rand() % num_points;

        BaseVecT random_point(p_arr[3 * random], p_arr[3 * random + 1], p_arr[3 * random + 2]);

        std::cout << random_point.x << "|" << random_point.y << "|"<< random_point.z << std::endl;

        //TODO: search the closest point of the mesh



        //TODO: smooth the winning vertex

        //TODO: smooth the winning vertices' neighbors (laplacian smoothing)

        //TODO: increase signal counter of winner by one

        //TODO: decrease signal counter of others by a fraction

    }



    //TODO: Vertex split execution
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeVertexSplit() {
        //TODO: find vertex with highst sc, split that vertex


    }



    //TODO: EDGECOLLAPSE execution
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeEdgeCollapse(VertexHandle handle){

        //TODO: select edge to collapse, examine whether it should be collapsed, collapse it
    }






    template <typename BaseVecT, typename NormalT>
    VertexHandle GrowingCellStructure<BaseVecT, NormalT>::getInitialMesh(HalfEdgeMesh<BaseVecT> &mesh){
        auto bounding_box = m_surface.get()->getBoundingBox();

        if(!bounding_box.isValid()){
            std::cout << "Bounding Box invalid" << std::endl;
            exit(-1);
        }

        BaseVecT centroid = bounding_box.getCentroid();
        BaseVecT min = bounding_box.getMin();
        BaseVecT max = bounding_box.getMax();

        float xdiff = (max.x - min.x) / 2;
        float ydiff = (max.y - min.y) / 2;
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

        BaseVecT top(BaseVecT(centroid.x, maxy, centroid.z));
        BaseVecT left(BaseVecT(minx, miny, minz));
        BaseVecT right(BaseVecT(maxx,miny,minz));
        BaseVecT back(BaseVecT(centroid.x, miny, maxz));


        std::cout << top << left << right << back << std::endl;

        auto vH1 = mesh.addVertex(top);
        auto vH2 = mesh.addVertex(left);
        auto vH3 = mesh.addVertex(right);
        auto vH4 = mesh.addVertex(back);

        //add faces to create tetrahedron
        mesh.addFace(vH2, vH3, vH4);
        mesh.addFace(vH1, vH2, vH4);
        mesh.addFace(vH1, vH4, vH3);
        mesh.addFace(vH3, vH2, vH1);

        //initial mesh done, doesnt need handle-return

        return vH1;
    }


    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getMesh(HalfEdgeMesh<BaseVecT> &mesh){

        m_mesh = mesh;

        VertexHandle test = getInitialMesh(m_mesh);
        mesh.splitGSVertex(test); //test

        //TODO: add gcs construction.. call to basic step, call to other functions
        // for { for { for { basicStep() } vertexSplit() } edgeCollapse()}

        for(int i = 0; i < getRuntime(); i++){
            executeBasicStep();
            if(i % getBasicSteps() == 0){
                executeVertexSplit();

                std::cout << "Vertex Split!!" << std::endl;
            }
        }
    }

}