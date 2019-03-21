//
// Created by patrick on 10.02.19.
//

//#include <lvr2/reconstruction/gs2/GrowingCellStructure.hpp>
#include <lvr2/util/Debug.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/config/BaseOption.hpp>
#include <cmath>

namespace lvr2 {

    template <typename BaseVecT, typename NormalT>
    GrowingCellStructure<BaseVecT, NormalT>::GrowingCellStructure(PointsetSurfacePtr<BaseVecT> surface)
    {
        m_surface = surface;
        m_mesh = 0;
    }



    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeBasicStep()
    {

        //TODO: get random point of the pointcloud

        auto pointer = m_surface.get()->pointBuffer();
        auto p_arr = pointer.get()->getPointArray(); //x,y,z,x,y,z .... 3*random, 3*random+2,3*random+3
        auto num_points = pointer.get()->numPoints();

        size_t random = rand() % num_points;

        BaseVecT random_point(p_arr[3 * random], p_arr[3 * random + 1], p_arr[3 * random + 2]);

        //TODO: search the closest point of the mesh

        auto vertices = m_mesh->vertices();

        VertexHandle closestVertexToRandomPoint(0);
        float smallestDistance = numeric_limits<float>::infinity();
        BaseVecT vectorToRandomPoint;

        for(auto vertexH : vertices)
        {
            BaseVecT& vertex = m_mesh->getVertexPosition(vertexH); //get Vertex from Handle

            BaseVecT distanceVector = random_point - vertex;
            float length = distanceVector.length();

            if(length < smallestDistance)
            {

                closestVertexToRandomPoint = vertexH;
                vectorToRandomPoint = distanceVector;
                smallestDistance = length;

            }
        }


        //TODO: smooth the winning vertex

        BaseVecT &winner = m_mesh->getVertexPosition(closestVertexToRandomPoint);
        winner += vectorToRandomPoint * getLearningRate();


        //TODO: smooth the winning vertices' neighbors (laplacian smoothing)

        vector<VertexHandle> neighborsOfWinner;
        m_mesh->getNeighboursOfVertex(closestVertexToRandomPoint, neighborsOfWinner);

        for(auto v : neighborsOfWinner)
        {
            BaseVecT& nb = m_mesh->getVertexPosition(v);
            nb += vectorToRandomPoint * getNeighborLearningRate();
        }

        //TODO: increase signal counter of winner by one

        winner.incSC();

        //TODO: decrease signal counter of others by a fraction

        for(auto vertexH : vertices){
            BaseVecT& vertex = m_mesh->getVertexPosition(vertexH);
            vertex.setSC(vertex.getSC() * 0.9);
        }

    }



    //TODO: Vertex split execution (For now only edge split)
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeVertexSplit()
    {
        //TODO: find vertex with highst sc, split that vertex

        auto vertices = m_mesh->vertices();

        VertexHandle highestSC(0);
        float maxSC = -1;
        for(auto vertexH : vertices)
        {
            BaseVecT& vertex = m_mesh->getVertexPosition(vertexH); //get Vertex from Handle

            if(vertex.signal_counter > maxSC )
            {

                highestSC = vertexH;
                maxSC = vertex.signal_counter;

            }
        }

        //TODO: split it.. :)
        VertexHandle newVH = m_mesh->splitVertex(highestSC);
        BaseVecT& newV = m_mesh->getVertexPosition(newVH);

        //TODO: reduce sc, set sc of newly added vertex
        BaseVecT& highestSCVec = m_mesh->getVertexPosition(highestSC);
        highestSCVec.signal_counter /= 2; //half of it..*/
        newV.signal_counter = highestSCVec.signal_counter;


    }

    //TODO: why does the second collapse (collapsable) fail?
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeEdgeCollapse()
    {
        //TODO: select edge to collapse, examine whether it should be collapsed, collapse it

        auto vertices = m_mesh->vertices();
        VertexHandle lowestSC(0);
        float minSC = numeric_limits<float>::infinity();
        for(auto vertexH : vertices)
        {
            BaseVecT& vertex = m_mesh->getVertexPosition(vertexH); //get Vertex from Handle

            if(vertex.signal_counter < minSC && vertex.signal_counter != 0)
            {

                lowestSC = vertexH;
                minSC = vertex.signal_counter;
            }
        }

        //found vertex with lowest sc
        //TODO: collapse the edge leading to the vertex with the valence closest to six
        if(minSC > this->getCollapseThreshold() && lowestSC.idx() != 0)
        {

            vector<VertexHandle> nbMinSc;
            m_mesh->getNeighboursOfVertex(lowestSC, nbMinSc);
            OptionalEdgeHandle eToSixVal(0);
            int difference = numeric_limits<int>::infinity();


            for(VertexHandle vertex : nbMinSc)
            {
                vector<VertexHandle> nbs;
                m_mesh->getNeighboursOfVertex(vertex, nbs);
                size_t length = nbs.size();

                if (abs((int) (6 - length)) < difference)
                {
                    difference = abs((int) (6 - length));
                    eToSixVal = m_mesh->getEdgeBetween(lowestSC, vertex);
                }
            }

            if(eToSixVal && m_mesh->isCollapsable(eToSixVal.unwrap()))
            {
                m_mesh->collapseEdge(eToSixVal.unwrap());
                std::cout << "Collapsed an Edge!" << endl;
            }

        }
    }



    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getInitialMesh(){
        auto bounding_box = m_surface.get()->getBoundingBox();

        if(!bounding_box.isValid())
        {
            std::cout << "Bounding Box invalid" << std::endl;
            exit(EXIT_FAILURE);
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
        BaseVecT left(BaseVecT(minx, miny, maxz));
        BaseVecT right(BaseVecT(maxx,miny,maxz));
        BaseVecT back(BaseVecT(centroid.x, miny, minz));


        std::cout << top << left << right << back << std::endl;

        auto vH1 = m_mesh->addVertex(top);
        auto vH2 = m_mesh->addVertex(left);
        auto vH3 = m_mesh->addVertex(right);
        auto vH4 = m_mesh->addVertex(back);

        //add faces to create tetrahedron
        m_mesh->addFace(vH4, vH3, vH2);
        m_mesh->addFace(vH4, vH2, vH1);
        m_mesh->addFace(vH4, vH1, vH3);
        m_mesh->addFace(vH3, vH1, vH2);
    }


    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getMesh(HalfEdgeMesh<BaseVecT> &mesh){

        //set pointer to mesh
        m_mesh = &mesh;

        //get initial tetrahedron mesh
        getInitialMesh();


        //TODO: add gcs construction.. call to basic step, call to other functions

        for(int i = 0; i < getRuntime(); i++){

            if(i == getRuntime() / 8) std::cout << "Achtel done!!" << endl;
            if(i == getRuntime() / 4) std::cout << "Viertel done!!" << endl;
            if(i == getRuntime() / 2) std::cout << "HALF done!!" << endl;
            for(int j = 0; j < getNumSplits(); j++){
                for(int k = 0; k < getBasicSteps(); k++){
                    executeBasicStep();
                }
                executeVertexSplit();

            }
            if(this->isWithCollapse()){
                //executeEdgeCollapse();
            }

        }
    }

}