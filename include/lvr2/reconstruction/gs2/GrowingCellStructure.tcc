//
// Created by patrick on 10.02.19.
//

#include <lvr2/util/Debug.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/config/BaseOption.hpp>
#include <cmath>
//#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/progress.hpp>

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
        auto p_arr = pointer.get()->getPointArray();
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
            performLaplacianSmoothing(v);

        }

        //TODO: increase signal counter of winner by one

        winner.incSC();

        //TODO: decrease signal counter of others by a fraction

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

        if(newVH.idx() != -1){
            BaseVecT& newV = m_mesh->getVertexPosition(newVH);

            //TODO: reduce sc, set sc of newly added vertex
            BaseVecT& highestSCVec = m_mesh->getVertexPosition(highestSC);
            highestSCVec.signal_counter /= 2; //half of it..*/
            newV.signal_counter = highestSCVec.signal_counter;

        }

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

            if(vertex.signal_counter < minSC)
            {
                lowestSC = vertexH;
                minSC = vertex.signal_counter;
            }
        }

        //found vertex with lowest sc
        //TODO: collapse the edge leading to the vertex with the valence closest to six
        if(minSC < this->getCollapseThreshold())
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
    void GrowingCellStructure<BaseVecT, NormalT>::initTestMesh(){
        VertexHandle v0 = m_mesh->addVertex(BaseVecT(3,0,0));
        VertexHandle v1 = m_mesh->addVertex(BaseVecT(0,3,0));
        VertexHandle v2 = m_mesh->addVertex(BaseVecT(6,3,0));
        VertexHandle v3 = m_mesh->addVertex(BaseVecT(0,6,0));
        VertexHandle v4 = m_mesh->addVertex(BaseVecT(6,6,0));
        VertexHandle v5 = m_mesh->addVertex(BaseVecT(0,9,0));
        VertexHandle v6 = m_mesh->addVertex(BaseVecT(6,9,0));
        VertexHandle v7 = m_mesh->addVertex(BaseVecT(3,12,0));
        VertexHandle v8 = m_mesh->addVertex(BaseVecT(3,7,0));

        m_mesh->addFace(v0,v8,v1);
        m_mesh->addFace(v0,v2,v8);
        m_mesh->addFace(v1,v8,v3);
        m_mesh->addFace(v2,v4,v8);
        m_mesh->addFace(v3,v8,v5);
        m_mesh->addFace(v4,v6,v8);
        m_mesh->addFace(v5,v8,v7);
        m_mesh->addFace(v6,v7,v8);

        m_mesh->splitEdgeNoRemove(m_mesh->getEdgeBetween(v8,v0).unwrap());
    };

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

        //add faces to create tetrahedron (interior/exterior)
        if(!isInterior())
        {
            m_mesh->addFace(vH4, vH3, vH2);
            m_mesh->addFace(vH4, vH2, vH1);
            m_mesh->addFace(vH4, vH1, vH3);
            m_mesh->addFace(vH3, vH1, vH2);
        }
        else{
            m_mesh->addFace(vH2, vH3, vH4);
            m_mesh->addFace(vH1, vH2, vH4);
            m_mesh->addFace(vH3, vH1, vH4);
            m_mesh->addFace(vH2, vH1, vH3);
        }
    }

    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::performLaplacianSmoothing(VertexHandle vertexH)
    {
        vector<VertexHandle> n_vertices = m_mesh->getNeighboursOfVertex(vertexH);
        BaseVecT& vertex = m_mesh->getVertexPosition(vertexH);
        BaseVecT avg_vec(0,0,0);

        for(VertexHandle vH : n_vertices)
        {
            BaseVecT v = m_mesh->getVertexPosition(vH);
            avg_vec += v - vertex;
        }

        avg_vec /= n_vertices.size();

        vertex += avg_vec * 0.01;
    }

    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::removeWrongFaces()
    {
        double avg_dist = 0;
        for(EdgeHandle eH: m_mesh->edges())
        {
            auto e_arr = m_mesh->getVerticesOfEdge(eH);
            BaseVecT v0 = m_mesh->getVertexPosition(e_arr[0]);
            BaseVecT v1 = m_mesh->getVertexPosition(e_arr[1]);
            avg_dist += v0.distance(v1);
        }

        avg_dist /= m_mesh->numEdges();

        for(VertexHandle vH : m_mesh->vertices())
        {
            auto edges = m_mesh->getEdgesOfVertex(vH);
            bool removable = true;
            for(EdgeHandle eH : edges)
            {
                auto e_arr = m_mesh->getVerticesOfEdge(eH);
                BaseVecT v0 = m_mesh->getVertexPosition(e_arr[0]);
                BaseVecT v1 = m_mesh->getVertexPosition(e_arr[1]);
                double dist = v0.distance(v1);
                if(dist <= 3 * avg_dist) {
                    removable = false;
                }
            }
            if(removable)
            {
                auto faces = m_mesh->getFacesOfVertex(vH);
                for(auto face : faces)
                {
                    m_mesh->removeFace(face);
                }
            }
        }
    }


    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getMesh(HalfEdgeMesh<BaseVecT> &mesh){

        //set pointer to mesh
        m_mesh = &mesh;

        //get initial tetrahedron mesh
        getInitialMesh();
        //initTestMesh();

        //TODO: add some progress...needs to include the fact, that the runtime of the algorithm is exponential (boost progress display)

        //TODO: add gcs construction.. call to basic step, call to other functions

        boost::progress_display show_progress((unsigned long)( m_runtime ));

        for(int i = 0; i < getRuntime(); i++){
            ++show_progress;
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

        //removeWrongFaces();

    }

}