//
// Created by patrick on 10.02.19.
//

#include <lvr2/util/Debug.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/io/Progress.hpp>

#include <cmath>


namespace lvr2 {


    // SHARED METHODS - methods both of the algorithms use, or methods, which may or may not use the gss extentions

    /**
     * Constructor setting the references to the instance of the surface storing the pointcloud
     *
     * @tparam NormalT
     * @param surface - reference to the surface storing the pointcloud
     */
    template <typename BaseVecT, typename NormalT>
    GrowingCellStructure<BaseVecT, NormalT>::GrowingCellStructure(PointsetSurfacePtr<BaseVecT>& surface)
    {
        m_surface = &surface;
        m_mesh = 0;
        tumble_tree = new TumbleTree(); //create tumble tree
        kd_tree = new DynamicKDTree<BaseVecT>(3); // create 3-dimensional kd-tree for distance evaluation
    }

    /**
     * Only public method, central remote for other function calls
     *
     * @tparam BaseVecT - the vector type used, needs an signal counter for GCS
     * @tparam NormalT - the normal type used, usually the default type is enough
     * @param mesh - reference to an halfedge mesh, which will contain the reconstruction afterwards
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getMesh(HalfEdgeMesh<BaseVecT> &mesh){

        //set pointer to mesh
        m_mesh = &mesh;

        //get initial tetrahedron mesh
        //getInitialMesh();
        cout << "KD-Tree size: " << kd_tree->size() << endl;

        initTestMesh();

        //progress bar
        PacmanProgressBar progress_bar((size_t)((((size_t)m_runtime*(size_t)m_numSplits)
                                     *(((size_t)m_numSplits*(size_t)m_runtime)+1)/(size_t)2) * (size_t)m_basicSteps));

        //algorithm
        for(int i = 0; i < getRuntime(); i++)
        {
            for(int j = 0; j < getNumSplits(); j++)
            {
                for(int k = 0; k < getBasicSteps(); k++)
                {
                    //executeBasicStep(progress_bar);
                }
                //executeVertexSplit(); //TODO: execute vertex split after a specific number of basic steps

            }
            if(this->isWithCollapse())
            {
                //executeEdgeCollapse(); //TODO: execute an edge collapse, only if the user specified so
            }

        }

        //final operations on the mesh (like removing wrong faces and filling the holes)

        if(m_mesh->numVertices() > 5000)
        {
           removeWrongFaces(); //removes faces which area is way bigger (3 times) than the average
        }

        //tumble_tree->display();

        cout << "Tumble Tree size: " << tumble_tree->size() << endl;
        cout << "KD-Tree size: " << kd_tree->size() << endl;
        cout << "VertexCell map size: " << vertexCellMap.numValues() << endl;

        delete tumble_tree;
    }


    /**
     * executes the basic step of the algorithm, getting a random point, finding the closest point of the mesh,
     * expanding and smoothing the neighbours (GCS).
     *
     * finds closest structure of the mesh, expands the mesh, smoothes neighbours (GSS)
     *
     * @tparam BaseVecT
     * @tparam NormalT
     * @param progress_bar needed to pass it to the getClosestPointInMesh operation
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeBasicStep(PacmanProgressBar& progress_bar)
    {
        //get random point of the pointcloud
        BaseVecT random_point = this->getRandomPointFromPointcloud();

        if(!m_useGSS) //if only gcs is used (gcs basic step)
        {
            VertexHandle winnerH = this->getClosestPointInMesh(random_point, progress_bar); //TODO: better runtime efficency(kd-tree)

            //smooth the winning vertex
            BaseVecT &winner = m_mesh->getVertexPosition(winnerH);
            //kd_tree->deleteNode(winner);
            winner += (random_point - winner) * getLearningRate();
            //kd_tree->insert(winner, winnerH);

            //smooth the winning vertices' neighbors (laplacian smoothing)

            vector<VertexHandle> neighborsOfWinner;
            m_mesh->getNeighboursOfVertex(winnerH, neighborsOfWinner);

            //perform laplacian smoothing on all the neighbors of the winning vertex
            for(auto v : neighborsOfWinner)
            {
                BaseVecT& nb = m_mesh->getVertexPosition(v);
                //kd_tree->deleteNode(nb);
                nb += (random_point - winner) * getNeighborLearningRate();

                performLaplacianSmoothing(v);
                //kd_tree->insert(nb, v);
            }

            //increase signal counter by one

            //TODO: use map and tumble tree correctly
            Cell* winnerNode = vertexCellMap.get(winnerH).get();
            float winnerSC = winnerNode->signal_counter; //obtain the signal counter from the map

            tumble_tree->remove(winnerNode, winnerH); //remove the winning vertex from the tumble tree

            //TODO: decrease signal counter of others by a fraction according to hennings implementation
            if(m_decreaseFactor == 1.0)
            {
                size_t n = m_allowMiss * m_mesh->numVertices();
                float dynamicDecrease = 1 - (float)pow(m_collapseThreshold, (1.0 / n));
                //tumble_tree->updateSC(dynamicDecrease, winnerH);

            }
            else
            {
                //tumble_tree->updateSC(m_decreaseFactor, winnerH);

            }

            vertexCellMap.get(winnerH).get() = tumble_tree->insertIterative(winnerSC+1, winnerH); //reinsert it with incremented sc, update map



        }
        else //GSS
        {
            std::cout << "Using GSS" << endl;
            //find closest structure

            //set approx error(s) and age of faces (using HashMap)

            //smoothing

            //coalescing

            //filter chain
        }
    }


    /**
     * Performs an vertex split operation on the vertex with the highest signal counter, reduces signal counters by half (GCS)
     *
     * Performs vertex split on the longest edge of the face with the highest approx error, splits approx errors, reduces age (GSS)
     *
     * @tparam BaseVecT
     * @tparam NormalT
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeVertexSplit()
    {
        if(!m_useGSS) //GCS
        {
            //find vertex with highst sc, split that vertex
            Cell* max = tumble_tree->max();
            //cout << "Max C: " << max->signal_counter;
            auto iter = max->duplicateMap.begin();
            VertexHandle highestSC = *iter;

            //split the found vertex
            VertexSplitResult result = m_mesh->splitVertex(highestSC);
            if(result.edgeCenter.idx() == numeric_limits<int>::infinity()) return; //if longest edge is a border edge
            VertexHandle newVH = result.edgeCenter;
            float sc_middle = max->signal_counter / 2;

            //now update tumble tree and the map
            tumble_tree->remove(max, highestSC);
            vertexCellMap.get(highestSC).get() = tumble_tree->insertIterative(sc_middle, highestSC);//reinsert and update links
            vertexCellMap.insert(newVH, tumble_tree->insertIterative(sc_middle, newVH)); //add the new vertex to the tree and the map

            //BaseVecT kdInsert = m_mesh->getVertexPosition(newVH);
            //kd_tree->insert(kdInsert, newVH);

        }
        else //GSS
        {
            //select triangle with highst err
            FaceHandle errorFaceH(0);
            float max_err = 0;
            for(auto faceH : m_mesh->faces())
            {
                float err = faceAgeErrorMap.get(faceH).value().second;
                if(err > max_err)
                {
                    errorFaceH = faceH;
                    max_err = err;
                }
            }

            //split vertex
            EdgeHandle longestEdgeH(0);
            float max_len = 0;
            for(EdgeHandle eH : m_mesh->getEdgesOfFace(errorFaceH))
            {
                auto vertices = m_mesh->getVerticesOfEdge(eH);
                BaseVecT v1 = m_mesh->getVertexPosition(vertices[0]);
                BaseVecT v2 = m_mesh->getVertexPosition(vertices[1]);

                float len = v1.distance2(v2);
                if(len > max_len)
                {
                    max_len = len;
                    longestEdgeH = eH;
                }
            }

            // TODO: after finding the longest edge.... we need a way to split it.. vertexSplit(VertexHandle vH) not enough

            //reset age to initial age

            ////?????????????


            //TODO: filter chain ( not yet programmed )
        }


    }


    /**
     * Performs an edge collapse on the vertex with the smallest signal counter(GCS)
     *
     * Performs an edge collapse on geometrically redundant edge from the face with the lowest approx error,
     * removes faces which exceed a certain age (GSS)
     *
     * @tparam BaseVecT
     * @tparam NormalT
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::executeEdgeCollapse()
    {
        //TODO: select edge to collapse, examine whether it should be collapsed, collapse it
        //TODO: tumble tree support
        if(!m_useGSS)
        {
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
        else
        {
            //select triangle with lowest err

            //perform edge collapse on geometrically redundant edge e

            //select triangle with highst age

            //age too high? remove face

            //filter chain
        }
    }


    /**
     * Gets a random point from the Pointcloud stored in m_surface
     * runtime: O(1)
     *
     * @tparam BaseVecT
     * @tparam NormalT
     * @return returns the random point
     */
    template <typename BaseVecT, typename NormalT>
    BaseVecT GrowingCellStructure<BaseVecT, NormalT>::getRandomPointFromPointcloud(){
        auto pointer = m_surface->get()->pointBuffer();
        auto p_arr = pointer.get()->getPointArray();
        auto num_points = pointer.get()->numPoints();

        size_t random = rand() % num_points; //random number

        BaseVecT random_point(p_arr[3 * random], p_arr[3 * random + 1], p_arr[3 * random + 2]);
        return random_point;
    }


    /**
     * Gets the closest point to the given point using the euclidean distance
     * runtime: O(n) //TODO: make it better. kd-tree?
     *
     * @tparam BaseVecT
     * @tparam NormalT
     * @param point - point of the pointcloud
     * @param progress_bar - needed to print the progress of the algorithm
     * @return a handle pointing to the closest point of the mesh to the point in the paramters
     */
    template <typename BaseVecT, typename NormalT>
    VertexHandle GrowingCellStructure<BaseVecT, NormalT>::getClosestPointInMesh(BaseVecT point, PacmanProgressBar& progress_bar)
    {
        //search the closest point of the mesh
        auto vertices = m_mesh->vertices();

        VertexHandle closestVertexToRandomPoint(numeric_limits<int>::max());
        float smallestDistance = numeric_limits<float>::infinity();
        float avg_counter = 0;

        for(auto vertexH : vertices)
        {
            ++progress_bar;
            BaseVecT& vertex = m_mesh->getVertexPosition(vertexH); //get Vertex from Handle
            //avg_counter += vertex.signal_counter; //calc the avg signal counter
            BaseVecT distanceVector = point - vertex;
            float length = distanceVector.length2();

            if(length < smallestDistance)
            {

                closestVertexToRandomPoint = vertexH;
                smallestDistance = length;

            }
        }

        //m_avgSignalCounter = avg_counter / m_mesh->numVertices();

        return closestVertexToRandomPoint;
    };

    /**
     * Test Method creating a mesh with 9 vertices. Used for testing vertex and edge split operations
     *
     * @tparam BaseVecT
     * @tparam NormalT
     */
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

        EdgeSplitResult result = m_mesh->splitEdgeNoRemove(m_mesh->getEdgeBetween(v8,v0).unwrap());
        m_mesh->flipEdge(m_mesh->getEdgeBetween(v2,v8).unwrap());
        m_mesh->flipEdge(m_mesh->getEdgeBetween(v8,v1).unwrap());
        std::cout << "Vertex(center): " << m_mesh->getVertexPosition(v8) << endl;
        std::cout << "Vertex(center): " << m_mesh->getVertexPosition(v2) << endl;
        std::cout << "Vertex(center): " << m_mesh->getVertexPosition(v1) << endl;
    }

    /**
     * Constructs the initial tetrahedron mesh, scales it and places it in the middle of the pointcloud
     *
     * @tparam BaseVecT - vector type used
     * @tparam BaseVecT - vector type used
     * @tparam NormalT - normal type used
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::getInitialMesh(){
        auto bounding_box = m_surface->get()->getBoundingBox();

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


        auto vH1 = m_mesh->addVertex(top);
        auto vH2 = m_mesh->addVertex(left);
        auto vH3 = m_mesh->addVertex(right);
        auto vH4 = m_mesh->addVertex(back);

        FaceHandle fH1(0);
        FaceHandle fH2(0);
        FaceHandle fH3(0);
        FaceHandle fH4(0);
        //add faces to create tetrahedron (interior/exterior)
        if(!isInterior())
        {
            fH1 = m_mesh->addFace(vH4, vH3, vH2);
            fH2 = m_mesh->addFace(vH4, vH2, vH1);
            fH3 = m_mesh->addFace(vH4, vH1, vH3);
            fH4 = m_mesh->addFace(vH3, vH1, vH2);
        }
        else{
            fH1 = m_mesh->addFace(vH2, vH3, vH4);
            fH2 = m_mesh->addFace(vH1, vH2, vH4);
            fH3 = m_mesh->addFace(vH3, vH1, vH4);
            fH4 = m_mesh->addFace(vH2, vH1, vH3);
        }

        //add faces to the hashmap with no error and zero age
        if(m_useGSS)
        {
            faceAgeErrorMap.insert(fH1, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH2, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH3, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH4, std::make_pair(0.0f, 0.0f));
        }
        else
        {
            //insert vertices to the hashmap as well as the tumbletree
            vertexCellMap.insert(vH1,tumble_tree->insertIterative(1, vH1));
            vertexCellMap.insert(vH2,tumble_tree->insertIterative(1, vH2));
            vertexCellMap.insert(vH3,tumble_tree->insertIterative(1, vH3));
            vertexCellMap.insert(vH4,tumble_tree->insertIterative(1, vH4));
            VertexHandle ret(numeric_limits<int>::max());
            tumble_tree->insertIterative(10.00001f, ret); //dummy root

            kd_tree->insert(top, vH1);
            kd_tree->insert(left, vH2);
            kd_tree->insert(right, vH3);
            kd_tree->insert(back, vH4);
            /*tumble_tree->remove(1, vH1);
            tumble_tree->insertIterative(2,vH1);
            tumble_tree->display();

            VertexHandle v1(5);
            VertexHandle v2(6);
            VertexHandle v3(7);
            VertexHandle v4(8);
            VertexHandle v5(9);
            tumble_tree->insertIterative(3,v1);
            tumble_tree->insertIterative(0.5,v2);
            tumble_tree->insertIterative(2,v3);
            tumble_tree->insertIterative(22,v4);
            tumble_tree->insertIterative(9,v5);
            tumble_tree->display();
            tumble_tree->remove(2,vH1);
            tumble_tree->remove(2,v3);*/
            tumble_tree->display();
            std::cout << "tt init size: " << tumble_tree->size() << endl;
        }
    }


    /**
     * Performs a laplacian smoothing operation on the vertex given as paramter
     *
     * @tparam BaseVecT - the vector type used
     * @tparam NormalT - the normal type used
     * @param vertexH - vertex, which will be smoothed
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::performLaplacianSmoothing(VertexHandle vertexH)
    {
        vector<VertexHandle> n_vertices = m_mesh->getNeighboursOfVertex(vertexH);
        BaseVecT& vertex = m_mesh->getVertexPosition(vertexH);
        BaseVecT avg_vec(0,0,0);

        for(VertexHandle vH : n_vertices)
        {
            BaseVecT v = m_mesh->getVertexPosition(vH);
            avg_vec += (v - vertex) ;
        }

        avg_vec /= n_vertices.size();

        vertex += avg_vec * 0.01/*getNeighborLearningRate()*/;
    }

    // GCS METHODS - Methods which are only used by the GCS-algorithm


    // GSS METHODS - Methods, which are used by the GSS extention of the growing algorithm,


    //Todo: find closest structure, move structure and neighbors

    //Todo: Coalescing (bridge building) - might make sense for gcs as well

    //Todo: filter chain - which algorithms are implemented in the halfedge mesh?

    // OTHER METHODS - Methods, which don't really belong to both of the algorithms, but are useful for the reconstruction

    /**
     * WIP: Method removing faces, which were inserted due to the nature of the algorithm, but dont belong to the reconstruction
     *
     * @tparam BaseVecT - vector type used
     * @tparam NormalT - normal type used
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::removeWrongFaces()
    {

        double avg_area = 0;

        for(FaceHandle face: m_mesh->faces())
        {
            avg_area += m_mesh->calcFaceArea(face);
        }

        avg_area /= m_mesh->numFaces();

        for(FaceHandle face: m_mesh->faces())
        {
            double area = m_mesh->calcFaceArea(face);
            if(area > 5 * avg_area)
            {
                m_mesh->removeFace(face);
            }
        }


        //now remove faces with one ore less neighbour faces, as those become redundant as well.
        /*for(FaceHandle face: m_mesh->faces())
        {
            vector<FaceHandle> n_faces;
            m_mesh->getNeighboursOfFace(face, n_faces);
            if(n_faces.size() == 0)
            {
                m_mesh->removeFace(face);
                //removeWrongFaces(); //maybe too long runtime on this one...
            }
        }*/

    }

}