//
// Created by patrick on 10.02.19.
//

#include "lvr2/util/Debug.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/reconstruction/PointsetSurface.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/reconstruction/LBKdTree.hpp"
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
        int max_depth = 0;
        //initialize cell array.
        std::vector<Cell*>::size_type size = (unsigned long)(m_runtime*m_numSplits+4);
        cellArr.resize(size, NULL);

        //initTestMesh(); //init a mesh used for vertex split and edge split testing

        //get initial tetrahedron mesh
        getInitialMesh();

        //progress bar
        size_t runtime_length = (size_t)((((size_t)m_runtime*(size_t)m_numSplits)
                                          *(((size_t)m_numSplits*(size_t)m_runtime)+1)/(size_t)2) * (size_t)m_basicSteps);
        PacmanProgressBar progress_bar(runtime_length);

        //algorithm
        for(int i = 0; i < getRuntime(); i++)
        {
            if(getRuntime() >= m_balances && m_balances > 0 && i % (getRuntime() / m_balances) == 0 )
            {
                int tmp = tumble_tree->maxDepth();
                if(tmp > max_depth) max_depth = tmp;
                tumble_tree->balance();
            }

            for(int j = 0; j < getNumSplits(); j++)
            {
                for(int k = 0; k < getBasicSteps(); k++)
                {
                    executeBasicStep(progress_bar);
                }
                executeVertexSplit(); //TODO: execute vertex split after a specific number of basic steps

            }
            if(this->isWithCollapse())
            {
                //executeEdgeCollapse(); //TODO: execute an edge collapse, only if the user specified so
            }

        }

        //final operations on the mesh (like removing wrong faces and filling the holes)

        /*int counter = 0;
        for(int i = 0; i < cellArr.size(); i++)
        {
            VertexHandle vH(i);
            if(cellArr[i] != NULL && tumble_tree->find(cellArr[i]->signal_counter, vH) == NULL)
            {
                counter ++;
            }
        }*/

        if(m_mesh->numVertices() > 2000)
        {
           removeWrongFaces(); //removes faces which area are way bigger (3 times) than the average
        }

        for(auto vertex : m_mesh->vertices())
        {
            performLaplacianSmoothing(vertex, m_mesh->getVertexPosition(vertex), 0.5); //no random point influence
        }


        cout << "Max depth of tt: " << (m_balances != 0 ? max_depth : tumble_tree->maxDepth()) << endl;
        cout << "Not Deleted in TT: " << tumble_tree->notDeleted << endl;
        cout << "Tumble Tree size: " << tumble_tree->size() << endl;
        cout << "KD-Tree size: " << kd_tree->size() << endl;
        cout << "Cell array size: " << cellVecSize() << endl;
        cout << "Not found counter: " << notFoundCounter << endl;
        cout << endl;
        cout << "Equilaterality test percentage: " << equilaterality().second << endl;
        cout << "Skewness test percentage: " << equilaterality().first << endl;
        cout << "Average Valence: " << avgValence() << endl;

        cout << "Valances >= 10: " << numVertexValences(10) << endl;
        cout << "Valances >= 15: " << numVertexValences(15) << endl;
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
        //cout << "basic step" << endl;
        if(!m_useGSS) //if only gcs is used (gcs basic step)
        {
            VertexHandle winnerH = this->getClosestPointInMesh(random_point, progress_bar); //TODO: better runtime efficency(kd-tree)
            /*Index winnerIndex = kd_tree->findNearest(random_point);
            VertexHandle winnerH(winnerIndex);*/

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
                if(m_mesh->numVertices() > 100) performLaplacianSmoothing(v, random_point, getNeighborLearningRate());

                //kd_tree->insert(nb, v);
            }


            Cell* winnerNode = cellArr[winnerH.idx()];

            //TODO: determine mistake in remove operation in basic step. why on earth is there a prob here

            //we need to remove the winner before updating.
            double winnerSC = tumble_tree->remove(winnerNode, winnerH); //remove the winning vertex from the tumble tree, get the real sc

            //decrease signal counter of others by a fraction according to hennings implementation
            if(m_decreaseFactor == 1.0)
            {
                size_t n = m_allowMiss * m_mesh->numVertices();
                float dynamicDecrease = 1 - (float)pow(m_collapseThreshold, (1 / n));
                tumble_tree->updateSC(dynamicDecrease);

            }
            else
            {
                tumble_tree->updateSC(m_decreaseFactor);

            }
            //reinsert the winner's vH with updated sc
            cellArr[winnerH.idx()] = tumble_tree->insert(winnerSC + 1, winnerH);

        }
        else //GSS TODO: INCLUDE GSS ADDITIONS
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
        //cout << "Vertex Split" << endl;
        if(!m_useGSS) //GCS
        {
            //find vertex with highst sc, split that vertex
            Cell* max = tumble_tree->max();

            auto iter = max->duplicateMap.begin(); //get the first vertexhandle, if there are more then one
            VertexHandle highestSC = *iter; //get the first of the duplicate map


            //split the found vertex
            VertexSplitResult result = m_mesh->splitVertex(highestSC);
            VertexHandle newVH = result.edgeCenter; //obtain the vertex newly added to the mesh

            //now update tumble tree and the cell array
            double actual_sc = tumble_tree->remove(max, highestSC);
            cellArr[highestSC.idx()] = tumble_tree->insert(actual_sc / 2, highestSC);
            cellArr[newVH.idx()] = tumble_tree->insert(actual_sc / 2, newVH);


            /*BaseVecT kdInsert = m_mesh->getVertexPosition(newVH);
            kd_tree->insert(kdInsert, newVH);*/

        }
        else //GSS TODO: INCLUDE GSS ADDITIONS
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

            //TODO: after finding the longest edge.... we need a way to split it.. vertexSplit(VertexHandle vH) not enough

            //reset age to initial age

            //TODO: filter chain ( not yet programmed )
        }


    }


    /**
     * GCS: Performs an edge collapse on the vertex with the smallest signal counter(GCS)
     *
     * GSS: Performs an edge collapse on geometrically redundant edge from the face with the lowest approx error,
     * removes faces which exceed a certain age
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
            Cell* min = tumble_tree->min();
            auto iter = min->duplicateMap.begin();
            VertexHandle lowestSC = *iter;

            if(m_mesh->getNeighboursOfVertex(lowestSC).size() > 50)
            {
                aggressiveCutOut(lowestSC); //cut out the lowest sc if its valence is too high (GSS).
            }
            else
            {

                cout << "Lowest SC from Tumble Tree: " << min->signal_counter << " | " << (*min->duplicateMap.begin()).idx() << endl;
                cout << "Colapse threshold: " << m_collapseThreshold << endl;
                //found vertex with lowest sc
                //TODO: collapse the edge leading to the vertex with the valence closest to six
                if(min->signal_counter < this->getCollapseThreshold())
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
                        //TODO: use collapse result to remove the (vertex) from the tumble tree and the kd tree
                        //otherwise this wont work
                        EdgeCollapseResult result = m_mesh->collapseEdge(eToSixVal.unwrap());
                        tumble_tree->remove(cellArr[result.removedPoint.idx()], result.removedPoint);
                        cellArr[result.removedPoint.idx()] = NULL;
                        std::cout << "Collapsed an Edge!" << endl;
                    }
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
     * runtime: O(n) //TODO: make it better. kd-tree? KD-TREE IMPLEMENTATION IN GS2 NOT GOOD.
     *
     * @tparam BaseVecT
     * @tparam NormalT
     * @param point - point of the pointcloud
     * @param progress_bar - needed to print the progress of the algorithm
     * @return a handle pointing to the closest point of the mesh to the point in the parameters
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
            /*if(m_mesh->numVertices() != 4) TODO: PRINT NUMBER OF VERTICES WHILE ALSO PRINTING THE PROGRESS BAR...
            {
                cout << "\33[2K\r" << endl;
            }*/
            ++progress_bar;
            //cout << "Vertices in Mesh: " << m_mesh->numVertices() << endl;

            BaseVecT vertex = m_mesh->getVertexPosition(vertexH); //get Vertex from Handle
            BaseVecT distanceVector = point - vertex;
            float length = distanceVector.length2();

            if(length < smallestDistance)
            {

                closestVertexToRandomPoint = vertexH;
                smallestDistance = length;

            }
        }

        return closestVertexToRandomPoint;
    }

    /**
     * Test Method creating a mesh with 9 vertices. Used for testing vertex and edge split operations,
     * also tests the tumble tree
     *
     * @tparam BaseVecT
     * @tparam NormalT
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::initTestMesh(){
        VertexHandle v0 = m_mesh->addVertex(BaseVecT(3,0,0));
        VertexHandle v1 = m_mesh->addVertex(BaseVecT(0,3,0));
        VertexHandle v2 = m_mesh->addVertex(BaseVecT(6,3,0));
        VertexHandle v3 = m_mesh->addVertex(BaseVecT(-1.5,6,0));
        VertexHandle v4 = m_mesh->addVertex(BaseVecT(7.5,6,0));
        VertexHandle v5 = m_mesh->addVertex(BaseVecT(0,9,0));
        VertexHandle v6 = m_mesh->addVertex(BaseVecT(6,9,0));
        VertexHandle v7 = m_mesh->addVertex(BaseVecT(3,12,0));
        VertexHandle v8 = m_mesh->addVertex(BaseVecT(3,7,0));

        FaceHandle fH1 = m_mesh->addFace(v0,v8,v1);
        FaceHandle fH2 = m_mesh->addFace(v0,v2,v8);
        FaceHandle fH3 = m_mesh->addFace(v1,v8,v3);
        FaceHandle fH4 = m_mesh->addFace(v2,v4,v8);
        FaceHandle fH5 = m_mesh->addFace(v3,v8,v5);
        FaceHandle fH6 = m_mesh->addFace(v4,v6,v8);
        FaceHandle fH7 = m_mesh->addFace(v5,v8,v7);
        FaceHandle fH8 = m_mesh->addFace(v6,v7,v8);

        auto pair = m_mesh->triCircumCenter(fH1);
        std::cout << "CircumCenter1: " << pair.first << "| Radius: " << pair.second << endl;
        auto pair1 = m_mesh->triCircumCenter(fH2);
        std::cout << "CircumCenter1: " << pair1.first << "| Radius: " << pair1.second << endl;
        auto pair2 = m_mesh->triCircumCenter(fH3);
        std::cout << "CircumCenter1: " << pair2.first << "| Radius: " << pair2.second << endl;
        auto pair3 = m_mesh->triCircumCenter(fH4);
        std::cout << "CircumCenter1: " << pair3.first << "| Radius: " << pair3.second << endl;
        auto pair4 = m_mesh->triCircumCenter(fH5);
        std::cout << "CircumCenter1: " << pair4.first << "| Radius: " << pair4.second << endl;

        //m_mesh->splitVertex(v8);
        //m_mesh->splitVertex(v8);


        Cell* c1 = tumble_tree->insert(5,v0);
        Cell* c2 = tumble_tree->insert(2,v1);
        Cell* c3 = tumble_tree->insert(10,v2);
        Cell* c4 = tumble_tree->insert(1,v3);
        Cell* c5 = tumble_tree->insert(3,v4);
        Cell* c6 = tumble_tree->insert(7,v6);
        Cell* c7 = tumble_tree->insert(12,v7);
        Cell* c8 = tumble_tree->insert(2.5,v5);
        Cell* c9 = tumble_tree->insert(2.5,v8);

        tumble_tree->display();
        /*tumble_tree->updateSC(0.9);
        tumble_tree->display();*/

        double sc = tumble_tree->remove(c1, v0);
        tumble_tree->insert(sc + 1,v0);
        tumble_tree->display();
        sc = tumble_tree->remove(c9, v5);
        tumble_tree->insert(sc+1,v5);
        tumble_tree->insert(sc+1,v5);
        tumble_tree->display();
        sc = tumble_tree->remove(c4, v3);
        tumble_tree->insert(sc+1,v3);
        tumble_tree->display();
        sc = tumble_tree->remove(c5, v4);
        tumble_tree->insert(sc+1,v4);
        tumble_tree->display();
        sc = tumble_tree->remove(c3, v2);
        tumble_tree->display();

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

        cout << "Bounding Box min: " << min << endl;
        cout << "Bounding Box max: " << max << endl;

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

        cout << vH1 << " | " << vH2 << " | " << vH3 << " | " << vH4 << endl;

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
        if(m_useGSS) //GSS
        {
            faceAgeErrorMap.insert(fH1, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH2, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH3, std::make_pair(0.0f, 0.0f));
            faceAgeErrorMap.insert(fH4, std::make_pair(0.0f, 0.0f));
        }
        else
        {
            //insert vertices to the cellindexarray as well as the tumbletree

            cellArr[vH1.idx()] = tumble_tree->insert(1, vH1);
            cellArr[vH2.idx()] = tumble_tree->insert(1, vH2);
            cellArr[vH3.idx()] = tumble_tree->insert(1, vH3);
            cellArr[vH4.idx()] = tumble_tree->insert(1, vH4);

            kd_tree->insert(top, vH1);
            kd_tree->insert(left, vH2);
            kd_tree->insert(right, vH3);
            kd_tree->insert(back, vH4);
        }
    }


    /**
     * Performs a laplacian smoothing operation on the vertex given as paramter
     * TODO: outsource it into the halfedge mesh or basemesh
     *
     * @tparam BaseVecT - the vector type used
     * @tparam NormalT - the normal type used
     * @param vertexH - vertex, which will be smoothed
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::performLaplacianSmoothing(VertexHandle vertexH, BaseVecT random, float factor)
    {
        vector<VertexHandle> n_vertices = m_mesh->getNeighboursOfVertex(vertexH);
        BaseVecT& vertex = m_mesh->getVertexPosition(vertexH);
        BaseVecT avg_vec(0,0,0);

        for(VertexHandle vH : n_vertices)
        {
            BaseVecT v = m_mesh->getVertexPosition(vH);
            avg_vec += (v - vertex) ;
        }
        avg_vec += (random - vertex);
        avg_vec /= n_vertices.size() + 1 ;

        vertex += avg_vec * factor;
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
        auto tree = m_surface->get()->searchTree();

        double avg_distance = 0;
        for(auto vertex : m_mesh->vertices())
        {
            vector<size_t> indexes;
            vector<float> distances;
            tree.get()->kSearch(m_mesh->getVertexPosition(vertex), 1, indexes, distances);
            avg_distance += distances[0];
        }

        avg_distance /= m_mesh->numVertices();

        std::cout << "avg_distance to cloud: " << avg_distance << endl;
        if(m_surface->get()->pointBuffer().get()->numPoints() < 10000000) std::cout << "avg distance between the points in the cloud: " << avgDistanceBetweenPointsInPointcloud() << endl;


        double avg_len = 0;

        for(auto eH : m_mesh->edges())
        {
            auto vertices = m_mesh->getVerticesOfEdge(eH);
            avg_len += (m_mesh->getVertexPosition(vertices[0]) - m_mesh->getVertexPosition(vertices[1])).length();
        }

        avg_len /= m_mesh->numEdges();


        //now remove .. :)

        vector<OptionalFaceHandle> f_handles;
        for(auto vertex : m_mesh->vertices())
        {
            vector<size_t> indexes;
            vector<float> distances;
            tree.get()->kSearch(m_mesh->getVertexPosition(vertex), 1, indexes, distances);
            if(distances[0] > 3 * avg_distance)
            {
                for(auto eH : m_mesh->getEdgesOfVertex(vertex))
                {
                    auto vertices = m_mesh->getVerticesOfEdge(eH);
                    float tmp = (m_mesh->getVertexPosition(vertices[0]) - m_mesh->getVertexPosition(vertices[1])).length();
                    if(tmp > 4 * avg_len)
                    {
                        auto faces = m_mesh->getFacesOfEdge(eH);
                        for(auto face : faces)
                        {
                            if(face)
                            {
                                bool in = false;
                                for(auto fH : f_handles)
                                {
                                    if(fH.unwrap().idx() == face.unwrap().idx())
                                    {
                                        in = true;
                                    }
                                }
                                if(!in)
                                {
                                    f_handles.push_back(face);
                                }
                            }
                        }
                    }
                }

            }
        }

        for(auto fH : f_handles)
        {
            if(fH)
            {
                m_mesh->removeFace(fH.unwrap());
            }

        }

    }

    /**
     * Aggressivly cuts out the vertex and all its adjacent faces
     * @tparam BaseVecT
     * @tparam NormalT
     * @param vH  Vertex which will be removed including its adjacent faces.
     */
    template <typename BaseVecT, typename NormalT>
    void GrowingCellStructure<BaseVecT, NormalT>::aggressiveCutOut(VertexHandle vH) {
        cout << "Aggressive Cutout..." << endl;
        auto faces = m_mesh->getFacesOfVertex(vH);
        tumble_tree->remove(cellArr[vH.idx()], vH);
        for(auto face : faces)
        {
            m_mesh->removeFace(face);
        }
    }

    /**
     * Calculates the average distance between the points in the poitncloud
     * @tparam BaseVecT
     * @tparam NormalT
     * @return the average distance TODO: AUSLAGERN IN SURFACE?
     */
    template<typename BaseVecT, typename NormalT>
    double GrowingCellStructure<BaseVecT,NormalT>::avgDistanceBetweenPointsInPointcloud()
    {
        //get average distance between the points
        auto pointer = m_surface->get()->pointBuffer();
        auto tree = m_surface->get()->searchTree();
        auto p_arr = pointer.get()->getPointArray();
        auto num_points = pointer.get()->numPoints();

        double sumDistance = 0;
        for(int i = 0; i < 3*num_points; i += 3)
        {
            BaseVecT point(p_arr[i], p_arr[i+1],p_arr[i+2]);
            vector<size_t> indices;
            vector<float> distances;
            tree->kSearch(point, 2, indices, distances);
            sumDistance += distances[1];
        }
        double avgDistance = sumDistance / num_points;
        return avgDistance;
    }


    /**
     * Number of vertices which exceed a given valance (including the valence itself)
     * @tparam BaseVecT
     * @tparam NormalT
     * @param minValence minimum valance which will be counted
     * @return
     */
    template <typename BaseVecT, typename NormalT>
    int GrowingCellStructure<BaseVecT, NormalT>::numVertexValences(int minValence)
    {
        auto vertices = m_mesh->vertices();
        int counter = 0;
        for(auto vertex : vertices)
        {
            size_t val = m_mesh->getNeighboursOfVertex(vertex).size();
            if(val >= minValence) counter++;
        }

        return counter;
    }

    /**
     * Calculates the average valance of the vertices in the mesh, doesnt really make sense. instead use a normal destribution
     * @tparam BaseVecT
     * @tparam NormalT
     * @return average valance in the current mesh
     */
    template <typename BaseVecT, typename NormalT>
    double GrowingCellStructure<BaseVecT, NormalT>::avgValence()
    {
        auto vertices = m_mesh->vertices();

        double val_counter = 0;

        for(auto vertex : vertices)
        {
            size_t val = m_mesh->getNeighboursOfVertex(vertex).size();
            val_counter += val;
        }

        return val_counter / m_mesh->numVertices();
    }


    /**
     * Calculates average skewness and equilaterality of the triangles in the mesh
     * @tparam BaseVecT
     * @tparam NormalT
     * @return a pair containing the average skewness and equilaterality value. Values in [0, ..., 1]
     */
    template <typename BaseVecT, typename NormalT>
    std::pair<double, double> GrowingCellStructure<BaseVecT, NormalT>::equilaterality()
    {
        auto faces = m_mesh->faces();
        size_t numFaces = m_mesh->numFaces();

        double skewnessPercentage = 0.0;
        double equilateralPercentage = 0.0;

        for(auto face : faces)
        {
            auto edges = m_mesh->getEdgesOfFace(face);
            double actual_area = m_mesh->calcFaceArea(face);

            float circumRadius = m_mesh->triCircumCenter(face).second;

            double sideLength = (3*circumRadius)/sqrt(3);

            double equilateralFaceArea = (sqrt(3) / 4) * (sideLength * sideLength);

            skewnessPercentage += (equilateralFaceArea - actual_area) / equilateralFaceArea;
            equilateralPercentage += actual_area / equilateralFaceArea;
        }

        return std::make_pair(skewnessPercentage / numFaces, equilateralPercentage / numFaces) ;
    }

    /**
     * calculates the size of the cell array
     * @tparam BaseVecT
     * @tparam NormalT
     * @return
     */
    template <typename BaseVecT, typename NormalT>
    int GrowingCellStructure<BaseVecT, NormalT>::cellVecSize()
    {
        int cellArrS = 0;
        for(int i = 0; i < cellArr.size(); i++)
        {
            if(cellArr[i] != NULL) cellArrS++;
        }
        return cellArrS;
    };

}