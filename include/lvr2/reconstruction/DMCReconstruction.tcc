/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
 * DMCReconstruction.tcc
 *
 *  Created on: 18.01.2019
 *      Author: Benedikt Schumacher
 */

#include "lvr2/geometry/BaseMesh.hpp"
#include <vector>
#include <random>
using std::vector;
namespace lvr2
{

template<typename BaseVecT, typename BoxT>
DMCReconstruction<BaseVecT, BoxT>::DMCReconstruction(
        PointsetSurfacePtr<BaseVecT> surface,
        BoundingBox<BaseVecT> bb,
        bool extrude) : PointsetMeshGenerator<BaseVecT>(surface)
{

    bb_min = bb.getMin();
    bb_max = bb.getMax();

    float step = 0.2;
    //float step = 0.5;
    //float step = 0.0;
    for(unsigned char a = 0; a < 3; a++)
    {
        bb_min[a] -= (fabs(bb_min[a]) * step);
        bb_max[a] += (fabs(bb_max[a]) * step);
        bb_size[a] = bb_max[a] - bb_min[a];
    }

    m_boundingBoxCenter = bb.getCentroid();
    m_maxSize = m_voxelSize;

    // Calculate a maximum voxelsize that is divisible by 2
    while (m_maxSize < bb.getLongestSide())
    {
        m_maxSize *= 2;
    }

    cout << timestamp << "longestSide of (previous) BoundingBox: " << bb.getLongestSide() << endl;
    cout << timestamp << "new calculated longestSide of BoundingBox: " << m_maxSize << endl;

    // Get all points
    /*size_t num_points;
    coord3fArr points = this->m_surface->pointBuffer()->getIndexedPointArray(num_points);
    vector<coord<float>*> containedPoints;
    for(size_t i = 0; i < num_points; i++)
    {
        containedPoints.push_back(&points[i]);
    }*/

    // Instanciate Octree
    octree = new C_Octree<BaseVecT, BoxT, my_dummy>();
    octree->initialize(MAX_LEVEL);

    // Create thread pool
    m_threadPool = new OctreeThreadPool<BaseVecT, BoxT>(boost::thread::hardware_concurrency());

    m_nodes = 0;
    m_nodesExtr = 0;
    m_leaves = 0;
    m_leavesExtr = 0;
    m_faces = 0;

    if (!m_extrude)
    {
        cout << timestamp << "Octree is not extruded." << endl;
    }
    cout << timestamp << "Creating Octree..." << endl;
    int n_levels = MAX_LEVEL;
    buildTree(*octree, n_levels);

}

template<typename BaseVecT, typename BoxT>
DMCReconstruction<BaseVecT, BoxT>::~DMCReconstruction()
{
    delete m_threadPool;
    delete m_progressBar;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::buildTree(
        C_Octree<BaseVecT, BoxT, my_dummy> &parent,
        int levels)
{
    m_leaves = 0;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(0, 100);

    // voll funktionstüchtig
    for(int cur_Level = levels; cur_Level > 0; --cur_Level)
    {
        CellHandle ch_end = parent.end();
        for (CellHandle ch = parent.root(); ch != ch_end; ++ch)
        {
            int rand = dis(gen);
            // std::cout << rand << std::endl;
            if ((parent.level(ch) == cur_Level))
            {
                //if(cur_Level > 3 || rand >= 20)
                //{
                if(parent.location(ch).loc_x() < 16 || cur_Level > 1)
                {
                    parent.split( ch );
                    m_leaves += 7;
                }
            }
        }
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getMesh(BaseMesh<BaseVecT> &mesh)
{
    m_globalIndex = 0;
    string comment = timestamp.getElapsedTime() + "Creating Mesh ";
    m_progressBar = new ProgressBar(m_leaves, comment);
    m_threadPool->startPool();
    traverseTree(mesh, *octree, m_threadPool);
    m_threadPool->stopPool();
    cout << endl;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::traverseTree(
        BaseMesh<BaseVecT> &mesh,
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        OctreeThreadPool<BaseVecT, BoxT>* threadPool)
{
    CellHandle ch_end = octree.end();
    int cells = 2;
    for(int i = 1; i < MAX_LEVEL; i++)
    {
        cells *= 2;
    }

    for (CellHandle ch = octree.root(); ch != ch_end; ++ch)
    {
        if (octree.is_leaf(ch))
        {
            // start building dual Leaf
            BaseVecT corners[8];
            float* max_bb_width = std::max_element(bb_size, bb_size+3);

            for(unsigned char c = 0; c < 8; c++)
            {
                bool outside = false;

                // get all neighbors
                std::vector<CellHandle> cellHandles = octree.all_corner_neighbors(ch, c);

                // find vertex of each cell
                unsigned char i = 0;
                while(i < 8 && !outside)
                {
                    BaseVecT tmp;
                    detectVertexForDualCell(octree, cellHandles[i], cells, *max_bb_width, i, tmp);

                    corners[i] = tmp;
                    for(unsigned char j = 0; j < 3; j++)
                    {
                        if(corners[i][j] > bb_max[j])
                        {
                            outside = true;
                        }
                    }
                    i++;
                }

                if(!outside)
                {
                    // resort the vertices to needed coordinate system
                    //        6------7         7------6
                    //       /|     /|        /|     /|
                    //      2------3 |       3------2 |
                    // FROM | 4----|-5  ===> | 4----|-5
                    //      |/     |/        |/     |/
                    //      0------1         0------1
                    //
                    BaseVecT tmp = corners[2];
                    corners[2] = corners[3];
                    corners[3] = tmp;
                    tmp = corners[6];
                    corners[6] = corners[7];
                    corners[7] = tmp;

                    // generate dual leaf
                    DualLeaf<BaseVecT, BoxT> *dualLeaf = new DualLeaf<BaseVecT, BoxT>(corners);

                    threadPool->insertTask((boost::function<void()>)boost::bind(
                            &lvr2::DMCReconstruction<BaseVecT, BoxT>::getSurface,
                            this,
                            boost::ref(mesh),
                            dualLeaf,
                            cells));
                }
            }
            ++(*m_progressBar);
        }
    }
    return;
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::detectVertexForDualCell(
        C_Octree<BaseVecT, BoxT, my_dummy> &octree,
        CellHandle ch,
        int cells,
        float max_bb_width,
        uint pos,
        BaseVecT &feature)
{
    feature = octree.cell_center(ch);
    BaseVecT tmp;
    tmp = octree.cell_corner(ch, 0);

    if (pos == 0)
    {
        if(tmp[0] == 0.0)
        {
            feature[0] = 0.0;
        }
        if(tmp[1] == 0.0)
        {
            feature[1] = 0.0;
        }
        if(tmp[2] == 0.0)
        {
            feature[2] = 0.0;
        }
    }
    else if (pos == 1)
    {
        if(tmp[0] == cells - 1)
        {
            feature[0] += 0.5;
        }
        if(tmp[1] == 0.0)
        {
            feature[1] = 0.0;
        }
        if(tmp[2] == 0.0)
        {
            feature[2] = 0.0;
        }
    }
    else if (pos == 2)
    {
        if(tmp[0] == 0.0)
        {
            feature[0] = 0.0;
        }
        if(tmp[1] == cells - 1)
        {
            feature[1] += 0.5;
        }
        if(tmp[2] == 0.0)
        {
            feature[2] = 0.0;
        }
    }
    else if (pos == 3)
    {
        if(tmp[0] == cells - 1)
        {
            feature[0] += 0.5;
        }
        if(tmp[1] == cells - 1)
        {
            feature[1] += 0.5;
        }
        if(tmp[2] == 0.0)
        {
            feature[2] = 0.0;
        }
    }
    else if (pos == 4)
    {
        if(tmp[0] == 0.0)
        {
            feature[0] = 0.0;
        }
        if(tmp[1] == 0.0)
        {
            feature[1] = 0.0;
        }
        if(tmp[2] == cells - 1)
        {
            feature[2] = 0.5;
        }
    }
    else if (pos == 5)
    {
        if(tmp[0] == cells - 1)
        {
            feature[0] += 0.5;
        }
        if(tmp[1] == 0.0)
        {
            feature[1] = 0.0;
        }
        if(tmp[2] == cells - 1)
        {
            feature[2] = 0.5;
        }
    }
    else if (pos == 6)
    {
        if(tmp[0] == cells - 1)
        {
            feature[0] += 0.5;
        }
        if(tmp[1] == cells - 1)
        {
            feature[1] += 0.5;
        }
        if(tmp[2] == cells - 1)
        {
            feature[2] = 0.5;
        }
    }
    else if (pos == 7)
    {
        if(tmp[0] == cells - 1)
        {
            feature[0] += 0.5;
        }
        if(tmp[1] == cells - 1)
        {
            feature[1] += 0.5;
        }
        if(tmp[2] == cells - 1)
        {
            feature[2] = 0.5;
        }
    }

    for(unsigned char i = 0; i < 3; i++)
    {
        feature[i] *= (max_bb_width / cells);
        feature[i] += (bb_min[i]);
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getSurface(
        BaseMesh<BaseVecT> &mesh,
        DualLeaf<BaseVecT, BoxT> *leaf,
        int cells)
{
    BaseVecT edges[8];
    float distances[8];
    BaseVecT vertex_positions[12];
    float projectedDistance;
    float euklideanDistance;
    HalfEdgeMesh<BaseVecT> *meshCast;

    if (m_reconstructionType == "PMC")
    {
        meshCast = static_cast<HalfEdgeMesh<BaseVecT>* >(&mesh);
    }

    for (unsigned char i = 0; i < 8; i++)
    {
        leaf->getVertices(edges);
        // die distance-values kommen nicht aus dem Octree sondern aus dem KD-Tree
        float projectedDistance;
        float euklideanDistance;
        std::tie(projectedDistance, euklideanDistance) = this->m_surface->distance(edges[i]);
        distances[i] = projectedDistance;
    }
    // lediglich setzen der distanzwerte, bzw markieren, wo vorzeichenwechsel stattfinden
    leaf->getIntersections(edges, distances, vertex_positions);

    // check whether the distances are close enough or not
    for(unsigned char a = 0; a < 8; a++)
    {
        float *max = std::max_element(bb_size, bb_size+3);
        float length = edges[1][0] - edges[0][0];

        // Distanzen auf zulaessige Laenge pruefen
        if(distances[a] > (length * 1.7) || distances[a] < (length * (-1.7)))
        {
            return;
        }
    }

    // index is for mc-table
    int index = leaf->getIndex(distances);
    uint edge_index = 0;

    //int triangle_indices[3];
    for(unsigned char a = 0; MCTable[index][a] != -1; a+= 3)
    {
        vector<VertexHandle> triangle_vertices;
        for(unsigned char b = 0; b < 3; b++)
        {
            edge_index = MCTable[index][a + b];
            boost::unique_lock<boost::mutex> globalLock(m_globalMutex);
            if(leaf->getIntersection(edge_index) == numeric_limits<uint>::max())
            {
                leaf->setIntersection(edge_index, m_globalIndex);
                BaseVecT v = vertex_positions[edge_index];
                VertexHandle vh = mesh.addVertex(v);
                triangle_vertices.push_back(vh);
                ++m_globalIndex;
            }
            else {
                BaseVecT v = vertex_positions[edge_index];
                VertexHandle vh = mesh.addVertex(v);
                triangle_vertices.push_back(vh);
            }
            globalLock.unlock();
        }
        boost::unique_lock<boost::mutex> addLock(m_addMutex);
        mesh.addFace(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]);
        addLock.unlock();
    }
}

template<typename BaseVecT, typename BoxT>
void DMCReconstruction<BaseVecT, BoxT>::getMesh(
    BaseMesh<BaseVecT>& mesh,
    BoundingBox<BaseVecT>& bb,
    vector<unsigned int>& duplicates,
    float comparePrecision
)
{
//    // Status message for mesh generation
//    string comment = timestamp.getElapsedTime() + "Creating Mesh ";
//    ProgressBar progress(m_grid->getNumberOfCells(), comment);

//    // Some pointers
//    BoxT* b;
//    unsigned int global_index = mesh.numVertices();

//    // Iterate through cells and calculate local approximations
//    typename HashGrid<BaseVecT, BoxT>::box_map_it it;
//    for(it = m_grid->firstCell(); it != m_grid->lastCell(); it++)
//    {
//        b = it->second;
//        b->getSurface(mesh, m_grid->getQueryPoints(), global_index, bb, duplicates, comparePrecision);
//        if(!timestamp.isQuiet())
//            ++progress;
//    }

//    if(!timestamp.isQuiet())
//        cout << endl;
}

} // namespace lvr2
