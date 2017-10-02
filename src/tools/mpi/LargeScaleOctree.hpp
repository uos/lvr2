//
// Created by eiseck on 08.12.15.
//

#ifndef LAS_VEGAS_LargeScaleOctree_H
#define LAS_VEGAS_LargeScaleOctree_H

#include <stack>
#include <lvr/geometry/BoundingBox.hpp>
#include <lvr/geometry/Vertex.hpp>
#include <boost/filesystem.hpp>
#include <limits>
#include "NodeData.hpp"
namespace lvr
{
    typedef lvr::Vertex<int> Vertexi;

    class LargeScaleOctree {
    private:



        // pointer to root of octree
        LargeScaleOctree* m_root;
        // Octree node width
        float m_size;
        // Center of Node
        Vertexf m_center;
        // object containing point dataa
        NodeData m_data;
        // maximum number of points a node can store
        const size_t m_maxPoints;
        // pointer to parent node
        LargeScaleOctree* m_parent;
        // depth in tree of node (root depth = 0 )
        int m_depth;
        // Real bounding box around the points of the node
        BoundingBox<Vertexf> m_pointbb;
        // Children of this node
        vector<LargeScaleOctree*>  m_children;
        // Neighbours of node: front, back, top, down, right, left
        vector<LargeScaleOctree*>  m_neighbours;

        /**
         * Finds neighbhours of node
         * @return vector<LargeScaleOctree*> vector of neighbours
         */
        vector<LargeScaleOctree*> getNeighbours();

        /**
         * used when a neighbour is not a leaf, gets all children that are located on the side next to the neighbour node
         * @param octant neighbour node
         * @param dir direction where to search
         * @return vector<LargeScaleOctree*>  vector of neighbours
         */
        vector<LargeScaleOctree*> getRecChildrenNeighbours(LargeScaleOctree* octant, int dir);

        /**
         * Private constructor, used to create nodes with depth >0
         * @param center center of octree
         * @param size width of octree
         * @param maxPoints maximum allowed points
         * @param parent parent node
         * @param root root node
         * @param depth depth of this node
         * @param bufferSize maximum buffer size of point data object
         */
        LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints,  LargeScaleOctree* parent, LargeScaleOctree* root, int depth, size_t bufferSize = 40000000);

        /**
         * Return all nodes after root (including root)
         * @param root start node
         * @param nodelist output vector of nodes
         */
        void getNodes(LargeScaleOctree* root, vector<LargeScaleOctree*>& nodelist);

        /**
         * Used to determin in which octant a point should be added
         * @param point
         * @return id of octant (0-7)
         */
        inline int getOctant(const Vertexf &point) const;

    public:

        /**
         * Public constructor
         * @param center center of octree
         * @param size width of octree
         * @param maxPoints  maxPoints maximum allowed points
         * @param bufferSize maximum buffer size of point data object
         */
        LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints,  size_t bufferSize = 40000000);

        /**
         * compare operator, compares amount of points stored in node
         * @param rhs compare object
         * @return getSize() < rhs.getSize()
         */
        bool operator<( LargeScaleOctree& rhs ) ;

        /**
         * Returns true if node is a leaf
         * @return
         */
        bool isLeaf();

        /**
         * Inserts a point to the octree
         * @param pos point to insert
         */
        void insert(Vertexf& pos, Vertexf normal = Vertexf(std::numeric_limits<float>::min(),std::numeric_limits<float>::min(),std::numeric_limits<float>::min()));

        /**
         * gets amount of points in node
         * @return
         */
        size_t getSize();

        /**
         * Get the center of octree
         * @return
         */
        Vertexf getCenter();

        /**
         * Get width of node
         * @return
         */
        float getWidth();

        /**
         * returns children of node (only children on level current level-1)
         * @return vector of children
         */
        vector<LargeScaleOctree*>& getChildren();

        /**
         * returns all neighbours of node (only returns data if generateNeighbourhood() has been called bevor)
         * @return vector of neighburs
         */
        vector<LargeScaleOctree*>& getSavedNeighbours();

        /**
         * returns path to the file where the point data is stored
         * @return string path
         */
        string getFilePath();

        /**
         * returns folder name where the data is stored
         * @return  string path
         */
        string getFolder();

        /**
         * returns all nodes stored in octree
         * @return vector of nodes
         */
        vector<LargeScaleOctree*> getNodes();

        /**
         * finds and stores neighbourhood of this node
         */
        void generateNeighbourhood();
        /**
         * returns depth of node
         * @return int depth
         */
        int getDepth(){return m_depth;}

        /**
         * returns reald bounding box of the points stored in this node
         * @return
         */
        BoundingBox<Vertexf>& getPointBB() {return m_pointbb;}

        /**
         * call to make sure all the buffers have been written to file
         */
        void writeData();

        void PrintTimer() {NodeData::printTimer ();}

    };


    const static Vertexi dirTable[6] = {
            // Vorne/Hinten, Oben/Unten, Rechts/Links
            {1,0,0},  // Vorne
            {-1,0,0}, // Hinten
            {0,1,0},  // Oben
            {0,-1,0}, // Unten
            {0,0,1},  // Rechts
            {0,0,-1}, // Links
    };

    // children of a neighbour in a direction which are also neighbours
    const static int dirChildrenTable[6][4] = {
            // Vorne/Hinten, Oben/Unten, Rechts/Links
            {4,5,6,7}, // Vorne
            {0,1,2,3}, // Hinten
            {2,3,6,7}, // Oben
            {0,1,4,5}, // Unten
            {1,3,5,7}, // Rechts
            {0,2,4,6}, // Links
    };


}



#endif //LAS_VEGAS_LargeScaleOctree_H