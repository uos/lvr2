//
// Created by patrick on 4/11/19.
//

#ifndef LAS_VEGAS_DYNAMICKDTREE_HPP
#define LAS_VEGAS_DYNAMICKDTREE_HPP

namespace lvr2{

    template <typename BaseVecT>
    struct Node
    {
        BaseVecT point; // To store k dimensional point
        Index vH; //To store the belonging vertexHandle index
        Node<BaseVecT> *left, *right;
    };

    template <typename BaseVecT>
    class DynamicKDTree {

    private:
        Node<BaseVecT>* root;
        int k;

        struct Node<BaseVecT>* newNode(BaseVecT point, VertexHandle vH);

        Node<BaseVecT>* insertRec(Node<BaseVecT>* node, BaseVecT point, VertexHandle vH, unsigned int depth);

        Node<BaseVecT>* minNode(Node<BaseVecT>* x, Node<BaseVecT>* y, Node<BaseVecT>* z, int d);

        Node<BaseVecT>* findMinRec(Node<BaseVecT>* node, int d, unsigned depth);

        Node<BaseVecT> *findMin(Node<BaseVecT>* node, int d);

        bool arePointsSame(BaseVecT point1, BaseVecT point2)
        {
            return point1 == point2;
        }

        void copyPoint(BaseVecT& p1, BaseVecT& p2)
        {
            p1 = p2;
        }

        Node<BaseVecT>* deleteNodeRec(Node<BaseVecT>* node, BaseVecT point, int depth);

        int sizeRec(Node<BaseVecT>* node);

        std::pair<Index, float> findNearestRec(Node<BaseVecT>* node, BaseVecT point, int depth, Index minDist, float minDistSq, BaseVecT currentBest);

    public:
        void insert(BaseVecT point, VertexHandle vH);

        void deleteNode(BaseVecT point);

        int size();

        Index findNearest(BaseVecT point);

        explicit DynamicKDTree(int k) : k(k)
        {
            root = NULL;
        }

        //TODO
        ~DynamicKDTree() = default;

    };
}

#include "DynamicKDTree.tcc"

#endif //LAS_VEGAS_DYNAMICKDTREE_HPP
