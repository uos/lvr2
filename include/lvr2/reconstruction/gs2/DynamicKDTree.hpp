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
        Node *left, *right;
    };

    template <typename BaseVecT>
    class DynamicKDTree {

    private:
        Node<BaseVecT>* root;

        struct Node<BaseVecT>* newNode(BaseVecT point);

        Node<BaseVecT>* insertRec(Node<BaseVecT>* root, BaseVecT point, unsigned int depth);

        Node<BaseVecT>* minNode(Node<BaseVecT>* x, Node<BaseVecT>* y, Node<BaseVecT>* z, int d);

        Node<BaseVecT>* findMinRec(Node<BaseVecT>* root, int d, unsigned depth);

        Node<BaseVecT> *findMin(Node<BaseVecT>* root, int d);

        bool arePointsSame(BaseVecT point1, BaseVecT point2)
        {
            return point1 == point2;
        }

        void copyPoint(BaseVecT p1, BaseVecT p2)
        {
            p1 = p2;
        }

        Node<BaseVecT>* deleteNodeRec(Node<BaseVecT>* root, BaseVecT point, int depth);

    public:
        Node<BaseVecT>* insert(BaseVecT point);

        Node<BaseVecT>* deleteNode(BaseVecT point);

    };
}

#include "DynamicKDTree.tcc"

#endif //LAS_VEGAS_DYNAMICKDTREE_HPP
