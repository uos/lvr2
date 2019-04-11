//
// Created by patrick on 4/11/19.
//

namespace lvr2{

    template <typename BaseVecT>
    Node<BaseVecT>* DynamicKDTree<BaseVecT>::newNode(BaseVecT point, VertexHandle vH)
    {
        auto temp = new Node<BaseVecT>;
        temp->point = point;
        temp->vH = vH.idx();
        temp->left = temp->right = NULL;
        return temp;
    }
    template <typename BaseVecT>
    Node<BaseVecT>*  DynamicKDTree<BaseVecT>::insertRec(Node<BaseVecT>* node, BaseVecT point,VertexHandle vH, unsigned int depth)
    {
        // Tree is empty?
        if (node == NULL)
            return newNode(point, vH);

        // Calculate current dimension (cd) of comparison
        unsigned cd = depth % k;

        // Compare the new point with root on current dimension 'cd'
        // and decide the left or right subtree
        if (point[cd] < (node->point[cd]))
            node->left = insertRec(node->left, point, vH, depth + 1);
        else
            node->right = insertRec(node->right, point, vH, depth + 1);

        return node;
    }

    
    template <typename BaseVecT>
    Node<BaseVecT>* DynamicKDTree<BaseVecT>::minNode(Node<BaseVecT>* x, Node<BaseVecT>* y, Node<BaseVecT>* z, int d)
    {
        Node<BaseVecT> *res = x;
        if (y != NULL && y->point[d] < res->point[d])
            res = y;
        if (z != NULL && z->point[d] < res->point[d])
            res = z;
        return res;
    }


    template <typename BaseVecT>
    Node<BaseVecT>* DynamicKDTree<BaseVecT>::findMinRec(Node<BaseVecT>* node, int d, unsigned depth)
    {
        // Base cases
        if (node == NULL)
            return NULL;

        // Current dimension is computed using current depth and total
        // dimensions (k)
        unsigned cd = depth % k;

        // Compare point with root with respect to cd (Current dimension)
        if (cd == d)
        {
            if (node->left == NULL)
                return node;
            return findMinRec(node->left, d, depth+1);
        }

        // If current dimension is different then minimum can be anywhere
        // in this subtree

        return minNode(node,
                       findMinRec(node->left, d, depth+1),
                       findMinRec(node->right, d, depth+1), d);

    }
    template <typename BaseVecT>
    Node<BaseVecT>* DynamicKDTree<BaseVecT>::findMin(Node<BaseVecT>* node, int d)
    {
        // Pass current level or depth as 0
        return findMinRec(node, d, 0);
    }


    template <typename BaseVecT>
    Node<BaseVecT>* DynamicKDTree<BaseVecT>::deleteNodeRec(Node<BaseVecT>* node, BaseVecT point, int depth)
    {
        // Given point is not present
        if (node == NULL)
            return NULL;

        // Find dimension of current node
        int cd = depth % k;

        // If the point to be deleted is present at root
        if (arePointsSame(node->point, point))
        {
            // 2.b) If right child is not NULL
            if (node->right != NULL)
            {
                // Find minimum of root's dimension in right subtree
                Node<BaseVecT> *min = findMin(node->right, cd);

                // Copy the minimum to root
                //copyPoint(node->point, min->point);
                node->point = min->point;
                node->vH = min->vH;

                // Recursively delete the minimum
                node->right = deleteNodeRec(node->right, min->point, depth+1);
            }
            else if (node->left != NULL) // same as above
            {
                Node<BaseVecT> *min = findMin(node->left, cd);
                //copyPoint(node->point, min->point);
                node->point = min->point;
                node->vH = min->vH;
                node->right = deleteNodeRec(node->left, min->point, depth+1);
            }
            else // If node to be deleted is leaf node
            {
                delete node;
                return NULL;
            }
            return node;
        }

        // 2) If current node doesn't contain point, search downward
        if (point[cd] < node->point[cd])
            node->left = deleteNodeRec(node->left, point, depth+1);
        else
            node->right = deleteNodeRec(node->right, point, depth+1);
        return node;
    }

    template <typename BaseVecT>
    int DynamicKDTree<BaseVecT>::sizeRec(Node<BaseVecT>* node)
    {
        if(node == NULL) return 0;
        return 1 + sizeRec(node->right) + sizeRec(node->left);
    }

    template <typename BaseVecT>
    void DynamicKDTree<BaseVecT>::insert(BaseVecT point, VertexHandle vH)
    {
        root = insertRec(root, point, vH, 0);
    }

    template <typename BaseVecT>
    void DynamicKDTree<BaseVecT>::deleteNode(BaseVecT point)
    {
        // Pass depth as 0
        std::cout << "Deleting..." << endl;
        root =  deleteNodeRec(root, point, 0);
    }

    template <typename BaseVecT>
    int DynamicKDTree<BaseVecT>::size()
    {
        return sizeRec(root);
    }

} //end namespace lvr2

