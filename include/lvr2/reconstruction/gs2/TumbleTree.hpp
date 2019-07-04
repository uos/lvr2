//
// Created by patrick on 4/4/19.
//

#ifndef LAS_VEGAS_TUMBLETREE_HPP
#define LAS_VEGAS_TUMBLETREE_HPP


#include <lvr2/geometry/Handles.hpp>
#include <lvr2/attrmaps/HashMap.hpp>

namespace  lvr2{

    typedef struct Cell
    {
        double alpha = 1;
        double signal_counter;
        HashMap<VertexHandle, double> duplicateMap;
        Cell* left; //left subtree
        Cell* right; //right subtree
        Cell* parent;
    } Cell;

    class TumbleTree {

        Cell* root;

        Cell* makeEmpty(Cell* c);

        Cell* remove(double sc, VertexHandle vH, Cell* c, bool removeWhole = false);
        Cell* insert(Cell* c, double sc, VertexHandle vH);
        Cell* findMin(Cell* c);
        Cell* findMax(Cell* c);
        Cell* find(double sc, VertexHandle vH, Cell* c, double alpha = 1);
        int size(Cell* c);
        int maxDepth(Cell* cell);
        int minDepth(Cell* cell);
        Cell* buildTree(vector<Cell*>& cells, int start, int end);
        void getCellsAsVector(Cell* c, vector<Cell*>& cells);

        void update(double alpha);

        void inorder(Cell* c);


    public:

        TumbleTree();
        ~TumbleTree();

        Cell* insertIterative(double sc, VertexHandle vH);
        Cell* insert(double sc, VertexHandle vH)
        {
            if(root == NULL)
            {
                cout << "new root.." << endl;
                auto newCell = new Cell();
                newCell->left = NULL;
                newCell->right = NULL;
                newCell->parent = NULL;
                newCell->alpha = 1;
                newCell->signal_counter = sc;
                newCell->duplicateMap.insert(vH, sc);

                root = newCell;

                return root;
            }

            return insert(root, sc, vH);


        }
        double remove(Cell* c, VertexHandle vH); //returns the real sc


        int maxDepth();
        int minDepth();
        Cell* find(double sc, VertexHandle vH);

        void display();
        void balance();

        Cell* min();
        Cell* max();

        int size();

        void updateSC(double alpha);
        int notDeleted = 0;
    };

    typedef struct node
    {
        double key;
        std::vector<VertexHandle> vertexHandles;
        node* left; //left subtree
        node* right; //right subtree

    } node;


    class BST
    {
    private:
        node* root = NULL;
        node* lastInserted = NULL;

        node* insert(node* current, double key, VertexHandle vH)
        {
            if(current == NULL)
            {
                auto newNode = new node();
                newNode->right = NULL;
                newNode->left = NULL;
                newNode->key = key;
                newNode->vertexHandles.push_back(vH);
#
                return newNode;
            }
            else if(key < current->key)
            {
                current->left = insert(current->left, key, vH);
            }
            else if(key > current->key)
            {
               current->right = insert(current->right, key, vH);
            }
            else
            {
                current->vertexHandles.push_back(vH);
                return current;
            }
        }

        node* remove(node* current, double key, VertexHandle vH, bool removeWhole = false)
        {
            if(current == NULL) return current;
            else if(key < current->key)
            {
                current->left = remove(current->left, key, vH, removeWhole);
            }
            else if(key > current->key)
            {
                current->right = remove(current->right, key, vH, removeWhole);
            }
            else
            {
                if(current->vertexHandles.size() > 1 && !removeWhole) current->vertexHandles.erase(std::find(current->vertexHandles.begin(), current->vertexHandles.end(), vH));
                else if(!current->left)
                {
                    node* tmp = current->right;
                    free(current);
                    return tmp;
                }
                else if(!current->right)
                {
                    node* tmp = current->left;
                    free(current);
                    return tmp;
                }
                else
                {
                    node* tmp = findMin(current->right);
                    current->key = tmp->key;
                    current->vertexHandles.swap(tmp->vertexHandles);
                    current->right = remove(current->right, tmp->key, tmp->vertexHandles[0], true);
                }
            }

            return current;
        }

        node* findMin(node* current)
        {
            if(current == NULL) return NULL;
            else if(current->left == NULL) return current;
            else return findMin(current->left);
        }

        node* findMax(node* current)
        {
            if(current == NULL) return NULL;
            else if(current->right == NULL) return current;
            else return findMin(current->right);
        }

        long size(node* current)
        {
            if(current == NULL) return 0;
            return current->vertexHandles.size() + size(current->left) + size(current->right);
        }

    public:

        void insert(double key, VertexHandle vH){root = insert(root, key, vH);}
        void remove(double key, VertexHandle vH){root = remove(root, key, vH);}
        long size(){return size(root);}

    };

}


#include "TumbleTree.tcc"

#endif //LAS_VEGAS_TUMBLETREE_HPP
