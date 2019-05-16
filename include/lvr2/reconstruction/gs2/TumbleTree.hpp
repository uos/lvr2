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
        float alpha = 1;
        float signal_counter;
        //Index vH; //VertexHandle-Index for mapping between VertexHandle and a Cell
        HashMap<VertexHandle, float> duplicateMap;
        Cell* left; //left subtree
        Cell* right; //right subtree
        Cell* parent;
    } Cell;

    class TumbleTree {

        Cell* root;

        Cell* makeEmpty(Cell* c);

        Cell* remove(float sc, VertexHandle vH, Cell* c, bool removeWhole = false, float alpha = 1);
        Cell* findMin(Cell* c);
        Cell* findMax(Cell* c);
        Cell* find(float sc, VertexHandle vH, Cell* c, float alpha = 1);
        int size(Cell* c);
        int maxDepth(Cell* cell);
        int minDepth(Cell* cell);
        Cell* buildTree(vector<Cell*>& cells, int start, int end);
        void getCellsAsVector(Cell* c, vector<Cell*>& cells);

        void update(float alpha);

        void inorder(Cell* c);


    public:

        TumbleTree();
        ~TumbleTree();

        Cell* insertIterative(float sc, VertexHandle vH);
        float remove(Cell* c, VertexHandle vH); //returns the real sc


        int maxDepth();
        int minDepth();
        Cell* find(float sc, VertexHandle vH);

        void display();
        void balance();

        Cell* min();
        Cell* max();

        int size();

        void updateSC(float alpha);
        int notDeleted = 0;
    };

}


#include "TumbleTree.tcc"

#endif //LAS_VEGAS_TUMBLETREE_HPP
