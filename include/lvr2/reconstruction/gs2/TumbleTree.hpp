//
// Created by patrick on 4/4/19.
//

#ifndef LAS_VEGAS_TUMBLETREE_HPP
#define LAS_VEGAS_TUMBLETREE_HPP


#include <lvr2/geometry/Handles.hpp>
#include <lvr2/attrmaps/HashMap.hpp>

namespace  lvr2{

    struct Cell
    {
        float alpha;
        float signal_counter;
        //Index vH; //VertexHandle-Index for mapping between VertexHandle and a Cell
        HashMap<VertexHandle, float> duplicateMap;
        Cell* left; //left subtree
        Cell* right; //right subtree
    };

    class TumbleTree {

        Cell* root;

        Cell* makeEmpty(Cell* c);

        Cell* insert(float sc, VertexHandle vH, Cell* c);
        Cell* remove(float sc, VertexHandle vH, Cell* c);
        Cell* findMin(Cell* c);
        Cell* findMax(Cell*);
        Cell* find(float sc, VertexHandle vH, Cell* c);
        int size(Cell* c);

        void update(Cell* c, float alpha, VertexHandle vH);

        void inorder(Cell* c);

    public:

        TumbleTree();
        ~TumbleTree();

        void insert(float sc, VertexHandle vH);
        void remove(Cell* c, VertexHandle vH);

        Cell* find(float sc, VertexHandle vH);

        void display();

        Cell* min();
        Cell* max();

        int size();

        void updateSC(float alpha, VertexHandle vH);
    };

}


#include "TumbleTree.tcc"

#endif //LAS_VEGAS_TUMBLETREE_HPP
