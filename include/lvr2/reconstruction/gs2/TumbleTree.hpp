//
// Created by patrick on 4/4/19.
//

#ifndef LAS_VEGAS_TUMBLETREE_HPP
#define LAS_VEGAS_TUMBLETREE_HPP


#include <lvr2/geometry/Handles.hpp>
namespace  lvr2{

    struct Cell
    {
        float alpha;
        float signal_counter;
        Index vH; //VertexHandle-Index for mapping between VertexHandle and a Cell
        Cell* left; //left subtree
        Cell* right; //right subtree
    };

    class TumbleTree {

        Cell* root;

        Cell* makeEmpty(Cell* c);

        Cell* insert(float sc, Index vH, Cell* c);
        Cell* remove(float sc, Index vH, Cell* c);
        Cell* findMin(Cell* c);
        Cell* findMax(Cell*);

        void update(Cell* c, float alpha);

        void inorder(Cell* c);

    public:

        TumbleTree();
        ~TumbleTree();

        Cell* insert(float sc, VertexHandle vH);
        void remove(Cell* c);

        void display();

        Cell* min();
        Cell* max();

        void updateSC(float alpha);
    };

}


#include "TumbleTree.tcc"

#endif //LAS_VEGAS_TUMBLETREE_HPP
