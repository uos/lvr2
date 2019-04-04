//
// Created by patrick on 4/4/19.
//

#ifndef LAS_VEGAS_TUMBLETREE_HPP
#define LAS_VEGAS_TUMBLETREE_HPP


#include <lvr2/geometry/Handles.hpp>
namespace  lvr2{

    class TumbleTree {

        struct node
        {
            float alpha;
            float signal_counter;
            //VertexHandle vH;
            int vH; //index of the corresponding vertexhandle
            node* left;
            node* right;
        };

        node* makeEmpty(node* t);

        node* insert(float sc, int vH, node* t);

        node* findMin(node* t);

        node* findMax(node* t);

        node* remove(float sc, node* t);

        void inorder(node* t);

        node* find(node* t, float sc);


        node* root;

    public:
        TumbleTree();
        ~TumbleTree();

        void insert(int x, VertexHandle vH);
        void remove(int x);

        void display();

        VertexHandle min();
        VertexHandle max();

        void update(float alpha);
    };
}


#include "TumbleTree.tcc"

#endif //LAS_VEGAS_TUMBLETREE_HPP
