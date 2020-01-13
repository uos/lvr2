//
// Created by patrick on 4/4/19.
//

#ifndef LAS_VEGAS_TUMBLETREE_HPP
#define LAS_VEGAS_TUMBLETREE_HPP

#include "lvr2/attrmaps/HashMap.hpp"
#include "lvr2/geometry/Handles.hpp"

namespace lvr2
{

typedef struct Cell
{
    double alpha = 1;
    double signal_counter;
    HashMap<VertexHandle, double> duplicateMap;
    Cell* left;  // left subtree
    Cell* right; // right subtree
    Cell* parent;
} Cell;

class TumbleTree
{

    Cell* root;

    Cell* makeEmpty(Cell* c);

    Cell* remove(double sc, VertexHandle vH, Cell* c, bool removeWhole = false);
    // Cell* removeTwo(double sc, VertexHandle vH, Cell* c, bool removeWhole = false, double alpha =
    // 1);

    Cell* insert(Cell* c, double sc, VertexHandle vH);
    Cell* findMin(Cell* c);
    Cell* findMax(Cell* c);
    Cell* find(double sc, VertexHandle vH, Cell* c, double alpha = 1);
    int size(Cell* c);
    int maxDepth(Cell* cell);
    int minDepth(Cell* cell);
    int sumDepth(Cell* c, int currentDepth = 1);
    int numLeafes(Cell* c);
    Cell* buildTree(vector<Cell*>& cells, int start, int end);
    void getCellsAsVector(Cell* c, vector<Cell*>& cells);

    void update(double alpha);

    void inorder(Cell* c);

  public:
    TumbleTree();
    ~TumbleTree();

    // Cell* insertIterative(double sc, VertexHandle vH);
    Cell* insert(double sc, VertexHandle vH);
    double remove(Cell* c, VertexHandle vH); // returns the real sc

    int maxDepth();
    int minDepth();
    int avgDepth();
    Cell* find(double sc, VertexHandle vH);

    void display();
    void balance();

    Cell* min();
    Cell* max();

    Cell* makeCell(double sc,
                   VertexHandle vH,
                   Cell* left = NULL,
                   Cell* right = NULL,
                   Cell* parent = NULL,
                   double alpha = 1);

    int size();

    void updateSC(double alpha);
    int notDeleted = 0;
};

} // namespace lvr2

#include "TumbleTree.tcc"

#endif // LAS_VEGAS_TUMBLETREE_HPP
