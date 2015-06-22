/**
 * @file 
 * @brief Efficient representation of an octree
 * @author Jan Elsberg. Automation Group, Jacobs University Bremen gGmbH, Germany. 
 * @author Kai Lingemann. Institute of Computer Science, University of Osnabrueck, Germany.
 * @author Andreas Nuechter. Institute of Computer Science, University of Osnabrueck, Germany.
 */

#ifndef BOCTREE_H
#define BOCTREE_H

#include "searchTree.h"
#include "point_type.h"
#include "data_types.h"
#include "allocator.h"
#include "limits.h"
#include "nnparams.h"
#include "globals.icc"


#include <stdio.h>

#include <vector>
using std::vector;
#include <deque>
using std::deque;
#include <set>
using std::set;
#include <list>
using std::list;
#include <iostream>
#include <fstream>
#include <string>

#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
  #define POPCOUNT(mask) __builtin_popcount(mask)
#else
  #define POPCOUNT(mask) _my_popcount_3(mask)
#endif

#include <boost/interprocess/offset_ptr.hpp> // to avoid ifdeffing for offset_ptr.get(), use &(*ptr)
namespace { namespace ip = boost::interprocess; }


// forward declaration
template <class T> union bitunion;

/**
 * This is our preferred representation for the leaf nodes (as it is the most compact). 
 * BOctTree makes an array of this, the first containing the number of points (not the 
 * number of coordinates) stored.
 */
template <class T> union dunion {
  T v;
  unsigned int    length;
  dunion() : length(0) {};

};
// typedefs in combination with templates are weird
//typedef dunion<T> pointrep<T>;
#define pointrep union dunion<T>




/**
 * This struct represents the nodes of the octree
 *
 * child_pointer is a relative pointer to the first child of this node, as it is only
 * 48 bit this will cause issues on systems with more than 268 TB of memory. All children
 * of this node must be stored sequentially. If one of the children is a leaf, that
 * child will be a pointer to however a set of points is represented (pointrep *).
 *
 * valid is a bitmask describing whether the corresponding buckets are filled.
 *
 * leaf is a bitmask describing whether the correpsonding bucket is a leaf node.
 *
 * The representation of the bitmask is somewhat inefficient. We use 16 bits for only 
 * 3^8 possible states, so in essence we could save 3 bits by compression.
 *
 */
class bitoct{
  public:

#ifdef _MSC_VER
  __int64 child_pointer        : 48;
  unsigned valid              :  8;
  unsigned leaf               :  8;
#else
  signed long child_pointer   : 48;
  unsigned valid              :  8;
  unsigned leaf               :  8;
#endif
  /**
   * sets the child pointer of parent so it points to child 
   */
  template <class T>
  static inline void link(bitoct &parent, bitunion<T> *child) {
    parent.child_pointer = (long)((char*)child - (char*)&parent);
  }

  /**
   * Returns the children of this node (given as parent).
   */
  template <class T>
  static inline void getChildren(const bitoct &parent, bitunion<T>* &children) {
    children = (bitunion<T>*)((char*)&parent + parent.child_pointer);
  }

  template <class T>
  inline bitunion<T>* getChild(unsigned char index) {
    bitunion<T> *children = (bitunion<T>*)((char*)this + this->child_pointer);
    for (unsigned char i = 0; i < index; i++) {
      if (  ( 1 << i ) & valid ) {   // if ith node exists
        children++;
      }
    }
    return children;
  }
};
  

/**
 * This union combines an octree node with a pointer to a set of points. This allows
 * us to use both nodes and leaves interchangeably.
 *
 * points is a pointer to the point representation in use
 *
 * node is simply the octree node
 *
 */
template <class T> union bitunion {
  pointrep *points;
  //union dunion<T> *points;
  bitoct node;

  bitunion(pointrep *p) : points(p) {};
  bitunion(bitoct b) : node(b) {};
  bitunion() : points(0) {
    node.child_pointer = 0;
    node.valid = 0;
    node.leaf = 0;
  };           // needed for new []
  
  //! Leaf node: links a pointrep array [length+values] to this union, saved as an offset pointer
  static inline void link(bitunion<T>* leaf, pointrep* points) {
    // use node child_pointer as offset_ptr, not pointrep
    leaf->node.child_pointer = (long)((char*)points - (char*)leaf);
  }
  
  //! Leaf node: points in the array
  inline T* getPoints() const {
    // absolute pointer
    //return &(this->points[1].v);
    // offset pointer
    return reinterpret_cast<T*>(
      reinterpret_cast<pointrep*>((char*)this + node.child_pointer) + 1
    );
  }
  
  //! Leaf node: length in the array
  inline unsigned int getLength() const {
    // absolute pointer
    //return this->points[0].length;
    // offset pointer
    return (reinterpret_cast<pointrep*>((char*)this + node.child_pointer))[0].length;
  }
  
  //! Leaf node: all points
  inline pointrep* getPointreps() const {
    return reinterpret_cast<pointrep*>((char*)this + node.child_pointer);
  }
  
  inline bitunion<T>* getChild(unsigned char index) const {
    bitunion<T> *children = (bitunion<T>*)((char*)this + this->node.child_pointer);
    for (unsigned char i = 0; i < index; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        children++;
      }
    }
    return children;
  }
  
  inline bool isValid(unsigned char index) {
    return  (  ( 1 << index ) & node.valid );
  }
 /*
  inline pointrep* getChild(unsigned char index) {
    bitunion<T> *children = (bitunion<T>*)((char*)this + this->node.child_pointer);
    return children[index].points; 
  }*/
  
  inline bool childIsLeaf(unsigned char index) {
    return (  ( 1 << index ) & node.leaf ); // if ith node is leaf get center
  }
};


// initialized in Boctree.cc, sequence intialized on startup
extern char amap[8][8];
extern char imap[8][8];
extern char sequence2ci[8][256][8];  // maps preference to index in children array for every valid_mask and every case



/**
 * @brief Octree
 * 
 * A cubic bounding box is calculated
 * from the given 3D points. Then it
 * is recusivly subdivided into smaller
 * subboxes
 */
template <typename T>
class BOctTree : public SearchTree {
public:
  BOctTree() {
  }

  template <class P>
  BOctTree(P * const* pts, int n, T voxelSize, PointType _pointtype = PointType(), bool _earlystop = false ) : pointtype(_pointtype), earlystop(_earlystop)
  {
    alloc = new PackedChunkAllocator;
    
    this->voxelSize = voxelSize;

    this->POINTDIM = pointtype.getPointDim();

    mins = alloc->allocate<T>(POINTDIM);
    maxs = alloc->allocate<T>(POINTDIM);

    // initialising
    for (unsigned int i = 0; i < POINTDIM; i++) { 
      mins[i] = pts[0][i]; 
      maxs[i] = pts[0][i];
    }

    for (unsigned int i = 0; i < POINTDIM; i++) { 
      for (int j = 1; j < n; j++) {
        mins[i] = min(mins[i], (T)pts[j][i]);
        maxs[i] = max(maxs[i], (T)pts[j][i]);
      }
    }

    center[0] = 0.5 * (mins[0] + maxs[0]);
    center[1] = 0.5 * (mins[1] + maxs[1]);
    center[2] = 0.5 * (mins[2] + maxs[2]);
    size = max(max(0.5 * (maxs[0] - mins[0]), 0.5 * (maxs[1] - mins[1])), 0.5 * (maxs[2] - mins[2]));
    size += 1.0; // for numerical reasons we increase size 

    // calculate new buckets
    T newcenter[8][3];
    T sizeNew = size / 2.0;

    for (unsigned char i = 0; i < 8; i++) {
      childcenter(center, newcenter[i], size, i);
    }
    // set up values
    uroot = alloc->allocate<bitunion<T> >();    
    root = &uroot->node;

    countPointsAndQueueFast(pts, n, newcenter, sizeNew, *root, center);
    init();
  }

  BOctTree(std::string filename) {
    alloc = new PackedChunkAllocator;
    deserialize(filename); 
    init();
  }

  template <class P>
  BOctTree(vector<P *> &pts, T voxelSize, PointType _pointtype = PointType(), bool _earlystop = false) : earlystop(_earlystop)
  {
    alloc = new PackedChunkAllocator;
    
    this->voxelSize = voxelSize;

    this->POINTDIM = pointtype.getPointDim();

    mins = alloc->allocate<T>(POINTDIM);
    maxs = alloc->allocate<T>(POINTDIM);

    // initialising
    for (unsigned int i = 0; i < POINTDIM; i++) { 
      mins[i] = pts[0][i]; 
      maxs[i] = pts[0][i];
    }

    for (unsigned int i = 0; i < POINTDIM; i++) { 
      for (unsigned int j = 1; j < pts.size(); j++) {
        mins[i] = min(mins[i], pts[j][i]);
        maxs[i] = max(maxs[i], pts[j][i]);
      }
    }

    center[0] = 0.5 * (mins[0] + maxs[0]);
    center[1] = 0.5 * (mins[1] + maxs[1]);
    center[2] = 0.5 * (mins[2] + maxs[2]);
    size = max(max(0.5 * (maxs[0] - mins[0]), 0.5 * (maxs[1] - mins[1])), 0.5 * (maxs[2] - mins[2]));
    
    size += 1.0; // for numerical reasons we increase size 

    // calculate new buckets
    T newcenter[8][3];
    T sizeNew = size / 2.0;

    for (unsigned char i = 0; i < 8; i++) {
      childcenter(center, newcenter[i], size, i);
    }
    // set up values
    uroot = alloc->allocate<bitunion<T> >();    
    root = &uroot->node;

    countPointsAndQueue(pts, newcenter, sizeNew, *root, center);
  }

  virtual ~BOctTree()
  {
    if(alloc) {
      delete alloc;
    }
  } 

  void init() {
    // compute maximal depth as well as the size of the smalles leaf
    real_voxelSize = size;
    max_depth = 1;
    while (real_voxelSize > voxelSize) {
      real_voxelSize = real_voxelSize/2.0;
      max_depth++;
    }
    
    child_bit_depth = alloc->allocate<unsigned int>(max_depth);
    child_bit_depth_inv = alloc->allocate<unsigned int>(max_depth);

    for(int d=0; d < max_depth; d++) {
      child_bit_depth[d] = 1 << (max_depth - d - 1);
      child_bit_depth_inv[d] = ~child_bit_depth[d];
    }
    
    mult = 1.0/real_voxelSize;
    add[0] = -center[0] + size;
    add[1] = -center[1] + size;
    add[2] = -center[2] + size;
    
    largest_index = child_bit_depth[0] * 2 -1;
  }
  
protected:
  
  /**
   * Serialization critical variables
   */
  //! the root of the octree
  ip::offset_ptr<bitoct> root;
  ip::offset_ptr<bitunion<T> > uroot;

  //! storing the center
  T center[3];

  //! storing the dimension
  T size;

  //! storing the voxel size
  T voxelSize;

  //! The real voxelsize of the leaves
  T real_voxelSize;

  //! Offset and real voxelsize inverse factor for manipulation points
  T add[3];
  T mult;

  //! Dimension of each point: 3 (xyz) + N (attributes)
  unsigned int POINTDIM;

  //! storing minimal and maximal values for all dimensions
  ip::offset_ptr<T> mins;
  ip::offset_ptr<T> maxs;

  //! Details of point attributes
  PointType pointtype;
  
  //! ?
  unsigned char max_depth;
  ip::offset_ptr<unsigned int> child_bit_depth;
  ip::offset_ptr<unsigned int> child_bit_depth_inv;
  int largest_index;

  /**
   * Serialization uncritical, runtime relevant variables
   */
  
  //! Threadlocal storage of parameters used in SearchTree operations
  static NNParams params[100];
  
  /**
   * Serialization uncritical, runtime irrelevant variables (constructor-stuff)
   */

  //! Whether to stop subdividing at N<10 nodes or not
  bool earlystop;

  //! Allocator used for creating nodes in the constructor
  Allocator* alloc;
  
public:
  
  inline const T* getMins() const { return &(*mins); }
  inline const T* getMaxs() const { return &(*maxs); }
  inline const T* getCenter() const { return center; }
  inline T getSize() const { return size; }
  inline unsigned int getPointdim() const { return POINTDIM; }
  inline const bitoct& getRoot() const { return *root; }
  inline unsigned int getMaxDepth() const { return max_depth; }
  
  inline void getCenter(double _center[3]) const {
    _center[0] = center[0];
    _center[1] = center[1];
    _center[2] = center[2];
  }

  void GetOctTreeCenter(vector<T*>&c) { GetOctTreeCenter(c, *root, center, size); }
  void GetOctTreeRandom(vector<T*>&c) { GetOctTreeRandom(c, *root); }
  void GetOctTreeRandom(vector<T*>&c, unsigned int ptspervoxel) { GetOctTreeRandom(c, ptspervoxel, *root); }
  void AllPoints(vector<T *> &vp) { AllPoints(*BOctTree<T>::root, vp); }

  long countNodes() { return 1 + countNodes(*root); } // computes number of inner nodes
  long countLeaves() { return countLeaves(*root); }   // computes number of leaves + points
  long countOctLeaves() { return countOctLeaves(*root); } // computes number of leaves

  void deserialize(std::string filename ) {
    char buffer[sizeof(T) * 20];
    T *p = reinterpret_cast<T*>(buffer);

    std::ifstream file;
    file.open (filename.c_str(), std::ios::in | std::ios::binary);

    // read magic bits
    file.read(buffer, 2);
    if ( buffer[0] != 'X' || buffer[1] != 'T') {
      std::cerr << "Not an octree file!!" << endl;
      file.close();
      return;
    }

    // read header
    pointtype = PointType::deserialize(file);

    file.read(buffer, 5 * sizeof(T));
    voxelSize = p[0];
    center[0] = p[1];
    center[1] = p[2];
    center[2] = p[3];
    size = p[4];

    file.read(buffer, sizeof(int));
    int *ip = reinterpret_cast<int*>(buffer);
    POINTDIM = *ip;

    mins = alloc->allocate<T>(POINTDIM);
    maxs = alloc->allocate<T>(POINTDIM);

    file.read(reinterpret_cast<char*>(&(*mins)), POINTDIM * sizeof(T));
    file.read(reinterpret_cast<char*>(&(*maxs)), POINTDIM * sizeof(T));

    // read root node
    uroot = alloc->allocate<bitunion<T> >();    
    root = &uroot->node;
    
    deserialize(file, *root);
    file.close();
  }
  
  static void deserialize(std::string filename, vector<Point> &points ) {
    char buffer[sizeof(T) * 20];

    std::ifstream file;
    file.open (filename.c_str(), std::ios::in | std::ios::binary);

    // read magic bits
    file.read(buffer, 2);
    if ( buffer[0] != 'X' || buffer[1] != 'T') {
      std::cerr << "Not an octree file!!" << endl;
      file.close();
      return;
    }

    // read header
    PointType pointtype = PointType::deserialize(file);

    file.read(buffer, 5 * sizeof(T)); // read over voxelsize, center and size
    file.read(buffer, sizeof(int));

    int *ip = reinterpret_cast<int*>(buffer);
    unsigned int POINTDIM = *ip;

    file.read(buffer, POINTDIM * sizeof(T));
    file.read(buffer, POINTDIM * sizeof(T));

    // read root node
    deserialize(file, points, pointtype);
    file.close();
  }

  void serialize(std::string filename) {
    char buffer[sizeof(T) * 20];
    T *p = reinterpret_cast<T*>(buffer);

    std::ofstream file;
    file.open (filename.c_str(), std::ios::out | std::ios::binary);

    // write magic bits
    buffer[0] = 'X';
    buffer[1] = 'T';
    file.write(buffer, 2);

    // write header
    pointtype.serialize(file);

    p[0] = voxelSize;
    p[1] = center[0]; 
    p[2] = center[1]; 
    p[3] = center[2];
    p[4] = size;

    int *ip = reinterpret_cast<int*>(&(buffer[5 * sizeof(T)]));
    *ip = POINTDIM;

    file.write(buffer, 5 * sizeof(T) + sizeof(int));


    for (unsigned int i = 0; i < POINTDIM; i++) {
      p[i] = mins[i];
    }
    for (unsigned int i = 0; i < POINTDIM; i++) {
      p[i+POINTDIM] = maxs[i];
    }

    file.write(buffer, 2*POINTDIM * sizeof(T));

    // write root node
    serialize(file, *root);

    file.close();
  }

  static PointType readType(std::string filename ) {
    char buffer[sizeof(T) * 20];

    std::ifstream file;
    file.open (filename.c_str(), std::ios::in | std::ios::binary);

    // read magic bits
    file.read(buffer, 2);
    if ( buffer[0] != 'X' || buffer[1] != 'T') {
      std::cerr << "Not an octree file!!" << endl;
      file.close();
      return PointType();
    }

    // read header
    PointType pointtype = PointType::deserialize(file);

    file.close();

    return pointtype;
  }
  

  /**
   * Picks the first point in depth first order starting from the given node
   *
   */
  T* pickPoint(bitoct &node) {
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          // absolute pointer
          //return &(children->points[1].v);
          // offset pointer
          return children->getPoints();
        } else { // recurse
          return pickPoint(children->node);
        }
        ++children; // next child
      }
    }
    return 0;
  } 

  static void childcenter(const T *pcenter, T *ccenter, T size, unsigned char i) {
    switch (i) {
      case 0:  // 000
        ccenter[0] = pcenter[0] - size / 2.0;
        ccenter[1] = pcenter[1] - size / 2.0;
        ccenter[2] = pcenter[2] - size / 2.0;
        break;
      case 1:  // 001
        ccenter[0] = pcenter[0] + size / 2.0;
        ccenter[1] = pcenter[1] - size / 2.0;
        ccenter[2] = pcenter[2] - size / 2.0;
        break;
      case 2:  // 010
        ccenter[0] = pcenter[0] - size / 2.0;
        ccenter[1] = pcenter[1] + size / 2.0;
        ccenter[2] = pcenter[2] - size / 2.0;
        break;
      case 3:  // 011
        ccenter[0] = pcenter[0] + size / 2.0;
        ccenter[1] = pcenter[1] + size / 2.0;
        ccenter[2] = pcenter[2] - size / 2.0;
        break;
      case 4:  // 100
        ccenter[0] = pcenter[0] - size / 2.0;
        ccenter[1] = pcenter[1] - size / 2.0;
        ccenter[2] = pcenter[2] + size / 2.0;
        break;
      case 5:  // 101
        ccenter[0] = pcenter[0] + size / 2.0;
        ccenter[1] = pcenter[1] - size / 2.0;
        ccenter[2] = pcenter[2] + size / 2.0;
        break;
      case 6:  // 110
        ccenter[0] = pcenter[0] - size / 2.0;
        ccenter[1] = pcenter[1] + size / 2.0;
        ccenter[2] = pcenter[2] + size / 2.0;
        break;
      case 7:  // 111
        ccenter[0] = pcenter[0] + size / 2.0;
        ccenter[1] = pcenter[1] + size / 2.0;
        ccenter[2] = pcenter[2] + size / 2.0;
        break;
      default:
        break;
    }
  }
  
  static void childcenter(int x, int y, int z, int &cx, int &cy, int &cz, char i, int size) {
    switch (i) {
      case 0:  // 000
        cx = x - size ;
        cy = y - size ;
        cz = z - size ;
        break;
      case 1:  // 001
        cx = x + size ;
        cy = y - size ;
        cz = z - size ;
        break;
      case 2:  // 010
        cx = x - size ;
        cy = y + size ;
        cz = z - size ;
        break;
      case 3:  // 011
        cx = x + size ;
        cy = y + size ;
        cz = z - size ;
        break;
      case 4:  // 100
        cx = x - size ;
        cy = y - size ;
        cz = z + size ;
        break;
      case 5:  // 101
        cx = x + size ;
        cy = y - size ;
        cz = z + size ;
        break;
      case 6:  // 110
        cx = x - size ;
        cy = y + size ;
        cz = z + size ;
        break;
      case 7:  // 111
        cx = x + size ;
        cy = y + size ;
        cz = z + size ;
        break;
      default:
        break;
    }
  }
  
protected:
  
  void AllPoints( bitoct &node, vector<T*> &vp) {
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf get center
          // absolute pointer
          //pointrep *points = children->points;
          // offset pointer
          pointrep* points = children->getPointreps();
          unsigned int length = points[0].length;
          T *point = &(points[1].v);  // first point
          for(unsigned int iterator = 0; iterator < length; iterator++ ) {
            //T *p = new T[BOctTree<T>::POINTDIM];
//            T *p = new T[3];
//            p[0] = point[0]; p[1] = point[1]; p[2] = point[2];
            T *p = new T[BOctTree<T>::POINTDIM];
            for (unsigned int k = 0; k < BOctTree<T>::POINTDIM; k++)
              p[k] = point[k];

            vp.push_back(p);

            //glVertex3f( point[0], point[1], point[2]);
            point+=BOctTree<T>::POINTDIM;
          }
        } else { // recurse
          AllPoints( children->node, vp);
        }
        ++children; // next child
      }
    }
  }
  
  static void deserialize(std::ifstream &f, vector<Point> &vpoints, PointType &pointtype) {
    char buffer[2];
    pointrep *point = new pointrep[pointtype.getPointDim()];
    f.read(buffer, 2);
    bitoct node;
    node.valid = buffer[0];
    node.leaf = buffer[1];

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf read points 
          pointrep first;
          f.read(reinterpret_cast<char*>(&first), sizeof(pointrep));
          unsigned int length = first.length;  // read first element, which is the length
          for (unsigned int k = 0; k < length; k++) {
            f.read(reinterpret_cast<char*>(point), sizeof(pointrep) * pointtype.getPointDim()); // read the points
            vpoints.push_back( pointtype.createPoint( &(point->v ) ) );
          }
        } else {  // write child 
          deserialize(f, vpoints, pointtype);
        }
      }
    }
    delete [] point;
  }

  void deserialize(std::ifstream &f, bitoct &node) {
    char buffer[2];
    f.read(buffer, 2);
    node.valid = buffer[0];
    node.leaf = buffer[1];

    unsigned short n_children = POPCOUNT(node.valid);

    // create children
    bitunion<T> *children = alloc->allocate<bitunion<T> >(n_children);    
    bitoct::link(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf read points 
          pointrep first;
          f.read(reinterpret_cast<char*>(&first), sizeof(pointrep));
          unsigned int length = first.length;  // read first element, which is the length
          pointrep *points = alloc->allocate<pointrep> (POINTDIM*length + 1);
          // absolute pointer
          //children->points = points;
          // offset pointer
          bitunion<T>::link(children, points);
          points[0] = first;
          points++;
          f.read(reinterpret_cast<char*>(points), sizeof(pointrep) * length * POINTDIM); // read the points
        } else {  // write child 
          deserialize(f, children->node);
        }
        ++children; // next child
      }
    }
  }

  void serialize(std::ofstream &of, bitoct &node) {
    char buffer[2];
    buffer[0] = node.valid;
    buffer[1] = node.leaf;
    of.write(buffer, 2);


    // write children
    bitunion<T> *children;
    bitoct::getChildren(node, children);
    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf write points
          // absolute pointer
          //pointrep *points = children->points;
          // offset pointer
          pointrep* points = children->getPointreps();
          unsigned int length = points[0].length;
          of.write(reinterpret_cast<char*>(points), sizeof(pointrep) * (length * POINTDIM  +1));
        } else {  // write child 
          serialize(of, children->node);
        }
        ++children; // next child
      }
    }
  }
  
  void GetOctTreeCenter(vector<T*>&c, bitoct &node, T *center, T size) {
    T ccenter[3];
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (unsigned char i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        childcenter(center, ccenter, size, i);  // childrens center
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf get center
          T * cp = new T[3];
          for (unsigned int iterator = 0; iterator < 3; iterator++) {
            cp[iterator] = ccenter[iterator];
          }
          c.push_back(cp);
        } else { // recurse
          GetOctTreeCenter(c, children->node, ccenter, size/2.0);
        }
        ++children; // next child
      }
    }
  }

  void GetOctTreeRandom(vector<T*>&c, bitoct &node) {
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          // absolute pointer
          //pointrep *points = children->points;
          // offset pointer
          pointrep* points = children->getPointreps();
          // new version to ignore leaves with less than 3 points
          /* 
          if(points[0].length > 2) { 
            for(int tmp = 0; tmp < points[0].length; tmp++) {
              T *point = &(points[POINTDIM*tmp+1].v);
              c.push_back(point);
            }
          }
          */  
          //old version
          
          int tmp = rand(points[0].length);
          T *point = &(points[POINTDIM*tmp+1].v);
          c.push_back(point);
          

        } else { // recurse
          GetOctTreeRandom(c, children->node);
        }
        ++children; // next child
      }
    }
  } 
  
  void GetOctTreeRandom(vector<T*>&c, unsigned int ptspervoxel, bitoct &node) {
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          // absolute pointer
          //pointrep *points = children->points;
          // offset pointer
          pointrep* points = children->getPointreps();
          unsigned int length = points[0].length;
          if (ptspervoxel >= length) {
            for (unsigned int j = 0; j < length; j++) 
              c.push_back(&(points[POINTDIM*j+1].v));

            ++children; // next child
            continue;
          }
          set<int> indices;
          while(indices.size() < ptspervoxel) {
            int tmp = rand(length-1);
            indices.insert(tmp);
          }
          for(set<int>::iterator it = indices.begin(); it != indices.end(); it++) 
            c.push_back(&(points[POINTDIM*(*it)+1].v));

        } else { // recurse
          GetOctTreeRandom(c, ptspervoxel, children->node);
        }
        ++children; // next child
      }
    }
  }

  long countNodes(bitoct &node) {
    long result = 0;
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          //++result;
        } else { // recurse
          result += countNodes(children->node) + 1;
        }
        ++children; // next child
      }
    }
    return result;
  }

  long countLeaves(bitoct &node) {
    long result = 0;
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          long nrpts = children->getLength();
          result += POINTDIM*nrpts;
        } else { // recurse
          result += countLeaves(children->node);
        }
        ++children; // next child
      }
    }
    return result;
  }
  
  long countOctLeaves(bitoct &node) {
    long result = 0;
    bitunion<T> *children;
    bitoct::getChildren(node, children);

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          result ++;
        } else { // recurse
          result += countTrueLeaves(children->node);
        }
        ++children; // next child
      }
    }
    return result;
  }

  // TODO: is this still needed? nodes and pointreps are all in the Allocator
  void deletetNodes(bitoct &node) {
    bitunion<T> *children;
    bitoct::getChildren(node, children);
    bool haschildren = false;

    for (short i = 0; i < 8; i++) {
      if (  ( 1 << i ) & node.valid ) {   // if ith node exists
        if (  ( 1 << i ) & node.leaf ) {   // if ith node is leaf
          // absolute pointer
          //delete [] children->points;
          // offset pointer
          delete [] children->getPointreps();
        } else { // recurse
          deletetNodes(children->node);
        }
        ++children; // next child
        haschildren = true;
      }
    }
    // delete children
    if (haschildren) {
      bitoct::getChildren(node, children);
    }
  }
  
  template <class P>
  void* branch( bitoct &node, P * const * splitPoints, int n,  T _center[3], T _size) {
    // if bucket is too small stop building tree
    // -----------------------------------------
    if ((_size <= voxelSize) || (earlystop && n <= 10) ) {

      // copy points
      pointrep *points = alloc->allocate<pointrep> (POINTDIM*n + 1);

      points[0].length = n;
      int i = 1;
      for (int j = 0; j < n; j++) {
        for (unsigned int iterator = 0; iterator < POINTDIM; iterator++) {
          points[i++].v = splitPoints[j][iterator];
        }
      }
      return points; 
    }  

    // calculate new buckets
    T newcenter[8][3];
    T sizeNew;

    sizeNew = _size / 2.0;

    for (unsigned char i = 0; i < 8; i++) {
      childcenter(_center, newcenter[i], _size, i);
    }

    countPointsAndQueueFast(splitPoints, n, newcenter, sizeNew, node, _center);
    return 0;
  }

  template <class P>
  void* branch( bitoct &node, vector<P*> &splitPoints, T _center[3], T _size) {
    // if bucket is too small stop building tree
    // -----------------------------------------
    if ((_size <= voxelSize) || (earlystop && splitPoints.size() <= 10) ) {
      // copy points
      pointrep *points = alloc->allocate<pointrep> (POINTDIM*splitPoints.size() + 1);
      points[0].length = splitPoints.size();
      int i = 1;
      for (typename vector<P *>::iterator itr = splitPoints.begin(); 
          itr != splitPoints.end(); itr++) {
        for (unsigned int iterator = 0; iterator < POINTDIM; iterator++) {
          points[i++].v = (*itr)[iterator];
        }
      }
      return points; 
    }  

    // calculate new buckets
    T newcenter[8][3];
    T sizeNew;

    sizeNew = _size / 2.0;

    for (unsigned char i = 0; i < 8; i++) {
      childcenter(_center, newcenter[i], _size, i);
    }

    countPointsAndQueue(splitPoints, newcenter, sizeNew, node, _center);
    return 0;
  }

  template <class P>
  void countPointsAndQueue(vector<P*> &i_points, T center[8][3], T size, bitoct &parent, T *pcenter) {
    vector<P*> points[8];
    int n_children = 0;
    for (typename vector<P *>::iterator itr = i_points.begin(); itr != i_points.end(); itr++) {
      points[childIndex<P>(pcenter, *itr)].push_back( *itr );
    }

    i_points.clear();
    vector<P*>().swap(i_points);
    for (int j = 0; j < 8; j++) {
      if (!points[j].empty()) {
        parent.valid = ( 1 << j ) | parent.valid;
        ++n_children;
      }
    }
    // create children
    bitunion<T> *children = alloc->allocate<bitunion<T> >(n_children);
    bitoct::link(parent, children);

    int count = 0;
    for (int j = 0; j < 8; j++) {
      if (!points[j].empty()) {
        pointrep *c = (pointrep*)branch(children[count].node, points[j], center[j], size);  // leaf node
        if (c) {
          // absolute pointer
          //children[count].points = c; // set this child to deque of points
          // offset pointer
          bitunion<T>::link(&children[count], c);
          parent.leaf = ( 1 << j ) | parent.leaf;  // remember this is a leaf
        }
        points[j].clear();
        vector<P*>().swap(points[j]);
        ++count;
      }
    }
  }
  
  template <class P>
  void countPointsAndQueueFast(P * const* points, int n,  T center[8][3], T size, bitoct &parent, T pcenter[3]) {
    P * const *blocks[9];
    blocks[0] = points;
    blocks[8] = points + n;
    fullsort(points, n, pcenter, blocks+1);

    int n_children = 0;
    
    for (int j = 0; j < 8; j++) {
      // if non-empty set valid flag for this child
      if (blocks[j+1] - blocks[j] > 0) {
        parent.valid = ( 1 << j ) | parent.valid;
        ++n_children;
      }
    }

    // create children
    bitunion<T> *children = alloc->allocate<bitunion<T> >(n_children);
    bitoct::link(parent, children);
    int count = 0;
    for (int j = 0; j < 8; j++) {
      if (blocks[j+1] - blocks[j] > 0) {
        pointrep *c = (pointrep*)branch(children[count].node, blocks[j], blocks[j+1] - blocks[j], center[j], size);  // leaf node
        if (c) {
          // absolute pointer
          //children[count].points = c; // set this child to vector of points
          // offset pointer
          bitunion<T>::link(&children[count], c); // set this child to vector of points
          parent.leaf = ( 1 << j ) | parent.leaf;  // remember this is a leaf
        }
        ++count;
      }
    }
  }


  void getByIndex(T *point, T *&points, unsigned int &length) {
    unsigned int x,y,z;
    x = (point[0] + add[0]) * mult;
    y = (point[1] + add[1]) * mult;
    z = (point[2] + add[2]) * mult;
    
    bitunion<T> *node = uroot; 
    unsigned char child_index;
    unsigned int child_bit;
    unsigned int depth = 0;

    while (true) {
      child_bit = child_bit_depth[depth];
      child_index = ((x & child_bit )!=0)  | (((y & child_bit )!=0 )<< 1) | (((z & child_bit )!=0) << 2);
      if (node->childIsLeaf(child_index) ) {
        node = node->getChild(child_index);
        points = node->getPoints();
        length = node->getLength();
        return;
      } else {
        node = node->getChild(child_index);
      }
      depth++;
    }
  }

  template <class P>
  inline unsigned char childIndex(const T *center, const P *point) {
    return  (point[0] > center[0] ) | ((point[1] > center[1] ) << 1) | ((point[2] > center[2] ) << 2) ;
  }

  /**
   * Given a leaf node, this function looks for the closest point to params[threadNum].closest
   * in the list of points.
   */
  inline void findClosestInLeaf(bitunion<T> *node, int threadNum) const {
    if (params[threadNum].count >= params[threadNum].max_count) return;
    params[threadNum].count++;
    T* points = node->getPoints();
    unsigned int length = node->getLength();
    for(unsigned int iterator = 0; iterator < length; iterator++ ) {
      double myd2 = Dist2(params[threadNum].p, points); 
      if (myd2 < params[threadNum].closest_d2) {
        params[threadNum].closest_d2 = myd2;
        params[threadNum].closest = points;
        if (myd2 <= 0.0001) {
          params[threadNum].closest_v = 0; // the search radius in units of voxelSize
        } else {
          params[threadNum].closest_v = sqrt(myd2) * mult + 1; // the search radius in units of voxelSize
        }
      }
      points+=BOctTree<T>::POINTDIM;
    }
  }
  


/** 
 * This function finds the closest point in the octree given a specified
 * radius. This implementation is quit complex, although it is already
 * simplified. The simplification incurs a significant loss in speed, as
 * several calculations have to be performed repeatedly and a high number of
 * unnecessary jumps are executed.
 */
  double *FindClosest(double *point, double maxdist2, int threadNum) const
  {
    params[threadNum].closest = 0; // no point found currently
    params[threadNum].closest_d2 = maxdist2;
    params[threadNum].p = point;
    params[threadNum].x = (point[0] + add[0]) * mult;
    params[threadNum].y = (point[1] + add[1]) * mult;
    params[threadNum].z = (point[2] + add[2]) * mult;
    params[threadNum].closest_v = sqrt(maxdist2) * mult + 1; // the search radius in units of voxelSize
    params[threadNum].count = 0;
    params[threadNum].max_count = 10000; // stop looking after this many buckets

   
    // box within bounds in voxel coordinates
    int xmin, ymin, zmin, xmax, ymax, zmax;
    xmin = max(params[threadNum].x-params[threadNum].closest_v, 0); 
    ymin = max(params[threadNum].y-params[threadNum].closest_v, 0); 
    zmin = max(params[threadNum].z-params[threadNum].closest_v, 0);

//    int largest_index = child_bit_depth[0] * 2 -1;
    
    xmax = min(params[threadNum].x+params[threadNum].closest_v, largest_index);
    ymax = min(params[threadNum].y+params[threadNum].closest_v, largest_index);
    zmax = min(params[threadNum].z+params[threadNum].closest_v, largest_index);
    
    unsigned char depth = 0;
    unsigned int child_bit;
    unsigned int child_index_min;
    unsigned int child_index_max;

    bitunion<T> *node = &(*uroot);

    int cx, cy, cz;
    
    child_bit = child_bit_depth[depth];
    cx = child_bit_depth[depth];
    cy = child_bit_depth[depth];
    cz = child_bit_depth[depth];

    while (true) { // find the first node where branching is required
      child_index_min = ((xmin & child_bit )!=0)  | (((ymin & child_bit )!=0 )<< 1) | (((zmin & child_bit )!=0) << 2);
      child_index_max = ((xmax & child_bit )!=0)  | (((ymax & child_bit )!=0 )<< 1) | (((zmax & child_bit )!=0) << 2);

      // if these are the same, go there
      // TODO: optimization: also traverse if only single child...
      if (child_index_min == child_index_max) {
        if (node->childIsLeaf(child_index_min) ) {  // luckily, no branching is required
          findClosestInLeaf(node->getChild(child_index_min), threadNum);
          return static_cast<double*>(params[threadNum].closest);
        } else {
          if (node->isValid(child_index_min) ) { // only descend when there is a child
            childcenter(cx,cy,cz, cx,cy,cz, child_index_min, child_bit/2 ); 
            node = node->getChild(child_index_min);
            child_bit /= 2;
          } else {  // there is no child containing the bounding box => no point is close enough
            return 0;
          }
        }
      } else {
        // if min and max are not in the same child we must branch
        break;
      }
    }
    
    // node contains all box-within-bounds cells, now begin best bin first search
    _FindClosest(threadNum, node->node, child_bit/2, cx, cy, cz);
    return static_cast<double*>(params[threadNum].closest);
  }
  
  /**
   * This is the heavy duty search function doing most of the (theoretically unneccesary) work. The tree is recursively searched.
   * Depending on which of the 8 child-voxels is closer to the query point, the children are examined in a special order.
   * This order is defined in map, imap is its inverse and sequence2ci is a speedup structure for faster access to the child indices. 
   */
  void _FindClosest(int threadNum, bitoct &node, int size, int x, int y, int z) const
  {
    // Recursive case
   
    // compute which child is closest to the query point
    unsigned char child_index =  ((params[threadNum].x - x) >= 0) | 
                                (((params[threadNum].y - y) >= 0) << 1) | 
                                (((params[threadNum].z - z) >= 0) << 2);
    
    char *seq2ci = sequence2ci[child_index][node.valid];  // maps preference to index in children array
    char *mmap = amap[child_index];  // maps preference to area index 

    bitunion<T> *children;
    bitoct::getChildren(node, children);
    int cx, cy, cz;
    cx = cy = cz = 0; // just to shut up the compiler warnings
    for (unsigned char i = 0; i < 8; i++) { // in order of preference
      child_index = mmap[i]; // the area index of the node 
      if (  ( 1 << child_index ) & node.valid ) {   // if ith node exists
        childcenter(x,y,z, cx,cy,cz, child_index, size); 
        if ( params[threadNum].closest_v == 0 ||  max(max(abs( cx - params[threadNum].x ), 
                 abs( cy - params[threadNum].y )),
                 abs( cz - params[threadNum].z )) - size
        > params[threadNum].closest_v ) { 
          continue;
        }
        // find the closest point in leaf seq2ci[i] 
        if (  ( 1 << child_index ) & node.leaf ) {   // if ith node is leaf
          findClosestInLeaf( &children[seq2ci[i]], threadNum);
        } else { // recurse
          _FindClosest(threadNum, children[seq2ci[i]].node, size/2, cx, cy, cz);
        }
      }
    }
  }


  /** 
   * This function shows the possible speedup that can be gained by using the
   * octree for nearest neighbour search, if a more sophisticated
   * implementation were given. Here, only the bucket in which the query point
   * falls is looked up. If doing the same thing in the kd-tree search, this
   * function is about 3-5 times as fast
   */
  double *FindClosestInBucket(double *point, double maxdist2, int threadNum) {
    params[threadNum].closest = 0;
    params[threadNum].closest_d2 = maxdist2;
    params[threadNum].p = point;
    unsigned int x,y,z;
    x = (point[0] + add[0]) * mult;
    y = (point[1] + add[1]) * mult;
    z = (point[2] + add[2]) * mult;
    T * points;
    unsigned int length;

    bitunion<T> *node = uroot; 
    unsigned char child_index;

    unsigned int  child_bit = child_bit_depth[0];

    while (true) {
      child_index = ((x & child_bit )!=0)  | (((y & child_bit )!=0 )<< 1) | (((z & child_bit )!=0) << 2);
      if (node->childIsLeaf(child_index) ) {
        node = node->getChild(child_index);
        points = node->getPoints();
        length = node->getLength();
        
        for(unsigned int iterator = 0; iterator < length; iterator++ ) {
          double myd2 = Dist2(params[threadNum].p, points); 
          if (myd2 < params[threadNum].closest_d2) {
            params[threadNum].closest_d2 = myd2;
            params[threadNum].closest = points;
          }
          points+=BOctTree<T>::POINTDIM;
        }
        return static_cast<double*>(params[threadNum].closest);
      } else {
        if (node->isValid(child_index) ) {
          node = node->getChild(child_index);
        } else {
          return 0;
        }
      }
      child_bit >>= 1;
    }
    return static_cast<double*>(params[threadNum].closest);
  }
  

template <class P>
  void fullsort(P * const * points, int n, T splitval[3], P * const * blocks[9]) {
    P* const * L0;
    P* const * L1;
    P* const * L2;
    unsigned int n0L, n0R, n1L, n1R ;

    // sort along Z
    L0 = sort(points, n, splitval[2], 2);
      
      n0L = L0 - points;
      // sort along Y (left of Z)   points -- L0
      L1 = sort(points, n0L, splitval[1], 1);

        n1L = L1 - points; 
        // sort along X (left of Y)  points -- L1
        L2 = sort(points, n1L, splitval[0], 0);
        
          blocks[0] = L2;
        
        n1R = n0L - n1L;
        // sort along X (right of Y) // L1 -- L0
        L2 = sort(L1, n1R, splitval[0], 0);
        
          blocks[1] = L1;
          blocks[2] = L2;

      n0R = n - n0L;
      // sort along Y (right of Z)  L0 -- end
      L1 = sort(L0, n0R, splitval[1], 1);
      
        n1L = L1 - L0; 
        // sort along X (left of Y)  points -- L1
        L2 = sort(L0, n1L, splitval[0], 0);
        
          blocks[3] = L0;
          blocks[4] = L2;
        
        n1R = n0R - n1L;
        // sort along X (right of Y) // L1 -- L0
        L2 = sort(L1, n1R, splitval[0], 0);
        
          blocks[5] = L1;
          blocks[6] = L2;
  }


  template <class P>
  P* const * sort(P* const * points, unsigned int n, T splitval, unsigned char index) {
    if (n==0) return points;
    
    if (n==1) {
      if (points[0][index] < splitval)
        return points+1;
      else
        return points;
    }

    P **left = const_cast<P**>(points);
    P **right = const_cast<P**>(points + n - 1);
    
    
      while (1) {
      while ((*left)[index] < splitval) 
      {
        left++;
        if (right < left)
          break;
      }
      while ((*right)[index] >= splitval)
      {
        right--;
        if (right < left)
          break;
      }
      if (right < left)
        break;

      std::swap(*left, *right);
    }
    return left;
  }

public:
  /**
   * Copies another (via new constructed) octtree into cache allocated memory and makes it position independant
   */
  BOctTree(const BOctTree& other, unsigned char* mem_ptr, unsigned int mem_max)
  {
    alloc = new SequentialAllocator(mem_ptr, mem_max);
    
    // "allocate" space for *this
    alloc->allocate<BOctTree<T> >();
    
    // take members
    unsigned int i;
    for(i = 0; i < 3; ++i)
      center[i] = other.center[i];
    size = other.size;
    voxelSize = other.voxelSize;
    real_voxelSize = other.real_voxelSize;
    for(i = 0; i < 3; ++i)
      add[i] = other.add[i];
    mult = other.mult;
    POINTDIM = other.POINTDIM;
    mins = alloc->allocate<T>(POINTDIM);
    maxs = alloc->allocate<T>(POINTDIM);
    for(i = 0; i < POINTDIM; ++i) {
      mins[i] = other.mins[i];
      maxs[i] = other.maxs[i];
    }
    pointtype = other.pointtype;
    max_depth = other.max_depth;
    child_bit_depth = alloc->allocate<unsigned int>(max_depth);
    child_bit_depth_inv = alloc->allocate<unsigned int>(max_depth);
    for(i = 0; i < max_depth; ++i) {
      child_bit_depth[i] = other.child_bit_depth[i];
      child_bit_depth_inv[i] = other.child_bit_depth_inv[i];
    }
    largest_index = other.largest_index;
    
    // take node structure
    uroot = alloc->allocate<bitunion<T> >();  
    root = &uroot->node;
    copy_children(*other.root, *root);
    
    // test if allocator has used up his space
    //alloc->printSize();
    
    // discard allocator, space is managed by the cache manager
    delete alloc; alloc = 0;
  }

private:
  void copy_children(const bitoct& other, bitoct& my) {
    // copy node attributes
    my.valid = other.valid;
    my.leaf = other.leaf;
    
    // other children
    bitunion<T>* other_children;
    bitoct::getChildren(other, other_children);
    
    // create own children
    unsigned int n_children = POPCOUNT(other.valid);
    bitunion<T>* my_children = alloc->allocate<bitunion<T> >(n_children);    
    bitoct::link(my, my_children);
    
    // iterate over all (valid) children and copy them
    for(unsigned int i = 0; i < 8; ++i) {
      if((1<<i) & other.valid) {
        if((1<<i) & other.leaf) {
          // copy points
          unsigned int length = other_children->getLength();
          pointrep* other_pointreps = other_children->getPointreps();
          pointrep* my_pointreps = alloc->allocate<pointrep>(POINTDIM * length + 1);
          for(unsigned int j = 0; j < POINTDIM * length + 1; ++j)
            my_pointreps[j] = other_pointreps[j];
          // assign
          bitunion<T>::link(my_children, my_pointreps);
        } else {
          // child is already created, copy and create children for it
          copy_children(other_children->node, my_children->node);
        }
        ++other_children;
        ++my_children;
      }
    }
  }

public:
  //! Size of the whole tree structure, including the main class, its serialize critical allocated variables and nodes, not the allocator
  unsigned int getMemorySize()
  {
    return sizeof(*this) // all member variables
      + 2*POINTDIM*sizeof(T) // mins, maxs
      + 2*max_depth*sizeof(unsigned int) // child_bit_depth(_inv)
      + sizeof(bitunion<T>) // uroot
      + sizeChildren(*root); // all nodes
  }
  
private:
  //! Recursive size of a node's children
  unsigned int sizeChildren(const bitoct& node) {
    unsigned int s = 0;
    bitunion<T>* children;
    bitoct::getChildren(node, children);
    
    // size of children allocation
    unsigned int n_children = POPCOUNT(node.valid);
    s += sizeof(bitunion<T>)*n_children;
    
    // iterate over all (valid) children and sum them up
    for(unsigned int i = 0; i < 8; ++i) {
      if((1<<i) & node.valid) {
        if((1<<i) & node.leaf) {
          // leaf only accounts for its points
          s += sizeof(pointrep)*(children->getLength()*POINTDIM+1);
        } else {
          // childe node is already accounted for, add its children
          s += sizeChildren(children->node);
        }
        ++children; // next (valid) child
      }
    }
    return s;
  }
};

typedef SingleObject<BOctTree<float> > DataOcttree;

template <class T>
NNParams BOctTree<T>::params[100];

#endif
