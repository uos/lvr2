/*
 * KdTree.h
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

#ifndef KDTREE_H_
#define KDTREE_H_

#include "geometry/BoundingBox.hpp"
#include "io/ModelFactory.hpp"
#include "io/PointBuffer.hpp"
#include "geometry/Vertex.hpp"
#include "io/Model.hpp"

#include "io/PLYIO.hpp"
#include "io/AsciiIO.hpp"
#include "io/UosIO.hpp"

// vorerst
#include <vector>
#include <iostream>
#include <list>
#include "KdNode.h"
//#include "/home/student/d/dofeldsc/Projekt_MPI/meshing.dofeldsc/Las vegas Projekt/src/lib/geometry/BoundingBox.hpp"
//#include "/home/student/d/dofeldsc/Projekt_MPI/meshing.dofeldsc/Las vegas Projekt/src/lib/io/ModelFactory.hpp"
//#include "/home/student/d/dofeldsc/Projekt_MPI/meshing.dofeldsc/Las vegas Projekt/src/lib/io/PointBuffer.hpp"
namespace lssr{
template<typename VertexT>
class KdTree {
public:
	KdTree();
	virtual ~KdTree();

	/**
	 * @brief Der Konstruktor speichert die Punktwolke ab.
	 */
	KdTree(PointBufferPtr pointcloud);

	//Funktion, welche die Rekursion startet
	void Rekursion(KdNode<VertexT> * first);

	void splitPointcloud(KdNode<VertexT> * child);

	/* Membervariable für die Punktwolke */
	PointBufferPtr         m_loader;

	/* Membervariable für die Indices der Punktwolke */
	coord3fArr m_points;

    /// The bounding box of the point cloud
    BoundingBox<VertexT>   m_boundingBox;

    /* Liste zum verwalten der Nodes */
    std::list<KdNode<VertexT>*> nodelist;

/* The Bounce of the BoundingBox */
	float m_xmin;

	float m_xmax;

	float m_ymin;

	float m_ymax;

	float m_zmin;

	float m_zmax;

	/*
	 * x = 0
	 * y = 1
	 * z = 2
	 */
	int splitaxis;
	float split;

	/* Anzahl der Punkte in der Punktwolke */
	size_t m_numpoint;

/* Anzahl der Buckets pro Datenpaket */
	double m_bucketsize;

/* Ein shared-Pointer für das Model, indem die Punktwolke steckt */
	//boost::shared_ptr<ModelPtr>    m_model;
	ModelPtr m_model;
};
}

#include "KdTree.tcc"
#endif /* KDTREE_H_ */
