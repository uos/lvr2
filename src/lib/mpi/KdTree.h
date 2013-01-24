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
#include <cstring>
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
	/**
	 * @brief Diese Funktion startet die Rekursion
	 *
	 * @param first Die Punktwolke, welche aufgeteilt werden soll
	 */
	void Rekursion(KdNode<VertexT> * first);


	/**
	 * @brief Diese Funktion teilt die Punktwolke an ihrer laengsten Achse in zwei Teilpunktwolken auf.
	 *
	 *        Wenn eine Teilpunktwolke weniger als die maximale Anzahl an Punkten enthält wird diese also 3d-Datei abgespeichert.
	 */
	void splitPointcloud(KdNode<VertexT> * child);


	/* Membervariable für die gesamte Punktwolke */
	PointBufferPtr         m_loader;

	/* Membervariable für die Indices der Punktwolke */
	coord3fArr   			m_points;

    /// The bounding box of the point cloud
    BoundingBox<VertexT>   m_boundingBox;

    /* Liste zum verwalten der Nodes */
    std::list<KdNode<VertexT>*> nodelist;





	/* Anzahl der Punkte in der Punktwolke */
	size_t m_numpoint;

/* Anzahl der Buckets pro Datenpaket */
	double m_bucketsize;

/* Ein shared-Pointer für das Model, indem die Punktwolke steckt */
	ModelPtr m_model;
};
}

#include "KdTree.tcc"
#endif /* KDTREE_H_ */
