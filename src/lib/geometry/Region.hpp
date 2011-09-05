/*
 * Region.h
 *
 *  Created on: 18.08.2011
 *      Author: PG2011
 */

#ifndef REGION_H_
#define REGION_H_

#include "Vertex.hpp"
#include "Normal.hpp"
#include "HalfEdgeVertex.hpp"
#include "HalfEdgeFace.hpp"
#include <vector>
#include <stack>


namespace lssr {

template<typename VertexT, typename NormalT>
class HalfEdgeFace;

template<typename VertexT, typename NormalT>
class HalfEdgeVertex;

template<typename VertexT, typename NormalT>
class HalfEdgeMesh;


/**
 * @brief 	This class represents a region.
 */
template<typename VertexT, typename NormalT>
class Region
{

public:

	typedef HalfEdgeFace<VertexT, NormalT> HFace;
	typedef HalfEdgeVertex<VertexT, NormalT> HVertex;
	typedef HalfEdge<HVertex, HFace> HEdge;

	/**
	 * @brief constructor.
	 *
	 * @param 	region_number 	the number of the region
	 */
	Region(int region_number);

	/**
	 * @brief Adds a face to the region.
	 *
	 * @param	f	the face to add
	 */
	virtual void addFace(HFace* f);

	/**
	 * @brief Removes a face from the region.
	 *
	 * @param	f	the face to remove
	 */
	virtual void removeFace(HFace* f);

	/**
	 * @brief Finds all contours of the region (outer contour + holes)
	 *
	 * @param	epsilon	controls the number of points used for a contour
	 *
	 * @return 	a list of all contours
	 */
	virtual vector<stack<HVertex*> > getContours(float epsilon);

	/**
	 * @brief caluclates a regression plane for the region and fits all faces into this plane
	 *
	 */
	virtual void regressionPlane();

	/**
	 * @brief	Finds faces whose normals are aligned to the wrong direction and flips them back.
	 * 			Will result in less flickering effects.
	 *
	 * @param	mesh	A pointer to the mesh
	 */
	virtual void backflipFaces(HalfEdgeMesh<VertexT, NormalT>* mesh);

    /**
     * @brief the number of faces contained in this region
     */
    virtual int size();

	/**
	 * @brief destructor.
	 */
	virtual ~Region();

	/// Indicates if the region was dragged into a regression plane
	bool m_inPlane;

	/// The faces in the region
	vector<HFace*>    m_faces;

	/// The number of the region
	int m_region_number;

	/// The normal of the region (updated every time regressionPlane() is called)
	NormalT m_normal;
	
    /**
	 * @brief calculates a valid normal of the region
	 *
	 * @return a normal of the region
	 */
	virtual NormalT calcNormal();

private:

};
}

#include "Region.tcc"

#endif /* REGION_H_ */
