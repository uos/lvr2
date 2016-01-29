/*
 * MeshUpdateThread.hpp
 *
 *  Created on: Jan 29, 2016
 *      Author: twiemann
 */

#ifndef EXT_KINTINUOUS_QT_APP_MESHUPDATETHREAD_HPP_
#define EXT_KINTINUOUS_QT_APP_MESHUPDATETHREAD_HPP_

#include <kfusion/kinfu.hpp>
#include <QThread>

#include <vtkSmartPointer.h>
#include <vtkActor.h>

class MeshUpdateThread : public QThread
{
public:
	MeshUpdateThread(kfusion::KinFu::Ptr kinfu);
	virtual ~MeshUpdateThread();

private:
	typedef lvr::HalfEdgeVertex<cVertex, lvr::Normal<float> >* VertexPtr;

	void run();
	void computeMeshActor(HMesh* meshbuffer);

	kfusion::KinFu::Ptr 						m_kinfu;
	unordered_map<VertexPtr, size_t> 			m_indexMap;
    vtkSmartPointer<vtkActor>       			m_meshActor;
    vtkSmartPointer<vtkActor>       			m_wireframeActor;
};

#endif /* EXT_KINTINUOUS_QT_APP_MESHUPDATETHREAD_HPP_ */
