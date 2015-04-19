/*
 * TSDFGrid.hpp
 *
 *  November 29, 2014
 *  Author: Tristan Igelbrink
 */

#ifndef _TSDFGRID_HPP_
#define _TSDFGRID_HPP_

#include "HashGrid.hpp"


namespace lvr
{

template<typename VertexT, typename BoxT, typename TsdfT>
class TsdfGrid: public HashGrid<VertexT, BoxT>
{
public:

	typedef unordered_map<size_t, BoxT*> box_map;
	typedef unordered_map<size_t, size_t> qp_map;
	typedef unordered_map<size_t, uint*> smallBox_map;;
	
	typedef unordered_map<size_t, size_t>::iterator qp_map_iterator;
	/// Typedef to alias iterators for box maps
	typedef typename unordered_map<size_t, BoxT*>::iterator  box_map_it;;

	/// Typedef to alias iterators to query points
	typedef typename vector<QueryPoint<VertexT> >::iterator	query_point_it;
	
	TsdfGrid(float cellSize,  BoundingBox<VertexT> bb, TsdfT* tsdf, size_t size,
			int shiftX, int shiftY, int shiftZ,
			TsdfGrid<VertexT, BoxT, TsdfT>* lastGrid, bool isVoxelsize = true);
	virtual ~TsdfGrid();
	void addTSDFLatticePoint(int index_x, int index_y, int index_z, float distance);
	box_map getFusionCells() { return m_fusion_cells;}
	qp_map getFusionIndices() { return m_fusion_qpIndices;}
	int repairCell(BoxT* box, int index_x, int index_y, int index_z, int corner, vector<size_t>& boxQps);
	vector<QueryPoint<VertexT> > getFusionPoints() {return m_fusionPoints;}
	inline int calcIndex(float f)
    {
        return f < 0 ? f-.5:f+.5;
    }
    box_map m_fusion_cells;
    qp_map	m_fusion_qpIndices;
    smallBox_map m_global_cells;
	vector<QueryPoint<VertexT> > m_fusionPoints;
    int m_fusionIndex_x;
    int m_fusionIndex_y;
    int m_fusionIndex_z;
    int m_fusionIndex;
    
};

} /* namespace lvr */

#include "TSDFGrid.tcc"


#endif /* INCLUDE_LIBLVR_RECONSTRUCTION_TsdfGrid_HPP_ */
