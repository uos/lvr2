/*
 * HashGrid.hpp
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#ifndef _HASHGRID_HPP_
#define _HASHGRID_HPP_

#include <unordered_map>
#include <vector>
#include <string>

#include "reconstruction/QueryPoint.hpp"
#include "geometry/BoundingBox.hpp"

using std::string;
using std::vector;
using std::unordered_map;

namespace lvr
{

template<typename VertexT, typename BoxT>
class HashGrid
{
public:

	/// Typedef to alias box map
	typedef unordered_map<size_t, BoxT*> 			box_map;

	/// Typedef to alias iterators for box maps
	typedef unordered_map<size_t, BoxT*>::iterator  box_map_it;

	/***
	 * @brief	Constructor
	 *
	 * If set to true, cell size is interpreted as
	 * absolute voxel size (default). Otherwise \ref cellSize
	 * is interpreted as number of intersections along the
	 * longest size of the given bounding box to estimate a suitable
	 * resolution.
	 *
	 * @param 	cellSize		Voxel size of the grid cells
	 * @param	isVoxelSize		Whether to interpret \ref cellSize as voxelsize or intersections
	 */
	HashGrid(float cellSize, BoundingBox<VertexT> boundingBox, bool isVoxelSize = true);

	/**
	 *
	 * @param i 		Discrete x position within the grid.
	 * @param j			Discrete y position within the grid.
	 * @param k			Discrete z position within the grid.
	 * @param distance	Signed distance to the represented surface
	 * 					at the position within the grid.
	 */
	void addLatticePoint(size_t i, size_t j, size_t k, float distance = 0.0);

	/**
	 * @brief	Saves a representation of the grid to the given file
	 *
	 * @param file		Output file name.
	 */
	void saveGrid(string file);

	/***
	 * @brief	Is extrude is set to true, additional cells within the
	 * 			grid will be create to fill up holes consisting of single
	 * 			cells.
	 *
	 * @param extrude	If set to true, additional cells will be created.
	 * 					Default value is true.
	 */
	void setExtrusion(bool extrude) {m_extrude = extrude;}

	/***
	 * @brief 	Returns the number of generated cells.
	 */
	size_t getNumberOfCells() {return m_cells.size();}

	/**
	 * @return	Returns an iterator to the first box in the cell map.
	 */
	box_map_it	first() {return m_cells.begin();}

	/**
	 * @return 	Returns an iterator to the last box in the cell map.
	 */
	box_map_it	last() {return m_cells.end();}


	/***
	 * @brief	Destructor
	 */
	virtual ~HashGrid();

private:

	/***
	 * @brief	Searches for a existing shared lattice point in the grid.
	 *
	 * @param position	Number of a possible neighbor
	 * @param x			x index within the grid
	 * @param y			y index within the grid
	 * @param z			z index within the grid
	 * @return			Query point index of the found point, INVALID_INDEX otherwise
	 */
	unsigned int findQueryPoint(
			const int &position,
			const int &x,
			const int &y,
			const int &z);

    /**
     * @brief Calculates the hash value for the given index triple
     */
    inline size_t hashValue(int i, int j, int k) const
    {
        return i * m_maxIndexSquare + j * m_maxIndex + k;
    }

	/// Map to handle the boxes in the grid
	box_map			m_cells;

	/// The voxelsize used for reconstruction
	float                  		m_voxelsize;

	/// The absolute maximal index of the reconstruction grid
	size_t                      m_maxIndex;

	/// The squared maximal index of the reconstruction grid
	size_t                      m_maxIndexSquare;

	/// The maximal index in x direction
	size_t                      m_maxIndexX;

	/// The maximal index in y direction
	size_t                      m_maxIndexY;

	/// The maximal index in z direction
	size_t                      m_maxIndexZ;

    /// A vector containing the query points for the reconstruction
    vector<QueryPoint<VertexT> > m_queryPoints;

    /// True if a local tetraeder decomposition is used for reconstruction
    string                      m_boxType;

    /// True if we want to create extra boxes at the end of the grid
    bool                        m_extrude;

    /// Bounding box of the covered volume
    BoundingBox<VertexT>		m_boundingBox;

    /// The maximum used cell index within the grid
    unsigned int				m_globalIndex;
};

} /* namespace lvr */

#endif /* _HASHGRID_HPP_ */
