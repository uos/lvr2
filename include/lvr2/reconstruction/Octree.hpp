#ifndef Octree_hpp
#define Octree_hpp

#include "CellHandle.hh"
#include "Location.hh"
#include "OctreeTables.hpp"

#include <vector>
#include <float.h>
#include <iostream>

//=============================================================================
namespace lvr2 {
//=============================================================================

/*!
    @class C_Octree
    An octree implementation using techniques described in

    "Simple and efficient traversal methods for quadtrees and octrees",
    Sarah F. Frisken, Ronald N. Perry
    Journal of Graphics Tools, 7(3):1--11, 2002
*/

template <typename BaseVecT, typename BoxT, typename T_CellData>
class C_Octree
{

public:
    typedef Location::LocCode LocCode;
    typedef float             Scalar;

    //! Maximal root level defined by size of data type of the cell coordinates (leafs have side length 1)
    static const int MAX_ROOT_LEVEL = sizeof(LocCode)*8-1;

    inline C_Octree() { }
    inline ~C_Octree() { }

    //! @name cell
    //@{
        inline bool is_leaf    ( CellHandle _ch ) const;
        inline bool is_extraction_leaf    ( CellHandle _ch ) const;
        inline bool is_root    ( CellHandle _ch ) const;
        inline int  level      ( CellHandle _ch ) const;
        inline int  depth      ( CellHandle _ch ) const;
        inline bool is_inner_boundary( CellHandle _ch ) const;

        int getChildIndex( BaseVecT center, coord<float> *point );
        int getChildIndex( BaseVecT center, BaseVecT point );

        //! Split the cell, i.e. create eight children
        void split( CellHandle _ch, const bool bGarbageCollectionOnTheFly=false );

        //! Set the cell handle of the picked cell
        inline void SetPickedCell( CellHandle _ch )
        {
            m_PickedCell = _ch;
        }

        //! Reset the cell handle of the picked cell
        inline void ResetPickedCell()
        {
            m_PickedCell = CellHandle();
        }

        //! Pick the parent cell of the currently picked cell
        inline bool PickParentCell()
        {
            if (parent(m_PickedCell).is_valid())
            {
                m_PickedCell = parent(m_PickedCell);
                return true;
            }
            return false;
        }

        //! Pick the child cell of the currently picked cell
        inline bool PickChildCell(int _idx)
        {
            if (child(m_PickedCell, _idx).is_valid())
            {
                m_PickedCell = child(m_PickedCell, _idx);
                return true;
            }
            return false;
        }

        //! Reset the number of generated cells
        inline void ResetNumberOfGeneratedCells() { m_NumberOfGeneratedCells = 1; }

        //! Set the level to be treated as leaf level
        inline LocCode SetExtractionLevel(LocCode _level) { return m_ExtractionLevel=_level; }
    //@}

    //! @name iterators
    //@{
        inline CellHandle root() const;
        inline CellHandle end()  const { return CellHandle( size() ); }
    //@}

    //! @name navigation
    //@{
        inline CellHandle parent         ( CellHandle _ch )           const;
        inline CellHandle child          ( CellHandle _ch, int _idx ) const;
        inline CellHandle face_neighbor  ( CellHandle _ch, int _idx ) const;
        inline CellHandle edge_neighbor  ( CellHandle _ch, int _idx ) const;
        inline std::pair<std::vector<CellHandle>, std::vector<uint> > all_corner_neighbors( CellHandle _ch, int _idx ) const;
        inline CellHandle corner_neighbor( CellHandle _ch, int _idx ) const;

        //! Moving the picked cell, i.e. picking one of the neighbouring cells (if possible)
        void MovePickedCell(int _key, double* _mvm);
        void MovePickedCellLeft(double* _mvm);
        void MovePickedCellRight(double* _mvm);
        void MovePickedCellUp(double* _mvm);
        void MovePickedCellDown(double* _mvm);
    //@}

    //! @name query
    //@{
        inline int size() const;
        inline int nr_cells() const;
        inline BaseVecT cell_corner( CellHandle _ch, int _idx ) const
        {
            Scalar x, y, z;
            cell_corner( _ch, _idx, x, y, z );
            BaseVecT point;
            point[0] = x;
            point[1] = y;
            point[2] = z;
            return point;
        }
        inline void  cell_corner( CellHandle _ch,
                        Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _size ) const;
        inline void  cell_corner( CellHandle _ch, int _idx,
                        Scalar & _x, Scalar & _y, Scalar & _z ) const;
        inline void  cell_corner( CellHandle _ch, int _idx,
                        Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _size ) const;
        inline BaseVecT cell_center( CellHandle _ch ) const
        {
            Scalar x, y, z;
            cell_center( _ch, x, y, z );
            BaseVecT point;
            point[0] = x;
            point[1] = y;
            point[2] = z;
            return point;
        }
        inline void  cell_center( CellHandle _ch,
                        Scalar & _x, Scalar & _y, Scalar & _z ) const;
        inline void  cell_center( CellHandle _ch,
                        Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _size ) const;
        inline Scalar cell_size( CellHandle _ch ) const
        {
            const Location & locinfo = location( _ch );
            LocCode binary_cell_size = 1 << locinfo.level();
            return binary_cell_size;
        }
        Location & location( CellHandle _h )
        {
            return m_OctreeCell[_h.idx()].location;
        }
        const Location & location( CellHandle _h ) const
        {
            return m_OctreeCell[_h.idx()].location;
        }
        inline CellHandle cell( Scalar _x, Scalar _y, Scalar _z ) const;

        //! Get the handle of the picked cell
        inline CellHandle GetPickedCell() const { return m_PickedCell; }

        //! Get number of generated cells
        inline int GetNumberOfGeneratedCells() const { return m_NumberOfGeneratedCells; }

        //! Get the level to be treated as leaf level
        inline LocCode GetExtractionLevel() const { return m_ExtractionLevel; }
    //@}

    //! @name memory
    //@{
        inline void reserve( unsigned int _size );
        inline void resize ( unsigned int _size );

        //! Initializing the octree with maximal level = root level (leafs have level 0)
        void initialize( int _max_octree_level )
        {
            // Security check...
            if ( _max_octree_level > MAX_ROOT_LEVEL )
            {
                fprintf( stderr, "Error: octree::initialize(): invalid root level\n" );
                exit( EXIT_FAILURE );
            }

            m_rootLevel = _max_octree_level;

            clear();
        }

        inline void clear()
        {
            // clear cell vector
            std::vector<T_CellData>().swap(m_OctreeCell);

            // 7 dummies and...
            T_CellData dummy;
            for (size_t i=0; i<7; ++i) m_OctreeCell.push_back( dummy );

            // ...the root cell at the beginning
            dummy.location = Location( 0, 0, 0, m_rootLevel, CellHandle() );
            m_OctreeCell.push_back( dummy );

            // Initialize the pointer to the next free block
            m_nextFreeBlock = 8;

            // Reset the number of generated cells
            ResetNumberOfGeneratedCells();

            // Reset the leaf-level
            m_ExtractionLevel=0;
        }
    //@}

protected:
    inline int cell( int _i ) const
    {
        return m_OctreeCell[ _i ].next;
    }

    inline void set_cell( int _i, int _cell )
    {
        m_OctreeCell[ _i ].next = _cell;
    }

    inline CellHandle traverse( CellHandle _ch, LocCode _loc_x, LocCode _loc_y, LocCode _loc_z ) const;
    inline CellHandle traverse_to_level( CellHandle _ch, LocCode _loc_x, LocCode _loc_y, LocCode _loc_z, LocCode _level ) const;

    inline CellHandle get_common_ancestor( CellHandle _ch, LocCode _diff ) const;
    inline CellHandle get_common_ancestor( CellHandle _ch, LocCode _diff0, LocCode _diff1 ) const;
    inline CellHandle get_common_ancestor( CellHandle _ch, LocCode _diff0, LocCode _diff1, LocCode _diff2 ) const;

    // ==================================================================================================== \/
    // ============================================================================================= FIELDS \/
    // ==================================================================================================== \/
protected:

    //! Root level
    LocCode m_rootLevel;

    //! Pointer to the next free block (garbage collection)
    size_t m_nextFreeBlock;

    //! The handle of the picked octree cell
    CellHandle m_PickedCell;

    //! Number of octree cells generated; this counter is adjusted in the split routine only
    int m_NumberOfGeneratedCells;

    //! Level of the cells to be treated as leafs during surface extraction
    LocCode m_ExtractionLevel;

public:
    //! Vector containing the cells
    std::vector<T_CellData> m_OctreeCell;
};

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::reserve( unsigned int _size )
{
    m_OctreeCell.reserve( _size );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::resize( unsigned int _size )
{
    m_OctreeCell.resize( _size );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::child( CellHandle _ch, int _idx ) const
{
    return CellHandle( m_OctreeCell[ _ch.idx() ].next + _idx );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::parent( CellHandle _ch ) const
{
    return location( _ch ).parent();
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::root() const
{
    return CellHandle( 7 );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
bool C_Octree< BaseVecT, BoxT, T_CellData >::is_leaf( CellHandle _ch ) const
{
    return (m_OctreeCell[ _ch.idx() ].next<0);
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
bool C_Octree< BaseVecT, BoxT, T_CellData >::is_extraction_leaf( CellHandle _ch ) const
{
    return ((m_OctreeCell[ _ch.idx() ].next<0) || (level(_ch)==m_ExtractionLevel));
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
bool C_Octree< BaseVecT, BoxT, T_CellData >::is_root( CellHandle _ch ) const
{
    return _ch.idx() == 7;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::cell_corner( CellHandle _ch, Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _size ) const
{
    const Location & locinfo = location( _ch );

    LocCode binary_cell_size = 1 << locinfo.level();

    _size = ( Scalar ) binary_cell_size;
    _x = ( Scalar ) locinfo.loc_x();
    _y = ( Scalar ) locinfo.loc_y();
    _z = ( Scalar ) locinfo.loc_z();
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::cell_center( CellHandle _ch, Scalar & _x, Scalar & _y, Scalar & _z ) const
{
    const Location & locinfo = location( _ch );
    LocCode binary_cell_size = 1 << locinfo.level();

    Scalar size = ( Scalar ) binary_cell_size;

    _x = ( Scalar ) locinfo.loc_x() + ( Scalar ) 0.5 * size;
    _y = ( Scalar ) locinfo.loc_y() + ( Scalar ) 0.5 * size;
    _z = ( Scalar ) locinfo.loc_z() + ( Scalar ) 0.5 * size;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::cell_center( CellHandle _ch, Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _s ) const
{
    const Location & locinfo = location( _ch );
    LocCode binary_cell_size = 1 << locinfo.level();

    _s = ( Scalar ) binary_cell_size;

    _x = ( Scalar ) locinfo.loc_x() + ( Scalar ) 0.5 * _s;
    _y = ( Scalar ) locinfo.loc_y() + ( Scalar ) 0.5 * _s;
    _z = ( Scalar ) locinfo.loc_z() + ( Scalar ) 0.5 * _s;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::cell_corner( CellHandle _ch, int _idx, Scalar & _x, Scalar & _y, Scalar & _z, Scalar & _size ) const
{
    const Location & locinfo = location( _ch );

    LocCode binary_cell_size = 1 << locinfo.level();

    _size = ( Scalar ) binary_cell_size;
    _x = ( Scalar ) locinfo.loc_x();
    _y = ( Scalar ) locinfo.loc_y();
    _z = ( Scalar ) locinfo.loc_z();

    switch ( _idx )
    {
        case 0 : break;
        case 1 : _x += _size; break;
        case 2 : _y += _size; break;
        case 3 : _x += _size; _y += _size; break;
        case 4 : _z += _size; break;
        case 5 : _x += _size; _z += _size; break;
        case 6 : _y += _size; _z += _size; break;
        case 7 : _x += _size; _y += _size; _z += _size; break;
    }
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::cell_corner( CellHandle _ch, int _idx,	Scalar & _x, Scalar & _y, Scalar & _z ) const
{
    const Location & locinfo = location( _ch );

    LocCode binary_cell_size = 1 << locinfo.level();

    Scalar size = ( Scalar ) binary_cell_size;
    _x = ( Scalar ) locinfo.loc_x();
    _y = ( Scalar ) locinfo.loc_y();
    _z = ( Scalar ) locinfo.loc_z();

    switch ( _idx )
    {
        case 0 : break;
        case 1 : _x += size; break;
        case 2 : _y += size; break;
        case 3 : _x += size; _y += size; break;
        case 4 : _z += size; break;
        case 5 : _x += size; _z += size; break;
        case 6 : _y += size; _z += size; break;
        case 7 : _x += size; _y += size; _z += size; break;
    }
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::traverse(
            CellHandle _ch,
            LocCode _loc_x,
            LocCode _loc_y,
            LocCode _loc_z ) const
{
    LocCode next_level = location( _ch ).level() - 1;

    while ( ! is_leaf( _ch ) )
    {
        LocCode child_branch_bit = 1 << next_level;
        unsigned int   child_index =
        ( ( _loc_x & child_branch_bit ) >> ( next_level ) ) +
        2 * ( ( _loc_y & child_branch_bit ) >> ( next_level ) ) +
        4 * ( ( _loc_z & child_branch_bit ) >> ( next_level ) );
        --next_level;
        _ch = child( _ch, child_index );
    }

    return _ch;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::traverse_to_level(
                CellHandle _ch,
                LocCode _loc_x,
                LocCode _loc_y,
                LocCode _loc_z,
                LocCode _level ) const
{
    LocCode next_level = level( _ch ) - 1;

    LocCode n = next_level - _level + 1;

    while ( n-- )
    {
        LocCode child_branch_bit = 1 << next_level;
        unsigned int   child_index =
        1 * ( ( _loc_x & child_branch_bit ) >> ( next_level ) ) +
        2 * ( ( _loc_y & child_branch_bit ) >> ( next_level ) ) +
        4 * ( ( _loc_z & child_branch_bit ) >> ( next_level ) );
        --next_level;
        _ch = child( _ch, child_index );

        if ( is_leaf( _ch ) ) break;
    }

    return _ch;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::face_neighbor( CellHandle _ch, int _idx ) const
{
    const Location & locinfo = location( _ch );
    const LocCode loc_x = locinfo.loc_x();
    const LocCode loc_y = locinfo.loc_y();
    const LocCode loc_z = locinfo.loc_z();
    //std::cout << "Loc: " << locinfo.level() << " " << std::endl;
    const LocCode binary_cell_size = ((int)1) << locinfo.level();

    LocCode new_loc_x = loc_x;
    LocCode new_loc_y = loc_y;
    LocCode new_loc_z = loc_z;
    LocCode diff      = 0;

    switch ( _idx )
    {
        case 0: // right
        if ( loc_x + binary_cell_size >= ( 1 << m_rootLevel ) ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        diff = loc_x ^ new_loc_x;
        break;

        case 1: // top
        if ( loc_y + binary_cell_size >= ( 1 << m_rootLevel ) ) return CellHandle();
        new_loc_y = loc_y + binary_cell_size;
        diff = loc_y ^ new_loc_y;
        break;

        case 2: // front
        if ( loc_z + binary_cell_size >= ( 1 << m_rootLevel ) ) return CellHandle();
        new_loc_z = loc_z + binary_cell_size;
        diff = loc_z ^ new_loc_z;
        break;

        case 3: // back
        if ( loc_z == 0 ) return CellHandle();
        new_loc_z = loc_z - 1;
        diff = loc_z ^ new_loc_z;
        break;

        case 4: // bottom
        if ( loc_y == 0 ) return CellHandle();
        new_loc_y = loc_y - 1;
        diff = loc_y ^ new_loc_y;
        break;

        case 5: // left
        if ( loc_x == 0 ) return CellHandle();
        new_loc_x = loc_x - 1;
        diff = loc_x ^ new_loc_x;
        break;
    }

    CellHandle parent = get_common_ancestor( _ch, diff );

    return traverse_to_level( parent, new_loc_x, new_loc_y, new_loc_z, locinfo.level() );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::get_common_ancestor( CellHandle _ch, LocCode _binary_diff ) const
{
    LocCode cell_level = level( _ch );

    while ( _binary_diff & ( 1 << cell_level ) )
    {
        _ch = parent( _ch );
        ++cell_level;
    }

    return _ch;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::get_common_ancestor(
                CellHandle _ch,
                LocCode _binary_diff0,
                LocCode _binary_diff1 ) const
{
    LocCode cell_level = level( _ch );

    while ( _binary_diff0 & _binary_diff1 & ( 1 << cell_level ) )
    {
        _ch = parent( _ch );
        ++cell_level;
    }

    if ( _binary_diff0 & ( 1 << cell_level ) )
        while ( _binary_diff0 & ( 1 << cell_level ) )
        {
        _ch = parent( _ch );
        ++cell_level;
        }
    else
        while ( _binary_diff1 & ( 1 << cell_level ) )
        {
        _ch = parent( _ch );
        ++cell_level;
        }

    return _ch;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::get_common_ancestor(
                CellHandle _ch,
                LocCode _binary_diff0,
                LocCode _binary_diff1,
                LocCode _binary_diff2 ) const
{
    return CellHandle( 0 );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::edge_neighbor( CellHandle _ch, int _idx ) const
{
    const Location & locinfo = location( _ch );

    LocCode loc_x = locinfo.loc_x();
    LocCode loc_y = locinfo.loc_y();
    LocCode loc_z = locinfo.loc_z();

    LocCode binary_cell_size = 1 << locinfo.level();

    LocCode new_loc_x = loc_x;
    LocCode new_loc_y = loc_y;
    LocCode new_loc_z = loc_z;

    int extent = ( 1 << m_rootLevel );

    switch ( _idx )
    {
        case 0 :
        {
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z - 1;
        break;
        }
        case 1 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_z = loc_z - 1;
        break;
        }
        case 2 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y - 1;
        break;
        }
        case 3 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_z = loc_z - 1;
        break;
        }
        case 4 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y + binary_cell_size;
        break;
        }
        case 5 :
        {
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 6 :
        {
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z - 1;
        break;
        }
        case 7 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y - 1;
        break;
        }
        case 8 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 9 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y + binary_cell_size;
        break;
        }
        case 10 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 11 :
        {
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }

    }

    CellHandle parent = root();

    return traverse_to_level( parent, new_loc_x, new_loc_y, new_loc_z, locinfo.level() );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
std::pair<std::vector<CellHandle>, std::vector<uint> > C_Octree< BaseVecT, BoxT, T_CellData >::all_corner_neighbors(CellHandle _ch, int _idx) const
{
    std::vector<CellHandle> cellHandles;
    std::vector<uint> markers;
    std::vector<Location*> locations;

    // get lower left back corner of given cell
    Location locinfo = location( _ch );
    LocCode loc_x = locinfo.loc_x();
    LocCode loc_y = locinfo.loc_y();
    LocCode loc_z = locinfo.loc_z();
    LocCode binary_cell_size = 1 << locinfo.level();
    int extent = ( 1 << m_rootLevel );

    // shift loc_x, loc_y, loc_z to corresponding corner position
    LocCode loc_X = loc_x + binary_cell_size * octreeVertexTable[_idx][0];
    LocCode loc_Y = loc_y + binary_cell_size * octreeVertexTable[_idx][1];
    LocCode loc_Z = loc_z + binary_cell_size * octreeVertexTable[_idx][2];

    // calculate the corner positions
    for(unsigned char i = 0; i < 8; i++)
    {
        // marker is binary code for detailed overhang position
        // z-axis-overhang y-axis-overhang x-axis-overhang
        uint marker = 0;

        int new_loc_x = loc_X;
        int new_loc_y = loc_Y;
        int new_loc_z = loc_Z;

        new_loc_x = loc_X + octreeCornerNeighborTable[i][0];
        if ( new_loc_x >= extent || new_loc_x < 0 )
        {
            new_loc_x = loc_x;
            marker = 1;
        }

        new_loc_y = loc_Y + octreeCornerNeighborTable[i][1];
        if ( new_loc_y >= extent || new_loc_y < 0 )
        {
            new_loc_y = loc_y;
            marker |= 1 << 1;
        }

        new_loc_z = loc_Z + octreeCornerNeighborTable[i][2];
        if ( new_loc_z >= extent || new_loc_z < 0 )
        {
            new_loc_z = loc_z;
            marker |= 1 << 2;
        }

        locations.push_back(new Location(new_loc_x, new_loc_y, new_loc_z, locinfo.level() + 1, locinfo.parent()));
        markers.push_back(marker);
    }

    for(Location * location : locations)
    {
        cellHandles.push_back(traverse(root(), location->loc_x(), location->loc_y(), location->loc_z()));
        delete location;
    }
    locations.clear();
    std::vector<Location*>().swap(locations);

    return make_pair(cellHandles, markers);
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::corner_neighbor( CellHandle _ch, int _idx ) const
{
    const Location & locinfo = location( _ch );

    LocCode loc_x = locinfo.loc_x();
    LocCode loc_y = locinfo.loc_y();
    LocCode loc_z = locinfo.loc_z();

    LocCode binary_cell_size = 1 << locinfo.level();

    LocCode new_loc_x = loc_x;
    LocCode new_loc_y = loc_y;
    LocCode new_loc_z = loc_z;

    int extent = ( 1 << m_rootLevel );

    switch ( _idx )
    {
        case 0 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z - 1;
        break;
        }
        case 1 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z - 1;
        break;
        }
        case 2 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z - 1;
        break;
        }
        case 3 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z == 0 ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z - 1;
        break;
        }
        case 4 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 5 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y == 0 ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y - 1;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 6 :
        {
        if ( loc_x == 0 ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x - 1;
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
        case 7 :
        {
        if ( loc_x + binary_cell_size >= extent ) return CellHandle();
        if ( loc_y + binary_cell_size >= extent ) return CellHandle();
        if ( loc_z + binary_cell_size >= extent ) return CellHandle();
        new_loc_x = loc_x + binary_cell_size;
        new_loc_y = loc_y + binary_cell_size;
        new_loc_z = loc_z + binary_cell_size;
        break;
        }
    }

    CellHandle parent = root();

    return traverse_to_level( parent, new_loc_x, new_loc_y, new_loc_z, locinfo.level() );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::level( CellHandle _ch ) const
{
    return location( _ch ).level();
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::depth( CellHandle _ch ) const
{
    return m_rootLevel - level( _ch );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::size() const
{
    return m_OctreeCell.size();
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::nr_cells() const
{
    // #cells ~ #elements in the m_OctreeCell vector minus 7 dummies at the beginning
    return m_OctreeCell.size()-7;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
bool C_Octree< BaseVecT, BoxT, T_CellData >::is_inner_boundary( CellHandle _ch ) const
{
    const Location & locinfo = location( _ch );

    LocCode loc_x = locinfo.loc_x();
    LocCode loc_y = locinfo.loc_y();
    LocCode loc_z = locinfo.loc_z();

    LocCode binary_cell_size = 1 << locinfo.level();

    LocCode tree_size = 1 << m_rootLevel;

    return ( loc_x == 0 ||
        loc_y == 0 ||
        loc_z == 0 ||
        loc_x + binary_cell_size == tree_size ||
        loc_y + binary_cell_size == tree_size ||
        loc_z + binary_cell_size == tree_size );
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
CellHandle C_Octree< BaseVecT, BoxT, T_CellData >::cell( Scalar _x, Scalar _y, Scalar _z ) const
{
    if ( _x < 0 ) return CellHandle();
    if ( _y < 0 ) return CellHandle();
    if ( _z < 0 ) return CellHandle();

    Scalar extent( 1 << m_rootLevel );

    if ( _x > extent ) return CellHandle();
    if ( _y > extent ) return CellHandle();
    if ( _z > extent ) return CellHandle();

    CellHandle ch = root();

    while ( ! is_leaf( ch ) )
    {
        Scalar cx, cy, cz;
        cell_center( ch, cx, cy, cz );

        unsigned int cidx = ( _x < cx ? 0 : 1 )
                + ( _y < cy ? 0 : 2 )
                + ( _z < cz ? 0 : 4 );
        ch = child( ch, cidx );
    }

    return ch;
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::MovePickedCellLeft(double* _mvm)
{
    MovePickedCell(0, _mvm);
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::MovePickedCellRight(double* _mvm)
{
    MovePickedCell(1, _mvm);
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::MovePickedCellUp(double* _mvm)
{
    MovePickedCell(2, _mvm);
}

//-----------------------------------------------------------------------------

template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::MovePickedCellDown(double* _mvm)
{
    MovePickedCell(3, _mvm);
}

//-----------------------------------------------------------------------------

/*!
    \param _direction Direction of the movement
    \param _mvm The current modelview matrix needed for estimating the plane of movement

    The rows of the current modelview matrix correspond to the camera axes in the world space.
    We use here the maximal components of the x- and y-axis as "guides" in which direction to
    move in the octree. This approach works only because the octree cells are world-axis-aligned!
*/
template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::MovePickedCell(int _direction, double* _mvm)
{
    bool left = (_direction == 0);
    bool right = (_direction == 1);
    bool up = (_direction == 2);
    bool down = (_direction == 3);

    BaseVecT xAxis(_mvm[0], _mvm[4], _mvm[8]);
    BaseVecT yAxis(_mvm[1], _mvm[5], _mvm[9]);

    // ============================================================================================
    // 1) Determine the movement axes
    int xIndex=2;
    if (fabs(xAxis[0])>fabs(xAxis[1]))
    {
        if (fabs(xAxis[0])>fabs(xAxis[2]))
            xIndex=0;
    }
    else if (fabs(xAxis[1])>fabs(xAxis[2]))
        xIndex=1;

    int yIndex=2;
    if (fabs(yAxis[0])>fabs(yAxis[1]))
    {
        if (fabs(yAxis[0])>fabs(yAxis[2]))
            yIndex=0;
    }
    else if (fabs(yAxis[1])>fabs(yAxis[2]))
        yIndex=1;

    // ============================================================================================
    // 2) Move the cell along the x-axis
    CellHandle neighCell;
    if (xIndex==0)
    {
        if (xAxis[xIndex]<0)
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 0);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 5);
        }
        else
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 5);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 0);
        }
    }
    else if (xIndex==1)
    {
        if (xAxis[xIndex]<0)
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 1);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 4);
        }
        else
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 4);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 1);
        }
    }
    else // if (xIndex==2)
    {
        if (xAxis[xIndex]<0)
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 2);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 3);
        }
        else
        {
            if (left)
                neighCell = face_neighbor(m_PickedCell, 3);
            else if (right)
                neighCell = face_neighbor(m_PickedCell, 2);
        }
    }

    // ============================================================================================
    // 3) Move the cell along the y-axis
    if (yIndex==0)
    {
        if (yAxis[yIndex]<0)
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 0);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 5);
        }
        else
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 5);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 0);
        }
    }
    else if (yIndex==1)
    {
        if (yAxis[yIndex]<0)
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 1);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 4);
        }
        else
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 4);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 1);
        }
    }
    else // if (yIndex==2)
    {
        if (yAxis[yIndex]<0)
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 2);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 3);
        }
        else
        {
            if (down)
                neighCell = face_neighbor(m_PickedCell, 3);
            else if (up)
                neighCell = face_neighbor(m_PickedCell, 2);
        }
    }

    // ============================================================================================
    // 3) Check whether the movement can be done
    if ((neighCell.is_valid()) && (level(neighCell)==level(m_PickedCell)))
    {
        m_PickedCell = neighCell;
    }
}

//-----------------------------------------------------------------------------

/*!
    \param center The center of a cell in real world coordinates
    \param point Point for to check it's position relative to the cell center

    INDICES:
    // 000 left down behind
    // 001 right down behind
    // 010 left top behind
    // 011 right top behind
    // 100 left down front
    // 101 right down front
    // 110 left top front
    // 111 right top front
*/
template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::getChildIndex( BaseVecT center, coord<float> *point )
{
    return (((*point)[0]) > center[0] ) | ((((*point)[1]) > center[1] ) << 1) | ((((*point)[2]) > center[2] ) << 2);
}

template <typename BaseVecT, typename BoxT, typename T_CellData>
int C_Octree< BaseVecT, BoxT, T_CellData >::getChildIndex( BaseVecT center, BaseVecT point )
{
    return ((point[0]) > center[0] ) | (((point[1]) > center[1] ) << 1) | (((point[2]) > center[2] ) << 2);
}

//-----------------------------------------------------------------------------

/*!
    \param _parent The cell handle of the parent cell
    \param bGarbageCollectionOnTheFly true, if garbage collection should be done

    The following method simply splits an octree cell and generates 8 children cells. Depending
    on the value of bGarbageCollectionOnTheFly EITHER (false; default value) the new cells are
    simply pushed back in the vector(s) OR (true) we fill the next free block with the new cells,
    i.e. that the pointer to the next free block "m_nextFreeBlock" must not pointer to the end
    of the vector(s).
*/
template <typename BaseVecT, typename BoxT, typename T_CellData>
void C_Octree< BaseVecT, BoxT, T_CellData >::split( CellHandle _parent, const bool bGarbageCollectionOnTheFly )
{
    //
    // Split a cell into 8 sub-cells.
    //
    //   2---3
    //  /|  /|
    // 6---7 |
    // | 0-|-1
    // |/  |/
    // 4---5
    //

    // At the beginning do not forget the counter
    m_NumberOfGeneratedCells += 8;

    // If the next free block is at the end of the vector then splitting as usual
    LocCode cell_level = level( _parent ) - 1;
    LocCode cell_bit_mask = ( 1 << cell_level );
    LocCode par_loc_x = location( _parent ).loc_x();
    LocCode par_loc_y = location( _parent ).loc_y();
    LocCode par_loc_z = location( _parent ).loc_z();

    if (!bGarbageCollectionOnTheFly || (m_nextFreeBlock==m_OctreeCell.size()))
    {
        m_OctreeCell[ _parent.idx() ].next = m_OctreeCell.size();

        T_CellData dummy;

        dummy.location = Location( par_loc_x, par_loc_y, par_loc_z, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x | cell_bit_mask, par_loc_y, par_loc_z, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x, par_loc_y | cell_bit_mask, par_loc_z, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x | cell_bit_mask, par_loc_y | cell_bit_mask, par_loc_z, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x, par_loc_y, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x | cell_bit_mask, par_loc_y, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x, par_loc_y | cell_bit_mask, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell.push_back( dummy );

        dummy.location = Location( par_loc_x | cell_bit_mask, par_loc_y | cell_bit_mask, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell.push_back( dummy );
    }
    // ..else we do not push back but fill in the empty block
    else
    {
        m_OctreeCell[ _parent.idx() ].next = m_nextFreeBlock;

        m_OctreeCell[m_nextFreeBlock].location   =
            Location( par_loc_x, par_loc_y, par_loc_z, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+1].location =
            Location( par_loc_x | cell_bit_mask, par_loc_y, par_loc_z, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+2].location =
            Location( par_loc_x, par_loc_y | cell_bit_mask, par_loc_z, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+3].location =
            Location( par_loc_x | cell_bit_mask, par_loc_y | cell_bit_mask, par_loc_z, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+4].location =
            Location( par_loc_x, par_loc_y, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+5].location =
            Location( par_loc_x | cell_bit_mask, par_loc_y, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+6].location =
            Location( par_loc_x, par_loc_y | cell_bit_mask, par_loc_z | cell_bit_mask, cell_level, _parent );
        m_OctreeCell[m_nextFreeBlock+7].location =
            Location( par_loc_x | cell_bit_mask, par_loc_y | cell_bit_mask, par_loc_z | cell_bit_mask, cell_level, _parent );
    }

    // Set the pointer to the next free block
    while ((m_nextFreeBlock<m_OctreeCell.size()) && m_OctreeCell[m_nextFreeBlock].location.parent().is_valid()) m_nextFreeBlock+=8;
}

//=============================================================================
} // namespace octree
//=============================================================================

#endif
