//=============================================================================
//
// CLASS Location
//
// This class encodes the location of an octree cell, i.e. the
// position of its lower left back corner, its size and its parent
// cell.
//
//=============================================================================

#ifndef OCTREE_LOCATION_HH
#define OCTREE_LOCATION_HH

//== INCLUDES =================================================================

#include "CellHandle.hh"

//== NAMESPACES ===============================================================

namespace lvr2 {

//== CLASS DEFINITION =========================================================

class Location
{
public:
	//! This datatype determines the maximal depth of the octree
	typedef unsigned short LocCode; // formerly short

	//! Standard constructor
	Location()
	: loc_x_( 0 ),	loc_y_( 0 ), loc_z_( 0 ), level_( 0 ), parent_( CellHandle() )
	{}

	Location( LocCode _loc_x, LocCode _loc_y, LocCode _loc_z, LocCode _level, CellHandle _parent )
	: loc_x_( _loc_x ),	loc_y_( _loc_y ), loc_z_( _loc_z ), level_( _level ), parent_( _parent )
	{}

	CellHandle parent() const { return parent_; }
	LocCode    loc_x()  const { return loc_x_; }
	LocCode    loc_y()  const { return loc_y_; }
	LocCode    loc_z()  const { return loc_z_; }
	LocCode    level()  const { return level_; }

	void set_parent( CellHandle _parent ) { parent_ = _parent; }
	void set_loc_x ( LocCode _loc_x ) { loc_x_ = _loc_x; }
	void set_loc_y ( LocCode _loc_y ) { loc_y_ = _loc_y; }
	void set_loc_z ( LocCode _loc_z ) { loc_z_ = _loc_z; }
	void set_level ( LocCode _level ) { level_ = _level; }

	Location neighbor( int _idx ) const {
	LocCode binary_cell_size = 1 << level();
	switch( _idx ) {
		case 0 :
	return Location( loc_x() + binary_cell_size, loc_y(), loc_z(), level(), CellHandle() );
		case 1 :
	return Location( loc_x(), loc_y() + binary_cell_size, loc_z(), level(), CellHandle() );
		case 2 :
	return Location( loc_x(), loc_y(), loc_z() + binary_cell_size, level(), CellHandle() );
		case 3 :
	return Location( loc_x(), loc_y(), loc_z() - binary_cell_size, level(), CellHandle() );
		case 4 :
	return Location( loc_x(), loc_y() - binary_cell_size, loc_z(), level(), CellHandle() );
		case 5 :
	return Location( loc_x() - binary_cell_size, loc_y(), loc_z(), level(), CellHandle() );
	}
	return (*this);
	}

private:
	LocCode loc_x_;
	LocCode loc_y_;
	LocCode loc_z_;
	LocCode level_;
	CellHandle parent_;
};

//=============================================================================
} // namespace octree
//=============================================================================
#endif // OCTREE_LOCATION_HH defined
//=============================================================================
