//=============================================================================
//
// CLASS CellHandle
//
// Handle to access octree cells.
//
//=============================================================================

#ifndef OCTREE_CELLHANDLE_HH
#define OCTREE_CELLHANDLE_HH

//=============================================================================

namespace lvr2 {

//=============================================================================

/*!
	@class CellHandle
	Handling of cells...negative indices identify invalid cells.
*/
class CellHandle
{
public:
	explicit CellHandle( int _idx = -8 )
	: idx_( _idx )
	{}

	int  idx() const { return idx_; }
	void idx( int _idx ) { idx_ = _idx; }

	bool is_valid()   const { return idx_ >= 0; }
	bool is_invalid() const { return idx_ <  0; }

	void reset() { idx_ = -8; }

	bool operator==( const CellHandle & _other ) const
	{ return idx_ == _other.idx_; }

	bool operator!=( const CellHandle & _other ) const
	{ return idx_ != _other.idx_; }

	bool operator<( const CellHandle & _other ) const
	{ return idx_ < _other.idx_; }

	void operator++() { __increment(); }

	void __increment() { ++idx_; }
	void __decrement() { --idx_; }

private:
	int idx_;
};

//=============================================================================
} // namespace lvr2
//=============================================================================
#endif // OCTREE_CELLHANDLE_HH defined
//=============================================================================
