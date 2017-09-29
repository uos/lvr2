#ifndef __SPARSE_VECTOR_HPP__
#define __SPARSE_VECTOR_HPP__

#include "SparseVector.hpp"

namespace lvr
{
/**
 * @brief Representation of a sparse matrix
 */
class SparseMatrix
{
    /// Array of sparse vectors to represent the columns
    SparseVector** m_rows;

    /// Number of colums of the matrix
    int m_numRows;

    /// Number of rows of the matrix
    int m_numColumns;

    /// Helper to copy a sparse matrix
    void copy(const SparseMatrix& other);

    /// Helper to delete all contends of a sparse matrix
    void clear();

public:
    /**
     * Default constructor. Initializes all values with zero.
     */
    SparseMatrix();

    /**
     * Copy constructor.
     */
    SparseMatrix(const SparseMatrix& other);

    /**
     * Constructs a matrix with \ref rows rows and \ref columns columns
     */
    SparseMatrix(const int& rows, const int& cols);

    /**
     * Destructor
     */
    ~SparseMatrix();

    /**
     * Assignment operator
     */
    SparseMatrix& operator=(const SparseMatrix& other);

    /**
     * Inserts an integer \ref value at given \ref row and \ref column
     *
     * @param row
     * @param col
     * @param value
     */
    void insert(const int& row, const int& col, const int& value);

    /// Returns the sparse vector at given \ref row
    SparseVector& operator[](int row) const;
};

}  // namespace jumper

#endif
