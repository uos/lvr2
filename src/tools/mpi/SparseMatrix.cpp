#include "SparseMatrix.hpp"

namespace lvr
{
SparseMatrix::SparseMatrix(const int& rows, const int& cols)
{
    // Save width and height
    m_numRows = rows;
    m_numColumns = cols;

    // Create rows
    m_rows = new SparseVector* [rows];
    for (int i = 0; i < rows; i++)
    {
        m_rows[i] = new SparseVector(cols);
    }
}

void SparseMatrix::copy(const SparseMatrix& other)
{
    // Check if rows was already used and free resources
    // if necessary.
    if (m_rows)
    {
        clear();
    }

    // Copy width and height
    m_numRows = other.m_numRows;
    m_numColumns = other.m_numColumns;

    // Create rows
    m_rows = new SparseVector* [other.m_numRows];

    // Crate sparse vectors for columns
    for (int i = 0; i < m_numRows; i++)
    {
        m_rows[i] = new SparseVector(other[i]);
    }
}

void SparseMatrix::clear()
{
    if (m_rows)
    {
        for (int i = 0; i < m_numRows; i++)
        {
            delete m_rows[i];
        }
    }
    delete[] m_rows;
    m_rows = 0;
}

SparseMatrix::SparseMatrix()
{
    m_numRows = 0;
    m_numColumns = 0;
    m_rows = 0;
}

SparseMatrix::SparseMatrix(const SparseMatrix& other)
{
    copy(other);
}

SparseMatrix::~SparseMatrix()
{
    clear();
}

SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other)
{
    if (&other != this)
    {
        clear();
        copy(other);
    }
    return *this;
}

SparseVector& SparseMatrix::operator[](int index) const
{
    return *m_rows[index];
}

void SparseMatrix::insert(const int& row, const int& col, const int& value)
{
    m_rows[row]->setElem(col, value);
}

}  // namespace jumper