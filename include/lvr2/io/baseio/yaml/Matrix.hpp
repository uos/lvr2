
#ifndef LVR2_IO_YAML_MATRIX_HPP
#define LVR2_IO_YAML_MATRIX_HPP

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <iostream>

namespace YAML
{

bool isMatrix(const Node& node);

template <class Scalar_, int A_, int B_, int C_, int D_, int E_>
struct convert<Eigen::Matrix<Scalar_, A_, B_, C_, D_, E_> > 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    template <class Scalar, int A, int B, int C, int D, int E>
    static Node encode(const Eigen::Matrix<Scalar, A, B, C, D, E>& M) 
    {
        typedef typename Eigen::Matrix<Scalar, A, B, C, D, E>::Index IndexType;

        IndexType rows = M.rows();
        IndexType cols = M.cols();

        Node node = Load("[]");
        for (IndexType i = 0; i < rows; ++i) {
            Node row = Load("[]");
            for (IndexType j = 0; j < cols; ++j) {
                row.push_back(M.coeff(i, j));
            }
            node.push_back(row);
        }

        return node;
    }

    template <class Scalar, int A, int B, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, A, B, C, D, E>& M) 
    {
        typedef typename Eigen::Matrix<Scalar, A, B, C, D, E>::Index IndexType;
        
        if(!isMatrix(node))
        {
            return false;
        }

        
        // count rows and cols
        IndexType rows = 0;
        IndexType cols = 0;

        // count rows
        YAML::const_iterator row_it = node.begin();
        YAML::const_iterator row_it_end = node.end();
        while(row_it != row_it_end)
        {
            rows++;
            ++row_it;
        }

        if(A != rows)
        {
            std::cout << "[YAML::convert<Matrix> - decode] rows in yaml (" << rows << ") differ from static matrix rows (" << A << ")." << std::endl;
            
            std::cout << node << std::endl;
            return false;
        }

        // count cols
        row_it = node.begin();
        YAML::const_iterator col_it = row_it->begin();
        YAML::const_iterator col_it_end = row_it->end();
        while(col_it != col_it_end)
        {
            cols++;
            ++col_it;
        }

        if(B != cols)
        {
            std::cout << "[YAML::convert<Matrix> - decode] cols in yaml (" << cols << ") differ from static matrix cols (" << B << ")." << std::endl;
            return false;
        }

        // Load data
        row_it = node.begin();
        row_it_end = node.end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                col_it = row_it->begin();
                for (IndexType j = 0; j < cols; ++j) {
                    try {
                        M.coeffRef(i, j) = col_it->as<Scalar>();
                    } catch(const YAML::TypedBadConversion<Scalar>& ex) {
                        std::cerr << "[YAML - Matrix - decode] ERROR: Could not decode matrix entry (" << i << ", " << j << "): " << *col_it << " to scalar" << std::endl;
                        return false;
                    }
                    ++col_it;
                }
                ++row_it;
            }
        }

        return true;
    }

    template <class Scalar, int B, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, Eigen::Dynamic, B, C, D, E>& M) 
    {

        typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, B, C, D, E>::Index IndexType;

        if(!isMatrix(node))
        {
            return false;
        }
        
        // count rows and cols
        IndexType rows = 0;
        IndexType cols = 0;

        // count rows
        YAML::const_iterator row_it = node.begin();
        YAML::const_iterator row_it_end = node.end();
        while(row_it != row_it_end)
        {
            rows++;
            ++row_it;
        }

        // count cols
        row_it = node.begin();
        YAML::const_iterator col_it = row_it->begin();
        YAML::const_iterator col_it_end = row_it->end();
        while(col_it != col_it_end)
        {
            cols++;
            ++col_it;
        }

        if(B != cols)
        {
            std::cout << "[YAML::convert<Matrix> - decode] cols in yaml (" << cols << ") differ from col-static matrix cols (" << B << ")." << std::endl;
            return false;
        }

        M.resize(rows, Eigen::NoChange);

        // Load data
        row_it = node.begin();
        row_it_end = node.end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                col_it = row_it->begin();
                for (IndexType j = 0; j < cols; ++j) {
                    try {
                        M.coeffRef(i, j) = col_it->as<Scalar>();
                    } catch(const YAML::TypedBadConversion<Scalar>& ex) {
                        std::cerr << "[YAML - Matrix - decode] ERROR: Could not decode matrix entry (" << i << ", " << j << "): " << *col_it << " to scalar" << std::endl;
                        return false;
                    }
                    ++col_it;
                }
                ++row_it;
            }
        }

        return true;
    }

    /**
     * Specialization for Eigen matrices of dynamic sized columns  
     */
    template <class Scalar, int A, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, A, Eigen::Dynamic, C, D, E>& M) 
    {
        typedef typename Eigen::Matrix<Scalar, A, Eigen::Dynamic, C, D, E>::Index IndexType;

        if(!isMatrix(node))
        {
            return false;
        }

        
        // count rows and cols
        IndexType rows = 0;
        IndexType cols = 0;

        // count rows
        YAML::const_iterator row_it = node.begin();
        YAML::const_iterator row_it_end = node.end();
        while(row_it != row_it_end)
        {
            rows++;
            ++row_it;
        }

        if(A != rows)
        {
            std::cout << "[YAML - Matrix - decode] rows in yaml (" << rows << ") differ from row-static matrix rows (" << A << ")." << std::endl;
            return false;
        }

        // count cols
        row_it = node.begin();
        YAML::const_iterator col_it = row_it->begin();
        YAML::const_iterator col_it_end = row_it->end();
        while(col_it != col_it_end)
        {
            cols++;
            ++col_it;
        }

        M.resize(Eigen::NoChange, cols);

        // Load data
        row_it = node.begin();
        row_it_end = node.end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                col_it = row_it->begin();
                for (IndexType j = 0; j < cols; ++j) {
                    try {
                        M.coeffRef(i, j) = col_it->as<Scalar>();
                    } catch(const YAML::TypedBadConversion<Scalar>& ex) {
                        std::cerr << "[YAML - Matrix - decode] ERROR: Could not decode matrix entry (" << i << ", " << j << "): " << *col_it << " to scalar" << std::endl;
                        return false;
                    }
                    ++col_it;
                }
                ++row_it;
            }
        }

        return true;
    }

    /**
     * Specialization for Eigen matrices of dynamic sized columns and rows 
     */
    template <class Scalar, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, C, D, E>& M) 
    {
        typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, C, D, E>::Index IndexType;

        if(!isMatrix(node))
        {
            return false;
        }

        
        // count rows and cols
        IndexType rows = 0;
        IndexType cols = 0;

        // count rows
        YAML::const_iterator row_it = node.begin();
        YAML::const_iterator row_it_end = node.end();
        while(row_it != row_it_end)
        {
            rows++;
            ++row_it;
        }

        // count cols
        row_it = node.begin();
        YAML::const_iterator col_it = row_it->begin();
        YAML::const_iterator col_it_end = row_it->end();
        while(col_it != col_it_end)
        {
            cols++;
            ++col_it;
        }

        M.resize(rows, cols);

        // Load data
        row_it = node.begin();
        row_it_end = node.end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                col_it = row_it->begin();
                for (IndexType j = 0; j < cols; ++j) {
                    try {
                        M.coeffRef(i, j) = col_it->as<Scalar>();
                    } catch(const YAML::TypedBadConversion<Scalar>& ex) {
                        std::cerr << "[YAML - Matrix - decode] ERROR: Could not decode matrix entry (" << i << ", " << j << "): " << *col_it << " to scalar" << std::endl;
                    }
                    ++col_it;
                }
                ++row_it;
            }
        }

        return true;
    }
};

}  // namespace YAML

#endif // LVR2_IO_YAML_MATRIX_HPP

