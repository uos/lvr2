
#ifndef LVR2_IO_YAML_MATRIX_IO_HPP
#define LVR2_IO_YAML_MATRIX_IO_HPP

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

namespace YAML
{


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

        Node node;

        node["rows"] = rows;
        node["cols"] = cols;
        node["data"] = Load("[]");

        for (IndexType i = 0; i < rows; ++i) {
            for (IndexType j = 0; j < cols; ++j) {
                node["data"].push_back(M.coeff(i, j));
            }
        }
        return node;
    }

    template <class Scalar, int A, int B, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, A, B, C, D, E>& M) 
    {
        typedef typename Eigen::Matrix<Scalar, A, B, C, D, E>::Index IndexType;
        
        IndexType rows = node["rows"].as<IndexType>();
        IndexType cols = node["cols"].as<IndexType>();

        size_t expected_size = M.rows() * M.cols();
        if (!node["data"].IsSequence() || node["data"].size() != expected_size) {
            return false;
        }

        YAML::const_iterator it = node["data"].begin();
        YAML::const_iterator it_end = node["data"].end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                for (IndexType j = 0; j < cols; ++j) {
                    M.coeffRef(i, j) = it->as<Scalar>();
                    ++it;
                }
            }
        }
        return true;
    }

    template <class Scalar, int B, int C, int D, int E>
    static bool decode(const Node& node, Eigen::Matrix<Scalar, Eigen::Dynamic, B, C, D, E>& M) 
    {

        typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, B, C, D, E>::Index IndexType;
        IndexType rows = node["rows"].as<IndexType>();
        IndexType cols = node["cols"].as<IndexType>();

        M.resize(rows, Eigen::NoChange);

        size_t expected_size = M.rows() * M.cols();
        if (!node["data"].IsSequence() || node["data"].size() != expected_size) {
        return false;
        }

        YAML::const_iterator it = node["data"].begin();
        YAML::const_iterator it_end = node["data"].end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                for (IndexType j = 0; j < cols; ++j) {
                    M.coeffRef(i, j) = it->as<Scalar>();
                    ++it;
                }
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
        IndexType rows = node["rows"].as<IndexType>();
        IndexType cols = node["cols"].as<IndexType>();

        M.resize(Eigen::NoChange, cols);

        size_t expected_size = M.rows() * M.cols();
        if (!node["data"].IsSequence() || node["data"].size() != expected_size) {
            return false;
        }

        YAML::const_iterator it = node["data"].begin();
        YAML::const_iterator it_end = node["data"].end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                for (IndexType j = 0; j < cols; ++j) {
                    M.coeffRef(i, j) = it->as<Scalar>();
                    ++it;
                }
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
        IndexType rows = node["rows"].as<IndexType>();
        IndexType cols = node["cols"].as<IndexType>();

        M.resize(rows, cols);

        size_t expected_size = M.rows() * M.cols();
        if (!node["data"].IsSequence() || node["data"].size() != expected_size) {
            return false;
        }

        YAML::const_iterator it = node["data"].begin();
        YAML::const_iterator it_end = node["data"].end();
        if (rows > 0 && cols > 0) {
            for (IndexType i = 0; i < rows; ++i) {
                for (IndexType j = 0; j < cols; ++j) {
                    M.coeffRef(i, j) = it->as<Scalar>();
                    ++it;
                }
            }
        }
        return true;
    }
};
}  // namespace YAML

#endif // LVR2_IO_YAML_MATRIX_IO_HPP

