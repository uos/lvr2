#ifndef LBVHINDEX_CUH
#define LBVHINDEX_CUH

#include <string>

namespace lbvh
{

struct AABB;        // Forward Declaration
struct BVHNode;     // Forward Declaration

class LBVHIndex
{
public:
    // CPU
    unsigned int m_num_objects;
    unsigned int m_num_nodes;
    unsigned int m_leaf_size;
    bool m_sort_queries;
    bool m_compact;

    // float* m_points;   
    // unsigned int* m_sorted_indices;

    char* m_mode;
    float m_radius;
    AABB* m_extent;
    BVHNode* m_nodes;
    unsigned int m_root_node;

    // TODO Do we need this here?
    float m_flip_x;
    float m_flip_y;
    float m_flip_z;

    // GPU
    float* m_d_points;
    unsigned int* m_d_sorted_indices;

    LBVHIndex();

    LBVHIndex(
        int leaf_size, bool sort_queries, bool compact,
        float flip_x=1000000.0f, float flip_y=1000000.0f, float flip_z=1000000.0f
    );

    void build(
        float* points, size_t num_points
    );
    // TODO Make these const
    void kSearch(
        float* query_points, size_t num_queries,
        int K, 
        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out
    );
    
    void kSearch_dev_ptr(
        float* query_points, size_t num_queries,
        int K, 
        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out
    );

    void radiusSearch(
        float* query_points, size_t num_queries,
        int K, float r,
        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out
    );
    
    void radiusSearch_dev_ptr(
        float* query_points, size_t num_queries,
        int K, float r,
        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out
    );

    void process_queries(
        float* queries_raw, size_t num_queries, 
        int K,
        unsigned int* n_neighbors_out, unsigned int* indices_out, float* distances_out
    ) const;

    void process_queries_dev_ptr(
        float* d_query_points, size_t num_queries,
        int K,
        unsigned int* d_n_neighbors_out, unsigned int* d_indices_out, float* d_distances_out
    ) const;

    void calculate_normals(
        float* normals, size_t num_normals,
        float* queries, size_t num_queries,
        int K,
        unsigned int* n_neighbors_out, unsigned int* indices_out
    );

    // TODO Neue Funktion, die im Kernel Nachbarn findet und gleichzeitig (im Anschluss) Normalen berechnet
    //      Rückgabe nur die Normalen
    void knn_normals(
        float* query_points, size_t num_queries,
        int K,
        float* normals, size_t num_normals
    );

    // TODO Neue Funktion, die indices_out, etc. als cuda Buffer (Pointer) "zurückgibt"

    AABB* getExtent(
        AABB* extent, float* points, size_t num_points
    );

    std::string getSampleDir() const;

    void getPtxFromCuString( 
        std::string& ptx, const char* sample_name, 
        const char* cu_source, const char* name, 
        const char** log_string 
    ) const;

};

}   // namespace lbvh

#endif // LBVHINDEX_CUH