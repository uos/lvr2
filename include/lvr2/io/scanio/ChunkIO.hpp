#pragma once
#ifndef CHUNKIO
#define CHUNKIO

#include "lvr2/io/baseio/ArrayIO.hpp"
#include "lvr2/io/meshio/MeshIO.hpp"
#include "PointCloudIO.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/baseio/BaseIO.hpp"
#include "lvr2/io/Model.hpp"

using lvr2::baseio::FeatureConstruct;

namespace lvr2
{

template <typename BaseIO, typename T>
struct IOType;

template <typename BaseIO>
struct IOType<BaseIO, MeshBufferPtr> {
    using io_type = MeshIO<BaseIO>;
};

template <typename BaseIO>
struct IOType<BaseIO, PointBufferPtr> {
    using io_type = PointCloudIO<BaseIO>;
};

template <typename BaseIO>
class ChunkIO
{
  public:
    void saveAmount(BaseVector<std::size_t> amount);

    void saveChunkSize(float chunkSize);

    void saveBoundingBox(BoundingBox<BaseVector<float>> boundingBox);

    void save(BaseVector<std::size_t> amount,
              float chunkSize,
              BoundingBox<BaseVector<float>> boundingBox);

    template <typename T>
    void saveChunk(T data, std::string layer, int x, int y, int z);

    BaseVector<size_t> loadAmount();

    float loadChunkSize();

    BoundingBox<BaseVector<float>> loadBoundingBox();

    template <typename T>
    T loadChunk(std::string layer, int x, int y, int z);

  protected:
    BaseIO* m_file_access                 = static_cast<BaseIO*>(this);
    ArrayIO<BaseIO>* m_array_io           = static_cast<ArrayIO<BaseIO>*>(m_file_access);

  private:
    const std::string m_chunkName       = "chunks";
    const std::string m_amountName      = "amount";
    const std::string m_chunkSizeName   = "size";
    const std::string m_boundingBoxName = "bounding_box";
};

/**
 * Define you dependencies here:
 */
template <typename T>
struct FeatureConstruct<ChunkIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::ArrayIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::meshio::MeshIO, T>::type;
    using dep3 = typename FeatureConstruct<lvr2::scanio::PointCloudIO, T>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

    // add actual feature
    using type = typename deps::template add_features<ChunkIO>::type;
};

} // namespace lvr2

#include "ChunkIO.tcc"

#endif // CHUNKIO
