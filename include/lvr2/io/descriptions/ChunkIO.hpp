#pragma once
#ifndef LVR2_IO_HDF5_CHUNKIO_HPP
#define LVR2_IO_HDF5_CHUNKIO_HPP

#include "ArrayIO.hpp"
#include "MeshIO.hpp"
#include "PointCloudIO.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
#include "lvr2/io/Model.hpp"

namespace lvr2
{

template <typename FeatureBase, typename T>
struct IOType;

template <typename FeatureBase>
struct IOType<FeatureBase, MeshBufferPtr> {
    using io_type = MeshIO<FeatureBase>;
};

template <typename FeatureBase>
struct IOType<FeatureBase, PointBufferPtr> {
    using io_type = PointCloudIO<FeatureBase>;
};

template <typename FeatureBase>
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
    FeatureBase* m_file_access                 = static_cast<FeatureBase*>(this);
    ArrayIO<FeatureBase>* m_array_io           = static_cast<ArrayIO<FeatureBase>*>(m_file_access);

  private:
    const std::string m_chunkName       = "chunks";
    const std::string m_amountName      = "amount";
    const std::string m_chunkSizeName   = "size";
    const std::string m_boundingBoxName = "bounding_box";
};

/**
 * Define you dependencies here:
 */
template <typename FeatureBase>
struct FeatureConstruct<ChunkIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<ArrayIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MeshIO, FeatureBase>::type;
    using dep3 = typename FeatureConstruct<PointCloudIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

    // add actual feature
    using type = typename deps::template add_features<ChunkIO>::type;
};

} // namespace lvr2

#include "ChunkIO.tcc"

#endif
