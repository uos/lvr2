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

namespace hdf5features
{

template <typename Derived, typename T>
struct IOType;

template <typename Derived>
struct IOType<Derived, MeshBufferPtr> {
    using io_type = MeshIO<Derived>;
};

template <typename Derived>
struct IOType<Derived, PointBufferPtr> {
    using io_type = PointCloudIO<Derived>;
};

template <typename Derived>
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
    Derived* m_file_access                 = static_cast<Derived*>(this);
    ArrayIO<Derived>* m_array_io           = static_cast<ArrayIO<Derived>*>(m_file_access);

  private:
    const std::string m_chunkName       = "chunks";
    const std::string m_amountName      = "amount";
    const std::string m_chunkSizeName   = "size";
    const std::string m_boundingBoxName = "bounding_box";
};

} // namespace hdf5features

/**
 * Define you dependencies here:
 */
template <typename Derived>
struct Hdf5Construct<hdf5features::ChunkIO, Derived>
{
    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::ArrayIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::MeshIO, Derived>::type;
    using dep3 = typename Hdf5Construct<hdf5features::PointCloudIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;

    // add actual feature
    using type = typename deps::template add_features<hdf5features::ChunkIO>::type;
};

} // namespace lvr2

#include "ChunkIO.tcc"

#endif
