/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * LazyMesh.hpp
 *
 * @date   13.05.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/util/Hdf5Util.hpp"

#include <memory>

namespace lvr2
{

/// The Group in which LazyMeshes are stored if the LazyMesh(mesh, file) constructor is called
constexpr const char* LAZY_MESH_TEMP_DIR = "/temp_meshes";

/**
 * @brief A Mesh that is only loaded into memory when needed.
 */
template<typename BaseVecT>
class LazyMesh
{
public:
    /**
     * @brief A thin wrapper around a PMPMesh. Note that the mesh is unloaded when this object is destroyed.
     *
     * This object can only be obtained inside of a shared_ptr through LazeMesh::get(). In order for
     * the loading and unloading of the LazyMesh to work properly, it is highly recommended to keep
     * this object in the shared_ptr as long as it is needed.
     */
    class MeshWrapper : public PMPMesh<BaseVecT>
    {
    public:
        virtual ~MeshWrapper()
        {
            if (m_parent && m_changed)
            {
                m_parent->update(*this);
            }
        }
    private:
        friend class LazyMesh;
        MeshWrapper(LazyMesh* parent) : m_parent(parent) {}
        MeshWrapper(PMPMesh<BaseVecT>&& src) : PMPMesh<BaseVecT>(std::move(src)) {}
        LazyMesh* m_parent = nullptr;
        bool m_changed = false;
    };

    /**
     * @brief Construct a new Lazy Mesh object from the given mesh. The mesh is stored in the file
     * in the LAZY_MESH_TEMP_DIR group.
     *
     * @param src The mesh to store.
     * @param file The file to store the mesh in whenever it is not needed.
     * 
     * Passing nullptr to file will create a LazyMesh that is permanently loaded. This completely
     * defeats the purpose of this class and should only be used if you need a consistent interface
     * with actual LazyMeshes.
     * Using the nullptr version will also move the mesh into this object, invalidating src.
     */
    LazyMesh(PMPMesh<BaseVecT>& src, std::shared_ptr<HighFive::File> file);

    // move constructor/assignment
    LazyMesh(LazyMesh&&) = default;
    LazyMesh& operator=(LazyMesh&&) = default;

    virtual ~LazyMesh();

    /**
     * @brief Gain read-only access to the mesh, loading it if necessary.
     *
     * @return MeshWrapper A wrapper around the mesh. The mesh is unloaded as soon as the last
     *         reference to the wrapper is destroyed.
     */
    std::shared_ptr<const MeshWrapper> get();

    /**
     * @brief Gain access to the mesh, loading it if necessary. The mesh in the underlying file is
     *        updated once the returned wrapper is destroyed.
     *
     * The instance returned by this is shared with those returned by get(). This allows calling
     * get() first and modify() only once you want to make changes to the mesh, without having to
     * load the mesh again.
     *
     * @return MeshWrapper A wrapper around the mesh. The mesh is unloaded as soon as the last
     *         reference to the wrapper is destroyed.
     */
    std::shared_ptr<MeshWrapper> modify();

    /**
     * @brief Loads the mesh (if not already loaded) and keeps it in memory until this instance is
     *        destroyed or allowUnload() is called.
     */
    void loadPermanently()
    {
        m_keepLoadedHelper = modify();
    }

    /**
     * @brief Inverse of loadPermanently(). Allows unloading the mesh if it is not needed anymore.
     * 
     * Note: If you call this after creating an instance with a nullptr file, the mesh will be
     *       permanently deleted on unload and cannot be loaded again.
     */
    void allowUnload()
    {
        m_keepLoadedHelper.reset();
    }

    /// Get the number of faces without loading the full mesh.
    size_t n_faces() const
    {
        auto inner = m_mesh.lock();
        return inner ? inner->numFaces() : hdf5util::getAttribute<uint64_t>(*m_source, "n_faces").value();
    }
    /// Get the number of vertices without loading the full mesh.
    size_t n_vertices() const
    {
        auto inner = m_mesh.lock();
        return inner ? inner->numVertices() : hdf5util::getAttribute<uint64_t>(*m_source, "n_vertices").value();
    }

    /// Returns the file this mesh is stored in (if any). Useful for creating a new LazyMesh in the same file.
    std::shared_ptr<HighFive::File> getFile() const
    {
        return m_file;
    }

    /**
     * @brief Remove the temp dir from the file. Make sure to only call this after any remaining
     *        LazyMesh instances in the file have been destroyed.
     * 
     * @param file The file to remove the temp dir from.
     */
    static void removeTempDir(std::shared_ptr<HighFive::File> file)
    {
        file->getGroup("/").unlink(LAZY_MESH_TEMP_DIR + 1); // +1 to skip the leading '/'
    }
private:
    // delete unwanted constructors/assignments
    LazyMesh() = delete;
    LazyMesh(const LazyMesh&) = delete;
    LazyMesh& operator=(const LazyMesh&) = delete;

    std::shared_ptr<MeshWrapper> getInternal();
    void update(const PMPMesh<BaseVecT>& mesh)
    {
        if (m_source)
        {
            mesh.write(*m_source);
            m_file->flush();
        }
    }

    static std::unique_ptr<HighFive::Group> nextGroup(const std::shared_ptr<HighFive::File>& file);

    /// A weak pointer to the mesh to check if it is still loaded somewhere.
    std::weak_ptr<MeshWrapper> m_mesh;
    /// Only used when for a permanently loaded mesh.
    std::shared_ptr<MeshWrapper> m_keepLoadedHelper;
    /// The Group in which the mesh is stored. nullptr if the mesh is permanently loaded.
    std::unique_ptr<HighFive::Group> m_source;
    /// The file in which the mesh is stored. nullptr if the mesh is permanently loaded.
    std::shared_ptr<HighFive::File> m_file;
};

} // namespace lvr2

#include "lvr2/geometry/LazyMesh.tcc"
