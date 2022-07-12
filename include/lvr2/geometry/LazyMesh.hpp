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

/**
 * The Group used to store LazyMeshes in the HDF5 file.
 * It is automatically created for the first LazyMesh, but has to be manually
 * removed by calling removeTempDir(file).
 */
constexpr const char* LAZY_MESH_TEMP_DIR = "/temp_lazy_meshes";

/**
 * @brief A Mesh that is only loaded into memory when needed.
 */
template<typename BaseVecT>
class LazyMesh
{
public:
    /**
     * @brief Construct a new Lazy Mesh object from the given mesh. The mesh is stored in the file
     * in the group specified by LAZY_MESH_TEMP_DIR.
     *
     * @param src The mesh to store.
     * @param file The file to store the mesh in whenever it is not needed.
     * @param keepLoaded Whether to keep the mesh loaded in memory.
     *
     * setting keepLoaded to true defeats the purpose of a LazyMesh, and should only be used if you
     * a) Need a consistent interface between several LazyMeshes and "Lazy"Meshes
     * b) Want to manually decide when to load and unload the mesh by using the keepLoaded()
     *    and allowUnload() methods
     *
     * file may be a nullptr if keepLoaded is true and stays true. Calling allowUnload() will
     * cause an exception to be thrown the next time the mesh is loaded.
     */
    LazyMesh(PMPMesh<BaseVecT>&& src, std::shared_ptr<HighFive::File> file, bool keepLoaded = false);

    /// move constructor
    LazyMesh(LazyMesh&&);
    /// move assignment operator
    LazyMesh& operator=(LazyMesh&&);

    virtual ~LazyMesh();

    /**
     * @brief Gain read-only access to the mesh, loading it if necessary.
     *
     * The instance returned by this is shared with any previous calls to get() or modify(),
     * provided that their shared_ptr are still alive. 
     *
     * @return A shared_ptr around the mesh. The mesh is unloaded again when the last
     *         shared_ptr instance is destroyed (unless keepLoaded is set).
     */
    std::shared_ptr<const PMPMesh<BaseVecT>> get();

    /**
     * @brief Same as get(), but the underlying file is updated when the mesh is unloaded.
     *
     * The instance returned by this is shared with those returned by get(). This allows calling
     * get() first and modify() only once you want to make changes to the mesh, without having to
     * load the mesh again.
     *
     * @return A shared_ptr around the mesh. The mesh is saved to file and unloaded as soon as the
     *         last reference is destroyed.
     */
    std::shared_ptr<PMPMesh<BaseVecT>> modify();

    /**
     * @brief Returns true if the mesh is currently loaded in memory.
     */
    bool isLoaded() const
    {
        return !m_weakPtr.expired();
    }

    /// Keeps the mesh loaded until the entire LazyMesh is destroyed or allowUnload() is called.
    void keepLoaded()
    {
        m_keepLoadedHelper = getInternal();
    }

    /// Returns true if the mesh is permanently loaded
    bool isKeptLoaded() const
    {
        return m_keepLoadedHelper != nullptr;
    }

    /**
     * @brief Inverse of keepLoaded(). Allows unloading the mesh if it is not needed anymore.
     *
     * Note: If you call this after creating an instance with a nullptr file, the mesh will be
     *       permanently deleted on unload and cannot be loaded again.
     */
    void allowUnload()
    {
        m_keepLoadedHelper.reset();
    }

    /// Get the number of faces without loading the mesh.
    size_t numFaces() const
    {
        return m_mesh->numFaces();
    }
    /// Get the number of vertices without loading the mesh.
    size_t numVertices() const
    {
        return m_mesh->numVertices();
    }
    /// Get the number of edges without loading the mesh.
    size_t numEdges() const
    {
        return m_mesh->numEdges();
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

    std::shared_ptr<PMPMesh<BaseVecT>> getInternal();
    void unload();

    /// The mesh. Either owned by us or the last remaining shared_ptr to it, depending on who is deleted first.
    PMPMesh<BaseVecT>* m_mesh;
    /// A weak pointer to the mesh to check if it is still loaded somewhere.
    std::weak_ptr<PMPMesh<BaseVecT>> m_weakPtr;
    /// An additional reference to the shared_ptr to ensure it stays loaded when keepLoaded is called/set.
    std::shared_ptr<PMPMesh<BaseVecT>> m_keepLoadedHelper;
    /// Indicates if the mesh was modified and needs to be saved.
    bool m_modified = false;
    /// The Group in which the mesh is stored. nullptr if the mesh is permanently loaded.
    std::unique_ptr<HighFive::Group> m_source;
    /// The name of m_source.
    std::string m_groupName;
    /// The file in which the mesh is stored. nullptr if the mesh is permanently loaded.
    std::shared_ptr<HighFive::File> m_file;

    /// Deleter for the shared_ptr
    struct Unloader
    {
        Unloader(LazyMesh* owner) : m_owner(owner) {}

        /// Called when the last shared_ptr to the mesh is destroyed.
        void operator()(PMPMesh<BaseVecT>* mesh);

        LazyMesh* m_owner;
    };
};

} // namespace lvr2

#include "lvr2/geometry/LazyMesh.tcc"
