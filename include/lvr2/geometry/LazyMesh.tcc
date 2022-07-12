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
 * LazyMesh.tcc
 *
 * @date   13.05.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

namespace lvr2
{

template<typename BaseVecT>
LazyMesh<BaseVecT>::LazyMesh(PMPMesh<BaseVecT>&& src, std::shared_ptr<HighFive::File> file, bool keepLoaded)
    : m_mesh(new PMPMesh<BaseVecT>(std::move(src))), m_file(file)
{
    if (m_file)
    {
        auto parentGroup = hdf5util::getGroup(file, LAZY_MESH_TEMP_DIR);
        uint64_t id = 0;
        if (parentGroup.hasAttribute("next_id"))
        {
            auto attr = parentGroup.getAttribute("next_id");
            attr.read(id);
            attr.write(id + 1);
        }
        else
        {
            parentGroup.createAttribute("next_id", id + 1);
        }
        m_groupName = std::to_string(id);
        m_source = std::make_unique<HighFive::Group>(parentGroup.createGroup(m_groupName));
    }
    else
    {
        keepLoaded = true;
    }

    // ensure the first call to unload() actually writes
    m_modified = true;

    if (keepLoaded)
    {
        m_keepLoadedHelper.reset(m_mesh, Unloader(this));
        m_weakPtr = m_keepLoadedHelper;
    }
    else
    {
        unload();
    }
}

template<typename BaseVecT>
LazyMesh<BaseVecT>::LazyMesh(LazyMesh<BaseVecT>&& src)
{
    operator=(std::move(src));
}

template<typename BaseVecT>
LazyMesh<BaseVecT>& LazyMesh<BaseVecT>::operator=(LazyMesh<BaseVecT>&& src)
{
    // this operator cannot use the default implementation because:
    // a) src.m_mesh has to be set to nullptr to avoid premature deletion of the mesh
    // b) the m_owner of an existing Unloader has to be updated

    if (this != &src)
    {
        m_mesh = src.m_mesh;
        m_weakPtr = std::move(src.m_weakPtr);
        m_keepLoadedHelper = std::move(src.m_keepLoadedHelper);
        m_modified = src.m_modified;
        m_source = std::move(src.m_source);
        m_groupName = std::move(src.m_groupName);
        m_file = std::move(src.m_file);

        auto existing = m_weakPtr.lock();
        if (existing)
        {
            std::get_deleter<Unloader>(existing)->m_owner = this;
        }

        src.m_mesh = nullptr;
        src.m_weakPtr.reset();
        src.m_keepLoadedHelper.reset();
        src.m_modified = false;
        src.m_source.reset();
        src.m_groupName = "";
        src.m_file.reset();
    }
    return *this;
}

template<typename BaseVecT>
LazyMesh<BaseVecT>::~LazyMesh()
{
    auto inner = m_weakPtr.lock();
    if (inner)
    {
        // the shared_ptr can live longer than this instance, but the Unloader shouldn't try to unload the mesh
        std::get_deleter<Unloader>(inner)->m_owner = nullptr;
    }
    else if (m_mesh)
    {
        // no one else has access to the mesh, delete it
        delete m_mesh;
    }

    if (m_file)
    {
        auto parentGroup = m_file->getGroup(LAZY_MESH_TEMP_DIR);
        parentGroup.unlink(m_groupName);
    }
}

template<typename BaseVecT>
std::shared_ptr<const PMPMesh<BaseVecT>> LazyMesh<BaseVecT>::get()
{
    return getInternal();
}
template<typename BaseVecT>
std::shared_ptr<PMPMesh<BaseVecT>> LazyMesh<BaseVecT>::modify()
{
    m_modified = true;
    return getInternal();
}
template<typename BaseVecT>
std::shared_ptr<PMPMesh<BaseVecT>> LazyMesh<BaseVecT>::getInternal()
{
    auto ret = m_weakPtr.lock();

    if (!ret)
    {
        if (!m_source)
        {
            throw std::runtime_error("LazyMesh: unloaded and loaded a mesh without specifying a file");
        }
        m_mesh->getSurfaceMesh().restore(*m_source);
        ret.reset(m_mesh, Unloader(this));
        m_weakPtr = ret;
    }

    return ret;
}

template<typename BaseVecT>
void LazyMesh<BaseVecT>::unload()
{
    if (m_modified && m_file)
    {
        m_mesh->getSurfaceMesh().unload(*m_source);
        m_file->flush();
        m_modified = false;
    }
    else
    {
        // file up-to-date or no file => no need to unload
        m_mesh->getSurfaceMesh().shallow_clear();
    }
}

template<typename BaseVecT>
void LazyMesh<BaseVecT>::Unloader::operator()(PMPMesh<BaseVecT>* mesh)
{
    if (m_owner)
    {
        m_owner->unload();
    }
    else
    {
        // owner was deleted first, last shared_ptr was destroyed => delete the mesh
        delete mesh;
    }
}

} // namespace lvr2
