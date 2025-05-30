// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see PMP_LICENSE.txt for details.

#include "lvr2/geometry/pmp/SurfaceMesh.h"

#include <cmath>
#include <thread>

#include <omp.h>

#include "lvr2/geometry/pmp/SurfaceMeshIO.h"
#include "lvr2/util/Hdf5Util.hpp"

namespace pmp {

SurfaceMesh::SurfaceMesh()
{
    initialize();
}

void SurfaceMesh::initialize()
{
    oprops_.resize(1);
    vpoint_ = vertex_property<Point>("v:point");
    vconn_ = vertex_property<VertexConnectivity>("v:connectivity");
    hconn_ = halfedge_property<HalfedgeConnectivity>("h:connectivity");
    fconn_ = face_property<FaceConnectivity>("f:connectivity");

    vdeleted_ = vertex_property<bool>("v:deleted", false);
    edeleted_ = edge_property<bool>("e:deleted", false);
    fdeleted_ = face_property<bool>("f:deleted", false);

    deleted_vertices_ = 0;
    deleted_edges_ = 0;
    deleted_faces_ = 0;
    has_garbage_ = false;
}

SurfaceMesh& SurfaceMesh::operator=(const SurfaceMesh& rhs)
{
    if (this != &rhs)
    {
        // deep copy of property containers
        oprops_ = rhs.oprops_;
        vprops_ = rhs.vprops_;
        hprops_ = rhs.hprops_;
        eprops_ = rhs.eprops_;
        fprops_ = rhs.fprops_;

        // property handles contain pointers, have to be reassigned
        initialize();

        // how many elements are deleted?
        deleted_vertices_ = rhs.deleted_vertices_;
        deleted_edges_ = rhs.deleted_edges_;
        deleted_faces_ = rhs.deleted_faces_;

        has_garbage_ = rhs.has_garbage_;
    }

    return *this;
}

SurfaceMesh& SurfaceMesh::assign(const SurfaceMesh& rhs)
{
    if (this != &rhs)
    {
        clear();

        // copy properties from other mesh
        vpoint_.array() = rhs.vpoint_.array();
        vconn_.array() = rhs.vconn_.array();
        hconn_.array() = rhs.hconn_.array();
        fconn_.array() = rhs.fconn_.array();

        vdeleted_.array() = rhs.vdeleted_.array();
        edeleted_.array() = rhs.edeleted_.array();
        fdeleted_.array() = rhs.fdeleted_.array();

        // resize (needed by property containers)
        vprops_.resize(rhs.vertices_size());
        hprops_.resize(rhs.halfedges_size());
        eprops_.resize(rhs.edges_size());
        fprops_.resize(rhs.faces_size());

        // how many elements are deleted?
        deleted_vertices_ = rhs.deleted_vertices_;
        deleted_edges_ = rhs.deleted_edges_;
        deleted_faces_ = rhs.deleted_faces_;
        has_garbage_ = rhs.has_garbage_;
    }

    return *this;
}

void SurfaceMesh::read(const std::string& filename, const IOFlags& flags)
{
    SurfaceMeshIO reader(filename, flags);
    reader.read(*this);
}

void SurfaceMesh::write_const(const std::string& filename, const IOFlags& flags) const
{
    SurfaceMeshIO writer(filename, flags);
    writer.write_const(*this);
}

void SurfaceMesh::read(const HighFive::Group& group)
{
    SurfaceMeshIO::read_hdf5(group, *this);
}
void SurfaceMesh::write_const(HighFive::Group& group) const
{
    SurfaceMeshIO::write_hdf5_const(group, *this);
}

void SurfaceMesh::unload(HighFive::Group& group)
{
    auto oprop_group = lvr2::hdf5util::getGroup(group, "oprops");
    oprops_.unload(oprop_group);
    auto vprop_group = lvr2::hdf5util::getGroup(group, "vprops");
    vprops_.unload(vprop_group);
    auto hprop_group = lvr2::hdf5util::getGroup(group, "hprops");
    hprops_.unload(hprop_group);
    auto eprop_group = lvr2::hdf5util::getGroup(group, "eprops");
    eprops_.unload(eprop_group);
    auto fprop_group = lvr2::hdf5util::getGroup(group, "fprops");
    fprops_.unload(fprop_group);
}
void SurfaceMesh::restore(const HighFive::Group& group)
{
    oprops_.restore(group.getGroup("oprops"));
    vprops_.restore(group.getGroup("vprops"));
    hprops_.restore(group.getGroup("hprops"));
    eprops_.restore(group.getGroup("eprops"));
    fprops_.restore(group.getGroup("fprops"));
}

void SurfaceMesh::clear()
{
    // remove all properties
    oprops_.clear();
    vprops_.clear();
    hprops_.clear();
    eprops_.clear();
    fprops_.clear();

    initialize();
}

void SurfaceMesh::shallow_clear()
{
    // remove all properties
    oprops_.shallow_clear();
    vprops_.shallow_clear();
    hprops_.shallow_clear();
    eprops_.shallow_clear();
    fprops_.shallow_clear();
}

void SurfaceMesh::free_memory()
{
    vprops_.free_memory();
    oprops_.free_memory();
    hprops_.free_memory();
    eprops_.free_memory();
    fprops_.free_memory();
}

void SurfaceMesh::reserve(size_t nvertices, size_t nedges, size_t nfaces)
{
    // Always give 10% extra space to be on the safe side
    nvertices *= 1.1;
    nedges *= 1.1;
    nfaces *= 1.1;

    // Euler's formula: V - E + F ~= 0
    if (nvertices == 0 && nedges != 0 && nfaces != 0)
    {
        // V = E - F
        nvertices = nedges - nfaces;
    }
    else if (nvertices != 0 && nedges == 0 && nfaces != 0)
    {
        // E = V + F
        nedges = nvertices + nfaces;
    }
    else if (nvertices != 0 && nedges != 0 && nfaces == 0)
    {
        // F = E - V
        nfaces = nedges - nvertices;
    }

    oprops_.reserve(1);
    vprops_.reserve(nvertices);
    hprops_.reserve(2 * nedges);
    eprops_.reserve(nedges);
    fprops_.reserve(nfaces);
}

void SurfaceMesh::property_stats() const
{
    std::vector<std::string> props;

    std::cout << "point properties:\n";
    props = vertex_properties();
    for (const auto& prop : props)
        std::cout << "\t" << prop << std::endl;

    std::cout << "halfedge properties:\n";
    props = halfedge_properties();
    for (const auto& prop : props)
        std::cout << "\t" << prop << std::endl;

    std::cout << "edge properties:\n";
    props = edge_properties();
    for (const auto& prop : props)
        std::cout << "\t" << prop << std::endl;

    std::cout << "face properties:\n";
    props = face_properties();
    for (const auto& prop : props)
        std::cout << "\t" << prop << std::endl;
}

Halfedge SurfaceMesh::find_halfedge(Vertex start, Vertex end) const
{
    assert(is_valid(start) && is_valid(end));

    for (Halfedge h : halfedges(start))
        if (to_vertex(h) == end)
            return h;

    return Halfedge();
}

Edge SurfaceMesh::find_edge(Vertex a, Vertex b) const
{
    Halfedge h = find_halfedge(a, b);
    return h.is_valid() ? h.edge() : Edge();
}

void SurfaceMesh::adjust_outgoing_halfedge(Vertex v)
{
    for (Halfedge h : halfedges(v))
    {
        if (is_boundary(h))
        {
            set_halfedge(v, h);
            return;
        }
    }
}

Vertex SurfaceMesh::add_vertex(const Point& p)
{
    Vertex v = new_vertex();
    if (v.is_valid())
        vpoint_[v] = p;
    return v;
}

Face SurfaceMesh::add_triangle(Vertex v0, Vertex v1, Vertex v2)
{
    temp_face_vertices_.resize(3);
    temp_face_vertices_[0] = v0;
    temp_face_vertices_[1] = v1;
    temp_face_vertices_[2] = v2;
    return add_face(temp_face_vertices_);
}

Face SurfaceMesh::add_quad(Vertex v0, Vertex v1, Vertex v2, Vertex v3)
{
    temp_face_vertices_.resize(4);
    temp_face_vertices_[0] = v0;
    temp_face_vertices_[1] = v1;
    temp_face_vertices_[2] = v2;
    temp_face_vertices_[3] = v3;
    return add_face(temp_face_vertices_);
}

Face SurfaceMesh::add_face(const std::vector<Vertex>& vertices)
{
    const size_t n(vertices.size());
    assert(n > 2);

    Vertex v;
    size_t i, ii, id;
    Halfedge inner_next, inner_prev, outer_next, outer_prev, boundary_next,
        boundary_prev, patch_start, patch_end;

    // use global arrays to avoid new/delete of local arrays!!!
    std::vector<Halfedge>& halfedges = add_face_halfedges_;
    std::vector<bool>& is_new = add_face_is_new_;
    std::vector<bool>& needs_adjust = add_face_needs_adjust_;
    NextCache& next_cache = add_face_next_cache_;
    halfedges.clear();
    halfedges.resize(n);
    is_new.clear();
    is_new.resize(n);
    needs_adjust.clear();
    needs_adjust.resize(n, false);
    next_cache.clear();
    next_cache.reserve(3 * n);

    // test for topological errors
    for (i = 0; i < n; ++i)
    {
        if (!is_valid(vertices[i]))
        {
            auto what = "SurfaceMesh::add_face: Invalid vertex.";
            throw TopologyException(what);
        }
        for (ii = i + 1; ii < n; ++ii)
        {
            if (vertices[i] == vertices[ii])
            {
                auto what = "SurfaceMesh::add_face: Duplicate vertices in face.";
                throw TopologyException(what);
            }
        }
    }
    for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n)
    {
        if (!is_boundary(vertices[i]))
        {
            auto what = "SurfaceMesh::add_face: Complex vertex.";
            throw TopologyException(what);
        }

        halfedges[i] = find_halfedge(vertices[i], vertices[ii]);
        is_new[i] = !halfedges[i].is_valid();

        if (!is_new[i] && !is_boundary(halfedges[i]))
        {
            auto what = "SurfaceMesh::add_face: Complex edge.";
            throw TopologyException(what);
        }
    }

    // re-link patches if necessary
    for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n)
    {
        if (!is_new[i] && !is_new[ii])
        {
            inner_prev = halfedges[i];
            inner_next = halfedges[ii];

            if (next_halfedge(inner_prev) != inner_next)
            {
                // here comes the ugly part... we have to relink a whole patch

                // search a free gap
                // free gap will be between boundaryPrev and boundaryNext
                outer_prev = inner_next.opposite();
                outer_next = inner_prev.opposite();
                boundary_prev = outer_prev;
                CirculatorLoopDetector loop_detector(boundary_prev);
                do
                {
                    boundary_prev = next_halfedge(boundary_prev).opposite();
                    loop_detector.loop_detection(boundary_prev);
                } while (!is_boundary(boundary_prev) || boundary_prev == inner_prev);
                boundary_next = next_halfedge(boundary_prev);
                assert(is_boundary(boundary_prev));
                assert(is_boundary(boundary_next));

                // ok ?
                if (boundary_next == inner_next)
                {
                    auto what =
                        "SurfaceMesh::add_face: Patch re-linking failed.";
                    throw TopologyException(what);
                }

                // other halfedges' handles
                patch_start = next_halfedge(inner_prev);
                patch_end = prev_halfedge(inner_next);

                // relink
                next_cache.emplace_back(boundary_prev, patch_start);
                next_cache.emplace_back(patch_end, boundary_next);
                next_cache.emplace_back(inner_prev, inner_next);
            }
        }
    }

    // create missing edges
    for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n)
    {
        if (is_new[i])
        {
            halfedges[i] = new_edge(vertices[i], vertices[ii]);
        }
    }

    // create the face
    Face f(new_face());
    set_halfedge(f, halfedges[n - 1]);

    // setup halfedges
    for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n)
    {
        v = vertices[ii];
        inner_prev = halfedges[i];
        inner_next = halfedges[ii];

        id = 0;
        if (is_new[i])
            id |= 1;
        if (is_new[ii])
            id |= 2;

        if (id)
        {
            outer_prev = inner_next.opposite();
            outer_next = inner_prev.opposite();

            // set outer links
            switch (id)
            {
                case 1: // prev is new, next is old
                    boundary_prev = prev_halfedge(inner_next);
                    next_cache.emplace_back(boundary_prev, outer_next);
                    set_halfedge(v, outer_next);
                    break;

                case 2: // next is new, prev is old
                    boundary_next = next_halfedge(inner_prev);
                    next_cache.emplace_back(outer_prev, boundary_next);
                    set_halfedge(v, boundary_next);
                    break;

                case 3: // both are new
                    if (!halfedge(v).is_valid())
                    {
                        set_halfedge(v, outer_next);
                        next_cache.emplace_back(outer_prev, outer_next);
                    }
                    else
                    {
                        boundary_next = halfedge(v);
                        boundary_prev = prev_halfedge(boundary_next);
                        next_cache.emplace_back(boundary_prev, outer_next);
                        next_cache.emplace_back(outer_prev, boundary_next);
                    }
                    break;
            }

            // set inner link
            next_cache.emplace_back(inner_prev, inner_next);
        }
        else
            needs_adjust[ii] = (halfedge(v) == inner_next);

        // set face handle
        set_face(halfedges[i], f);
    }

    // process next halfedge cache
    NextCache::const_iterator ncIt(next_cache.begin()), ncEnd(next_cache.end());
    for (; ncIt != ncEnd; ++ncIt)
    {
        set_next_halfedge(ncIt->first, ncIt->second);
    }

    // adjust vertices' halfedge handle
    for (i = 0; i < n; ++i)
    {
        if (needs_adjust[i])
        {
            adjust_outgoing_halfedge(vertices[i]);
        }
    }

    return f;
}

size_t SurfaceMesh::valence(Vertex v) const
{
    size_t count(0);

    for (auto vv : vertices(v))
    {
        PMP_ASSERT(vv.is_valid());
        ++count;
    }

    return count;
}

size_t SurfaceMesh::valence(Face f) const
{
    size_t count(0);

    for (auto v : vertices(f))
    {
        PMP_ASSERT(v.is_valid());
        ++count;
    }

    return count;
}

Point SurfaceMesh::center(Face f) const
{
    size_t count(0);
    Point sum(Point::Zero());

    for (auto v : vertices(f))
    {
        PMP_ASSERT(v.is_valid());
        sum += position(v);
        ++count;
    }

    return sum / count;
}

BoundingBox SurfaceMesh::bounds() const
{
    BoundingBox bb;
    if (has_garbage_)
    {
        #pragma omp parallel for schedule(static) reduction(+:bb)
        for (size_t i = 0; i < vertices_size(); i++)
        {
            Vertex v(i);
            if (!is_deleted(v))
                bb += position(v);
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) reduction(+:bb)
        for (size_t i = 0; i < vertices_size(); i++)
        {
            bb += position(Vertex(i));
        }
    }
    return bb;
}

bool SurfaceMesh::is_triangle_mesh() const
{
    bool is_triangle_mesh = true;
    if (has_garbage_)
    {
        #pragma omp parallel for schedule(static) reduction(&&:is_triangle_mesh)
        for (size_t i = 0; i < faces_size(); i++)
        {
            Face f(i);
            if (!is_deleted(f))
                is_triangle_mesh = is_triangle_mesh && (valence(f) == 3);
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) reduction(&&:is_triangle_mesh)
        for (size_t i = 0; i < faces_size(); i++)
        {
            is_triangle_mesh = is_triangle_mesh && (valence(Face(i)) == 3);
        }
    }

    return is_triangle_mesh;
}

bool SurfaceMesh::is_quad_mesh() const
{
    bool is_quad_mesh = true;
    if (has_garbage_)
    {
        #pragma omp parallel for schedule(static) reduction(&&:is_quad_mesh)
        for (size_t i = 0; i < faces_size(); i++)
        {
            Face f(i);
            if (!is_deleted(f))
            {
                is_quad_mesh = is_quad_mesh && (valence(f) == 4);
            }
        }
    }
    else
    {
        #pragma omp parallel for schedule(static) reduction(&&:is_quad_mesh)
        for (size_t i = 0; i < faces_size(); i++)
        {
            is_quad_mesh = is_quad_mesh && (valence(Face(i)) == 4);
        }
    }

    return is_quad_mesh;
}

void SurfaceMesh::duplicate_non_manifold_vertices(bool print)
{
    auto reachable = add_halfedge_property<uint8_t>("h:reachable", false);
    // reachable should be HalfedgeProperty<bool>, but std::vector<bool> is specialized as a
    // bit-vector, which would break when using OpenMP.

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vertices_size(); i++)
    {
        Vertex v(i);
        if (!is_deleted(v))
            for (Halfedge h : halfedges(v))
                reachable[h] = true;
    }
    size_t count = 0;
    auto vprop_map = gen_vertex_copy_map(*this);
    #pragma omp parallel reduction(+:count)
    {
        std::vector<Halfedge> unreachable;
        #pragma omp for schedule(dynamic,64) nowait
        for (size_t i = 0; i < halfedges_size(); i++)
        {
            Halfedge h(i);
            if (!is_deleted(h) && !reachable[h])
                unreachable.push_back(h);
        }
        #pragma omp critical
        {
            for (Halfedge h : unreachable)
            {
                if (reachable[h])
                {
                    continue;
                }
                Vertex v = from_vertex(h);
                Point p = position(v);
                Vertex new_v = add_vertex(p);
                vprop_map.copy(v, new_v);
                set_halfedge(new_v, h);
                for (Halfedge h : halfedges(new_v))
                {
                    set_vertex(h.opposite(), new_v);
                    reachable[h] = true;
                }
                adjust_outgoing_halfedge(new_v);
                ++count;
            }
        }
    }
    if (print && count > 0)
    {
        std::cout << "Duplicated " << count << " non-manifold vertices" << std::endl;
    }

    remove_halfedge_property(reachable);
}

void SurfaceMesh::remove_degenerate_faces(bool print)
{
    size_t nf = n_faces(), ne = n_edges();

    #pragma omp parallel
    {
        std::vector<Halfedge> broken;
        #pragma omp for schedule(dynamic,64)
        for (size_t i = 0; i < halfedges_size(); i++)
        {
            Halfedge h(i);
            if (!is_deleted(h) && next_halfedge(next_halfedge(h)) == h)
                if (h.idx() < next_halfedge(h).idx()) // loops will always be detected by both halfedges => only keep one
                    broken.push_back(h);
        }
        #pragma omp critical
        {
            for (Halfedge h : broken)
                if (!is_deleted(h) && next_halfedge(next_halfedge(h)) == h)
                    remove_loop_helper(h);
        }
    }

    if (print)
    {
        // remove_loop_helper removes one edge and possibly one face
        size_t f_diff = nf - n_faces();
        if (f_diff > 0)
            std::cout << "Removed " << f_diff << " degenerate faces" << std::endl;
        size_t e_diff = ne - n_edges() - f_diff;
        if (e_diff > 0)
            std::cout << "Removed " << e_diff << " degenerate edges without a face" << std::endl;
    }
}

void SurfaceMesh::split(Face f, Vertex v)
{
    // Split an arbitrary face into triangles by connecting each vertex of face
    // f to vertex v . Face f will remain valid (it will become one of the
    // triangles). The halfedge handles of the new triangles will point to the
    // old halfedges.

    Halfedge hend = halfedge(f);
    Halfedge h = next_halfedge(hend);

    Halfedge hold = new_edge(to_vertex(hend), v);

    set_next_halfedge(hend, hold);
    set_face(hold, f);

    hold = hold.opposite();

    while (h != hend)
    {
        Halfedge hnext = next_halfedge(h);

        Face fnew = new_face();
        set_halfedge(fnew, h);

        Halfedge hnew = new_edge(to_vertex(h), v);

        set_next_halfedge(hnew, hold);
        set_next_halfedge(hold, h);
        set_next_halfedge(h, hnew);

        set_face(hnew, fnew);
        set_face(hold, fnew);
        set_face(h, fnew);

        hold = hnew.opposite();

        h = hnext;
    }

    set_next_halfedge(hold, hend);
    set_prev_halfedge(hold, next_halfedge(hend));

    set_face(hold, f);

    set_halfedge(v, hold);
}

Halfedge SurfaceMesh::split(Edge e, Vertex v)
{
    Halfedge h0 = e.halfedge(0);
    Halfedge o0 = e.halfedge(1);

    Vertex v2 = to_vertex(o0);

    Halfedge e1 = new_edge(v, v2);
    Halfedge t1 = e1.opposite();

    Face f0 = face(h0);
    Face f3 = face(o0);

    set_halfedge(v, h0);
    set_vertex(o0, v);

    if (!is_boundary(h0))
    {
        Halfedge h1 = next_halfedge(h0);
        Halfedge h2 = next_halfedge(h1);

        Vertex v1 = to_vertex(h1);

        Halfedge e0 = new_edge(v, v1);
        Halfedge t0 = e0.opposite();

        Face f1 = new_face();
        set_halfedge(f0, h0);
        set_halfedge(f1, h2);

        set_face(h1, f0);
        set_face(t0, f0);
        set_face(h0, f0);

        set_face(h2, f1);
        set_face(t1, f1);
        set_face(e0, f1);

        set_next_halfedge(h0, h1);
        set_next_halfedge(h1, t0);
        set_next_halfedge(t0, h0);

        set_next_halfedge(e0, h2);
        set_next_halfedge(h2, t1);
        set_next_halfedge(t1, e0);
    }
    else
    {
        set_prev_halfedge(t1, prev_halfedge(h0));
        set_next_halfedge(t1, h0);
        // halfedge handle of vh already is h0
    }

    if (!is_boundary(o0))
    {
        Halfedge o1 = next_halfedge(o0);
        Halfedge o2 = next_halfedge(o1);

        Vertex v3 = to_vertex(o1);

        Halfedge e2 = new_edge(v, v3);
        Halfedge t2 = e2.opposite();

        Face f2 = new_face();
        set_halfedge(f2, o1);
        set_halfedge(f3, o0);

        set_face(o1, f2);
        set_face(t2, f2);
        set_face(e1, f2);

        set_face(o2, f3);
        set_face(o0, f3);
        set_face(e2, f3);

        set_next_halfedge(e1, o1);
        set_next_halfedge(o1, t2);
        set_next_halfedge(t2, e1);

        set_next_halfedge(o0, e2);
        set_next_halfedge(e2, o2);
        set_next_halfedge(o2, o0);
    }
    else
    {
        set_next_halfedge(e1, next_halfedge(o0));
        set_next_halfedge(o0, e1);
        set_halfedge(v, e1);
    }

    if (halfedge(v2) == h0)
        set_halfedge(v2, t1);

    return t1;
}

Halfedge SurfaceMesh::insert_vertex(Halfedge h0, Vertex v)
{
    // before:
    //
    // v0      h0       v2
    //  o--------------->o
    //   <---------------
    //         o0
    //
    // after:
    //
    // v0  h0   v   h1   v2
    //  o------>o------->o
    //   <------ <-------
    //     o0       o1

    Halfedge h2 = next_halfedge(h0);
    Halfedge o0 = h0.opposite();
    Halfedge o2 = prev_halfedge(o0);
    Vertex v2 = to_vertex(h0);
    Face fh = face(h0);
    Face fo = face(o0);

    Halfedge h1 = new_edge(v, v2);
    Halfedge o1 = h1.opposite();

    // adjust halfedge connectivity
    set_next_halfedge(h1, h2);
    set_next_halfedge(h0, h1);
    set_vertex(h0, v);
    set_vertex(h1, v2);
    set_face(h1, fh);

    set_next_halfedge(o1, o0);
    set_next_halfedge(o2, o1);
    set_vertex(o1, v);
    set_face(o1, fo);

    // adjust vertex connectivity
    set_halfedge(v2, o1);
    adjust_outgoing_halfedge(v2);
    set_halfedge(v, h1);
    adjust_outgoing_halfedge(v);

    // adjust face connectivity
    if (fh.is_valid())
        set_halfedge(fh, h0);
    if (fo.is_valid())
        set_halfedge(fo, o1);

    return o1;
}

Halfedge SurfaceMesh::insert_edge(Halfedge h0, Halfedge h1)
{
    assert(face(h0) == face(h1));
    assert(face(h0).is_valid());

    Vertex v0 = to_vertex(h0);
    Vertex v1 = to_vertex(h1);

    Halfedge h2 = next_halfedge(h0);
    Halfedge h3 = next_halfedge(h1);

    Halfedge h4 = new_edge(v0, v1);
    Halfedge h5 = h4.opposite();

    Face f0 = face(h0);
    Face f1 = new_face();

    set_halfedge(f0, h0);
    set_halfedge(f1, h1);

    set_next_halfedge(h0, h4);
    set_next_halfedge(h4, h3);
    set_face(h4, f0);

    set_next_halfedge(h1, h5);
    set_next_halfedge(h5, h2);
    Halfedge h = h2;
    do
    {
        set_face(h, f1);
        h = next_halfedge(h);
    } while (h != h2);

    return h4;
}

bool SurfaceMesh::is_flip_ok(Edge e) const
{
    // boundary edges cannot be flipped
    if (is_boundary(e))
        return false;

    // check if the flipped edge is already present in the mesh
    Halfedge h0 = e.halfedge(0);
    Halfedge h1 = e.halfedge(1);

    Vertex v0 = to_vertex(next_halfedge(h0));
    Vertex v1 = to_vertex(next_halfedge(h1));

    if (v0 == v1) // this is generally a bad sign !!!
        return false;

    if (find_halfedge(v0, v1).is_valid())
        return false;

    return true;
}

void SurfaceMesh::flip(Edge e)
{
    //let's make it sure it is actually checked
    assert(is_flip_ok(e));

    Halfedge a0 = e.halfedge(0);
    Halfedge b0 = e.halfedge(1);

    Halfedge a1 = next_halfedge(a0);
    Halfedge a2 = next_halfedge(a1);

    Halfedge b1 = next_halfedge(b0);
    Halfedge b2 = next_halfedge(b1);

    Vertex va0 = to_vertex(a0);
    Vertex va1 = to_vertex(a1);

    Vertex vb0 = to_vertex(b0);
    Vertex vb1 = to_vertex(b1);

    Face fa = face(a0);
    Face fb = face(b0);

    set_vertex(a0, va1);
    set_vertex(b0, vb1);

    set_next_halfedge(a0, a2);
    set_next_halfedge(a2, b1);
    set_next_halfedge(b1, a0);

    set_next_halfedge(b0, b2);
    set_next_halfedge(b2, a1);
    set_next_halfedge(a1, b0);

    set_face(a1, fb);
    set_face(b1, fa);

    set_halfedge(fa, a0);
    set_halfedge(fb, b0);

    if (halfedge(va0) == b0)
        set_halfedge(va0, a1);
    if (halfedge(vb0) == a0)
        set_halfedge(vb0, b1);
}

void SurfaceMesh::split_mesh(std::vector<SurfaceMesh>& output,
                             FaceProperty<IndexType>& face_dist,
                             VertexProperty<IndexType>& vertex_dist)
{
    garbage_collection();

    std::vector<PropertyMap<Face>> fprop_maps;
    std::vector<PropertyMap<Vertex>> vprop_maps;
    std::vector<PropertyMap<Edge>> eprop_maps;
    std::vector<PropertyMap<Halfedge>> hprop_maps;
    for (auto& mesh : output)
    {
        mesh.clear();
        mesh.copy_properties(*this);
        mesh.oprops_ = oprops_;
        fprop_maps.push_back(mesh.gen_face_copy_map(*this));
        vprop_maps.push_back(mesh.gen_vertex_copy_map(*this));
        eprop_maps.push_back(mesh.gen_edge_copy_map(*this));
        hprop_maps.push_back(mesh.gen_halfedge_copy_map(*this));
    }

    auto f_map = add_face_property<Face>("f:split_map");
    auto v_map = add_vertex_property<Vertex>("v:split_map");
    auto h_map = add_halfedge_property<Halfedge>("h:split_map");

    size_t n = output.size();

    size_t num_threads = std::min((size_t)omp_get_max_threads(), n);

    std::vector<size_t> next_face(n, 0);
    std::vector<size_t> next_vertex(n, 0);
    std::vector<size_t> next_edge(n, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_i = omp_get_thread_num();
        size_t start = thread_i * n / num_threads;
        size_t end = (thread_i + 1) * n / num_threads;

        for (Face f : faces())
        {
            auto id = face_dist[f];
            if (id >= start && id < end)
                f_map[f] = Face(next_face[id]++);
        }
        for (Vertex v : vertices())
        {
            auto id = vertex_dist[v];
            if (id >= start && id < end)
                v_map[v] = Vertex(next_vertex[id]++);
        }
        for (Edge e : edges())
        {
            auto id = vertex_dist[vertex(e, 0)];
            if (id != vertex_dist[vertex(e, 1)])
                throw std::runtime_error("Edge split error");
            if (id >= start && id < end)
            {
                auto h = e.halfedge(0);
                h_map[h] = Edge(next_edge[id]++).halfedge(0);
                h_map[h.opposite()] = h_map[h].opposite();
            }
        }
        for (size_t i = start; i < end; i++)
        {
            output[i].new_faces(next_face[i]);
            output[i].new_vertices(next_vertex[i]);
            output[i].new_edges(next_edge[i]);
        }
    }

    #pragma omp parallel for schedule(static,256)
    for (size_t i = 0; i < faces_size(); i++)
    {
        Face f(i);
        auto id = face_dist[f];
        if (is_deleted(f) || id >= n)
            continue;
        auto& mesh = output[id];
        fprop_maps[id].copy(f, f_map[f]);
        mesh.set_halfedge(f_map[f], h_map[halfedge(f)]);
    }
    #pragma omp parallel for schedule(static,256)
    for (size_t i = 0; i < vertices_size(); i++)
    {
        Vertex v(i);
        auto id = vertex_dist[v];
        if (is_deleted(v) || id >= n)
            continue;
        auto& mesh = output[id];
        vprop_maps[id].copy(v, v_map[v]);
        mesh.position(v_map[v]) = position(v);
        Halfedge h = halfedge(v);
        if (h.is_valid())
            mesh.set_halfedge(v_map[v], h_map[h]);
    }
    #pragma omp parallel for schedule(static,256)
    for (size_t i = 0; i < halfedges_size(); i++)
    {
        Halfedge h(i);
        auto id = vertex_dist[to_vertex(h)];
        if (is_deleted(h) || id >= n)
            continue;
        auto& mesh = output[id];
        hprop_maps[id].copy(h, h_map[h]);
        if (i % 2 == 0)
            eprop_maps[id].copy(h.edge(), h_map[h].edge());
        auto& in = hconn_[h];
        auto& out = mesh.hconn_[h_map[h]];
        out.vertex_ = v_map[in.vertex_];
        out.next_halfedge_ = h_map[in.next_halfedge_];
        out.prev_halfedge_ = h_map[in.prev_halfedge_];
        if (in.face_.is_valid())
            out.face_ = f_map[in.face_];
    }

    remove_face_property(f_map);
    remove_vertex_property(v_map);
    remove_halfedge_property(h_map);
    for (auto& mesh : output)
    {
        mesh.remove_face_property<IndexType>(face_dist.name());
        mesh.remove_vertex_property<IndexType>(vertex_dist.name());
    }
}

void SurfaceMesh::join_mesh(const std::vector<const SurfaceMesh*>& input)
{
    if (input.empty())
        return;

    garbage_collection();

    oprops_ = input[0]->oprops_;

    std::vector<PropertyMap<Face>> fprop_maps;
    std::vector<PropertyMap<Vertex>> vprop_maps;
    std::vector<PropertyMap<Edge>> eprop_maps;
    std::vector<PropertyMap<Halfedge>> hprop_maps;
    size_t added_faces = 0, added_vertices = 0, added_edges = 0;
    std::vector<IndexType> face_offsets = { (IndexType)faces_size() };
    std::vector<IndexType> vertex_offsets = { (IndexType)vertices_size() };
    std::vector<IndexType> halfedge_offsets = { (IndexType)halfedges_size() };
    for (auto mesh : input)
    {
        copy_properties(*mesh);
        fprop_maps.push_back(gen_face_copy_map(*mesh));
        vprop_maps.push_back(gen_vertex_copy_map(*mesh));
        eprop_maps.push_back(gen_edge_copy_map(*mesh));
        hprop_maps.push_back(gen_halfedge_copy_map(*mesh));

        if (mesh->has_garbage())
            throw InvalidInputException("SurfaceMesh::join_mesh: Input Mesh has garbage");

        added_faces += mesh->n_faces();
        added_vertices += mesh->n_vertices();
        added_edges += mesh->n_edges();

        if (face_offsets.size() < input.size())
        {
            face_offsets.push_back(face_offsets.back() + mesh->n_faces());
            vertex_offsets.push_back(vertex_offsets.back() + mesh->n_vertices());
            halfedge_offsets.push_back(halfedge_offsets.back() + mesh->n_halfedges());
        }
    }

    new_faces(added_faces);
    new_vertices(added_vertices);
    new_edges(added_edges);

    for (size_t i = 0; i < input.size(); i++)
    {
        auto& mesh = *input[i];
        IndexType face_offset = face_offsets[i];
        IndexType vertex_offset = vertex_offsets[i];
        IndexType halfedge_offset = halfedge_offsets[i];
        auto& fprop_map = fprop_maps[i];
        auto& vprop_map = vprop_maps[i];
        auto& eprop_map = eprop_maps[i];
        auto& hprop_map = hprop_maps[i];
        #pragma omp parallel for schedule(static,256)
        for (size_t j = 0; j < mesh.faces_size(); j++)
        {
            Face f(j), fm(face_offset + j);
            fprop_map.copy(f, fm);
            set_halfedge(fm, Halfedge(halfedge_offset + mesh.halfedge(f).idx()));
        }
        #pragma omp parallel for schedule(static,256)
        for (size_t j = 0; j < mesh.vertices_size(); j++)
        {
            Vertex v(j), vm(vertex_offset + j);
            vprop_map.copy(v, vm);
            position(vm) = mesh.position(v);
            Halfedge h = mesh.halfedge(v);
            if (h.is_valid())
                set_halfedge(vm, Halfedge(halfedge_offset + h.idx()));
        }
        #pragma omp parallel for schedule(static,256)
        for (size_t j = 0; j < mesh.halfedges_size(); j++)
        {
            Halfedge h(j), hm(halfedge_offset + j);
            hprop_map.copy(h, hm);
            if (j % 2 == 0)
                eprop_map.copy(h.edge(), hm.edge());
            auto& in = mesh.hconn_[h];
            auto& out = hconn_[hm];
            out.vertex_ = Vertex(vertex_offset + in.vertex_.idx());
            out.next_halfedge_ = Halfedge(halfedge_offset + in.next_halfedge_.idx());
            out.prev_halfedge_ = Halfedge(halfedge_offset + in.prev_halfedge_.idx());
            if (in.face_.is_valid())
                out.face_ = Face(face_offset + in.face_.idx());
        }
    }
}

bool SurfaceMesh::is_collapse_ok(Halfedge v0v1)
{
    Halfedge v1v0(v0v1.opposite());
    Vertex v0(to_vertex(v1v0));
    Vertex v1(to_vertex(v0v1));
    Vertex vl, vr;
    Halfedge h1, h2;

    // the edges v1-vl and vl-v0 must not be both boundary edges
    if (!is_boundary(v0v1))
    {
        vl = to_vertex(next_halfedge(v0v1));
        h1 = next_halfedge(v0v1);
        h2 = next_halfedge(h1);
        if (is_boundary(h1.opposite()) &&
            is_boundary(h2.opposite()))
            return false;
    }

    // the edges v0-vr and vr-v1 must not be both boundary edges
    if (!is_boundary(v1v0))
    {
        vr = to_vertex(next_halfedge(v1v0));
        h1 = next_halfedge(v1v0);
        h2 = next_halfedge(h1);
        if (is_boundary(h1.opposite()) &&
            is_boundary(h2.opposite()))
            return false;
    }

    // if vl and vr are equal or both invalid -> fail
    if (vl == vr)
        return false;

    // edge between two boundary vertices should be a boundary edge
    if (is_boundary(v0) && is_boundary(v1) && !is_boundary(v0v1) &&
        !is_boundary(v1v0))
        return false;

    // test intersection of the one-rings of v0 and v1
    for (Vertex vv : vertices(v0))
    {
        if (vv != v1 && vv != vl && vv != vr)
            if (find_halfedge(vv, v1).is_valid())
                return false;
    }

    // passed all tests
    return true;
}

bool SurfaceMesh::is_removal_ok(Edge e)
{
    Halfedge h0 = e.halfedge(0);
    Halfedge h1 = e.halfedge(1);
    Vertex v0 = to_vertex(h0);
    Vertex v1 = to_vertex(h1);
    Face f0 = face(h0);
    Face f1 = face(h1);

    // boundary?
    if (!f0.is_valid() || !f1.is_valid())
        return false;

    // same face?
    if (f0 == f1)
        return false;

    // are the two faces connect through another vertex?
    for (auto v : vertices(f0))
        if (v != v0 && v != v1)
            for (auto f : faces(v))
                if (f == f1)
                    return false;

    return true;
}

bool SurfaceMesh::remove_edge(Edge e)
{
    if (!is_removal_ok(e))
        return false;

    Halfedge h0 = e.halfedge(0);
    Halfedge h1 = e.halfedge(1);

    Vertex v0 = to_vertex(h0);
    Vertex v1 = to_vertex(h1);

    Face f0 = face(h0);
    Face f1 = face(h1);

    Halfedge h0_prev = prev_halfedge(h0);
    Halfedge h0_next = next_halfedge(h0);
    Halfedge h1_prev = prev_halfedge(h1);
    Halfedge h1_next = next_halfedge(h1);

    // adjust vertex->halfedge
    if (halfedge(v0) == h1)
        set_halfedge(v0, h0_next);
    if (halfedge(v1) == h0)
        set_halfedge(v1, h1_next);

    // adjust halfedge->face
    for (auto h : halfedges(f0))
        set_face(h, f1);

    // adjust halfedge->halfedge
    set_next_halfedge(h1_prev, h0_next);
    set_next_halfedge(h0_prev, h1_next);

    // adjust face->halfedge
    if (halfedge(f1) == h1)
        set_halfedge(f1, h1_next);

    // delete face f0 and edge e
    fdeleted_[f0] = true;
    ++deleted_faces_;
    edeleted_[e] = true;
    ++deleted_edges_;
    has_garbage_ = true;

    return true;
}

void SurfaceMesh::collapse(Halfedge h)
{
    Halfedge h0 = h;
    Halfedge h1 = prev_halfedge(h0);
    Halfedge o0 = h0.opposite();
    Halfedge o1 = next_halfedge(o0);

    // remove edge
    remove_edge_helper(h0);

    // remove loops
    if (next_halfedge(next_halfedge(h1)) == h1)
        remove_loop_helper(h1);
    if (next_halfedge(next_halfedge(o1)) == o1)
        remove_loop_helper(o1);
}

void SurfaceMesh::remove_edge_helper(Halfedge h)
{
    Halfedge hn = next_halfedge(h);
    Halfedge hp = prev_halfedge(h);

    Halfedge o = h.opposite();
    Halfedge on = next_halfedge(o);
    Halfedge op = prev_halfedge(o);

    Face fh = face(h);
    Face fo = face(o);

    Vertex vh = to_vertex(h);
    Vertex vo = to_vertex(o);

    // halfedge -> vertex
    for (Halfedge hh : halfedges(vo))
        set_vertex(hh.opposite(), vh);

    // halfedge -> halfedge
    set_next_halfedge(hp, hn);
    set_next_halfedge(op, on);

    // face -> halfedge
    if (fh.is_valid())
        set_halfedge(fh, hn);
    if (fo.is_valid())
        set_halfedge(fo, on);

    // vertex -> halfedge
    if (halfedge(vh) == o)
        set_halfedge(vh, hn);
    adjust_outgoing_halfedge(vh);
    set_halfedge(vo, Halfedge());

    // delete stuff
    vdeleted_[vo] = true;
    ++deleted_vertices_;
    edeleted_[h.edge()] = true;
    ++deleted_edges_;
    has_garbage_ = true;
}

void SurfaceMesh::remove_loop_helper(Halfedge h)
{
    Halfedge h0 = h;
    Halfedge h1 = next_halfedge(h0);

    Halfedge o0 = h0.opposite();
    Halfedge o1 = h1.opposite();

    Vertex v0 = to_vertex(h0);
    Vertex v1 = to_vertex(h1);

    Face fh = face(h0);
    Face fo = face(o0);

    // is it a loop ?
    assert((next_halfedge(h1) == h0) && (h1 != o0));

    // halfedge -> halfedge
    set_next_halfedge(h1, next_halfedge(o0));
    set_prev_halfedge(h1, prev_halfedge(o0));

    // halfedge -> face
    set_face(h1, fo);

    // vertex -> halfedge
    set_halfedge(v0, h1);
    adjust_outgoing_halfedge(v0);
    set_halfedge(v1, o1);
    adjust_outgoing_halfedge(v1);

    // face -> halfedge
    if (fo.is_valid() && halfedge(fo) == o0)
        set_halfedge(fo, h1);

    // delete stuff
    if (fh.is_valid())
    {
        fdeleted_[fh] = true;
        ++deleted_faces_;
    }
    edeleted_[h.edge()] = true;
    ++deleted_edges_;
    has_garbage_ = true;
}

void SurfaceMesh::delete_vertex(Vertex v)
{
    if (is_deleted(v))
        return;

    // collect incident faces
    std::vector<Face>& incident_faces = delete_incident_faces_;
    incident_faces.clear();
    incident_faces.reserve(6);

    for (auto f : faces(v))
        incident_faces.push_back(f);

    // delete incident faces
    for (auto f : incident_faces)
        delete_face(f);

    // mark v as deleted if not yet done by delete_face()
    if (!vdeleted_[v])
    {
        vdeleted_[v] = true;
        deleted_vertices_++;
        has_garbage_ = true;
    }
}

void SurfaceMesh::delete_edge(Edge e)
{
    if (is_deleted(e))
        return;

    Face f0 = face(e.halfedge(0));
    Face f1 = face(e.halfedge(1));

    if (f0.is_valid())
        delete_face(f0);
    if (f1.is_valid())
        delete_face(f1);
}

void SurfaceMesh::delete_face(Face f)
{
    if (fdeleted_[f])
        return;

    // mark face deleted
    if (!fdeleted_[f])
    {
        fdeleted_[f] = true;
        deleted_faces_++;
    }

    if (!halfedge(f).is_valid())
        return;

    // boundary edges of face f to be deleted
    std::vector<Edge>& deleted_edges = delete_deleted_edges_;
    deleted_edges.clear();

    // vertices of face f for updating their outgoing halfedge
    std::vector<Vertex>& vertices = temp_face_vertices_;
    vertices.clear();

    // for all halfedges of face f do:
    //   1) invalidate face handle.
    //   2) collect all boundary halfedges, set them deleted
    //   3) store vertex handles
    for (Halfedge hc : halfedges(f))
    {
        set_face(hc, Face());

        if (is_boundary(hc.opposite()))
            deleted_edges.push_back(hc.edge());

        vertices.push_back(to_vertex(hc));
    }

    // delete all collected (half)edges
    // delete isolated vertices
    if (!deleted_edges.empty())
    {
        Halfedge h0, h1, next0, next1, prev0, prev1;
        Vertex v0, v1;

        for (auto e : deleted_edges)
        {
            h0 = e.halfedge(0);
            v0 = to_vertex(h0);
            next0 = next_halfedge(h0);
            prev0 = prev_halfedge(h0);

            h1 = e.halfedge(1);
            v1 = to_vertex(h1);
            next1 = next_halfedge(h1);
            prev1 = prev_halfedge(h1);

            // adjust next and prev handles
            set_next_halfedge(prev0, next1);
            set_next_halfedge(prev1, next0);

            // mark edge deleted
            if (!edeleted_[e])
            {
                edeleted_[e] = true;
                deleted_edges_++;
            }

            // update v0
            if (halfedge(v0) == h1)
            {
                if (next0 == h1)
                {
                    if (!vdeleted_[v0])
                    {
                        vdeleted_[v0] = true;
                        deleted_vertices_++;
                    }
                }
                else
                    set_halfedge(v0, next0);
            }

            // update v1
            if (halfedge(v1) == h0)
            {
                if (next1 == h0)
                {
                    if (!vdeleted_[v1])
                    {
                        vdeleted_[v1] = true;
                        deleted_vertices_++;
                    }
                }
                else
                    set_halfedge(v1, next1);
            }
        }
    }

    // update outgoing halfedge handles of remaining vertices
    for (auto v : vertices)
        if (!is_deleted(v))
            adjust_outgoing_halfedge(v);

    has_garbage_ = true;
}

void SurfaceMesh::delete_many_faces(const FaceProperty<bool>& faces)
{
    size_t deleted = 0;
    #pragma omp parallel
    {
        std::vector<Face> delete_faces;
        #pragma omp for schedule(dynamic,64) nowait
        for (size_t i = 0; i < faces_size(); i++)
        {
            Face f(i);
            if (!fdeleted_[f] && faces[f])
                delete_faces.push_back(f);
        }
        #pragma omp critical
        {
            deleted += delete_faces.size();
            // this has to be done in a critical section, because fdeleted_ is std::vector<bool>
            // and we cannot assign to elements in a parallel loop
            for (Face f : delete_faces)
                fdeleted_[f] = true;
        }
    }
    deleted_faces_ += deleted;
    if (deleted == 0)
        return;
    has_garbage_ = true;

    #pragma omp parallel
    {
        std::vector<Edge> delete_edges;
        #pragma omp for schedule(dynamic,64) nowait
        for (size_t i = 0; i < edges_size(); i++)
        {
            Edge e(i);
            if (edeleted_[e])
                continue;

            Halfedge h0 = e.halfedge(0);
            Halfedge h1 = e.halfedge(1);
            Face f0(face(h0));
            Face f1(face(h1));
            if (f0.is_valid() && fdeleted_[f0])
            {
                set_face(h0, Face());
                f0 = Face();
            }
            if (f1.is_valid() && fdeleted_[f1])
            {
                set_face(h1, Face());
                f1 = Face();
            }
            if (!f0.is_valid() && !f1.is_valid())
                delete_edges.push_back(e);
        }
        #pragma omp critical
        {
            deleted_edges_ += delete_edges.size();
            for (Edge e : delete_edges)
                edeleted_[e] = true;
        }

        #pragma omp barrier

        std::vector<Vertex> delete_vertices;
        #pragma omp for schedule(dynamic,64) nowait
        for (size_t i = 0; i < vertices_size(); i++)
        {
            Vertex v(i);
            if (vdeleted_[v] || !edeleted_[halfedge(v).edge()])
                continue;

            // find new outgoing halfedge for v
            bool found = false;
            for (Halfedge h : halfedges(v))
            {
                if (!edeleted_[h.edge()])
                {
                    set_halfedge(v, h);
                    found = true;
                    break;
                }
            }
            if (!found)
                delete_vertices.push_back(v);
        }
        #pragma omp critical
        {
            deleted_vertices_ += delete_vertices.size();
            for (Vertex v : delete_vertices)
                vdeleted_[v] = true;
        }

        #pragma omp barrier

        #pragma omp critical
        {
            for (Edge e : delete_edges)
            {
                // see edge removal in delete_face
                Halfedge h0 = e.halfedge(0);
                Halfedge h1 = e.halfedge(1);
                set_next_halfedge(prev_halfedge(h0), next_halfedge(h1));
                set_next_halfedge(prev_halfedge(h1), next_halfedge(h0));
            }
        }

        #pragma omp barrier

        #pragma omp for schedule(dynamic,64)
        for (size_t i = 0; i < vertices_size(); i++)
            if (!vdeleted_[Vertex(i)])
                adjust_outgoing_halfedge(Vertex(i));
    }
}

void SurfaceMesh::garbage_collection()
{
    if (!has_garbage_)
        return;

    int nV(vertices_size()), nE(edges_size()), nH(halfedges_size()),
        nF(faces_size());

    Vertex v;
    Halfedge h;
    Face f;

    // setup handle mapping
    VertexProperty<Vertex> vmap =
        add_vertex_property<Vertex>("v:garbage-collection");
    HalfedgeProperty<Halfedge> hmap =
        add_halfedge_property<Halfedge>("h:garbage-collection");
    FaceProperty<Face> fmap = add_face_property<Face>("f:garbage-collection");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nV; ++i)
        vmap[Vertex(i)] = Vertex(i);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nH; ++i)
        hmap[Halfedge(i)] = Halfedge(i);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nF; ++i)
        fmap[Face(i)] = Face(i);

    // remove deleted vertices
    std::thread v_thread([&]()
    {
        int i0 = 0;
        int i1 = nV - 1;

        while (1)
        {
            // find first deleted and last un-deleted
            while (!vdeleted_[Vertex(i0)] && i0 < i1)
                ++i0;
            while (vdeleted_[Vertex(i1)] && i0 < i1)
                --i1;
            if (i0 >= i1)
                break;

            // swap
            vprops_.swap(i0, i1);
        };

        // remember new size
        nV = vdeleted_[Vertex(i0)] ? i0 : i0 + 1;
    });

    // remove deleted edges
    std::thread e_thread([&]()
    {
        int i0 = 0;
        int i1 = nE - 1;

        while (1)
        {
            // find first deleted and last un-deleted
            while (!edeleted_[Edge(i0)] && i0 < i1)
                ++i0;
            while (edeleted_[Edge(i1)] && i0 < i1)
                --i1;
            if (i0 >= i1)
                break;

            // swap
            eprops_.swap(i0, i1);
            hprops_.swap(2 * i0, 2 * i1);
            hprops_.swap(2 * i0 + 1, 2 * i1 + 1);
        };

        // remember new size
        nE = edeleted_[Edge(i0)] ? i0 : i0 + 1;
        nH = 2 * nE;
    });

    // remove deleted faces
    {
        int i0 = 0;
        int i1 = nF - 1;

        while (1)
        {
            // find 1st deleted and last un-deleted
            while (!fdeleted_[Face(i0)] && i0 < i1)
                ++i0;
            while (fdeleted_[Face(i1)] && i0 < i1)
                --i1;
            if (i0 >= i1)
                break;

            // swap
            fprops_.swap(i0, i1);
        };

        // remember new size
        nF = fdeleted_[Face(i0)] ? i0 : i0 + 1;
    }

    v_thread.join();
    e_thread.join();

    // update vertex connectivity
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nV; ++i)
    {
        Vertex v(i);
        if (!is_isolated(v))
            set_halfedge(v, hmap[halfedge(v)]);
    }

    // update halfedge connectivity
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nH; ++i)
    {
        Halfedge h(i);
        set_vertex(h, vmap[to_vertex(h)]);
        set_next_halfedge(h, hmap[next_halfedge(h)]);
        if (!is_boundary(h))
            set_face(h, fmap[face(h)]);
    }

    // update handles of faces
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nF; ++i)
    {
        Face f(i);
        set_halfedge(f, hmap[halfedge(f)]);
    }

    // remove handle maps
    remove_vertex_property(vmap);
    remove_halfedge_property(hmap);
    remove_face_property(fmap);

    // finally resize arrays
    vprops_.resize(nV);
    vprops_.free_memory();
    hprops_.resize(nH);
    hprops_.free_memory();
    eprops_.resize(nE);
    eprops_.free_memory();
    fprops_.resize(nF);
    fprops_.free_memory();

    deleted_vertices_ = deleted_edges_ = deleted_faces_ = 0;
    has_garbage_ = false;
}

} // namespace pmp
