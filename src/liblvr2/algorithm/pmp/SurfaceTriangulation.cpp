// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see PMP_LICENSE.txt for details.

#include "lvr2/algorithm/pmp/SurfaceTriangulation.h"

#include <limits>

namespace pmp {

SurfaceTriangulation::SurfaceTriangulation(SurfaceMesh& mesh) : mesh_(mesh)
{
    points_ = mesh_.vertex_property<Point>("v:point");
}

void SurfaceTriangulation::triangulate(Objective o)
{
    for (auto f : mesh_.faces())
        triangulate(f, o);
}

void SurfaceTriangulation::triangulate(Face f, Objective o)
{
    // store objective
    objective_ = o;

    // collect polygon halfedges
    Halfedge h0 = mesh_.halfedge(f);
    halfedges_.clear();
    vertices_.clear();
    Halfedge h = h0;
    do
    {
        if (!mesh_.is_manifold(mesh_.to_vertex(h)))
        {
            auto what = "[SurfaceTriangulation] Non-manifold polygon";
            throw InvalidInputException(what);
        }

        halfedges_.push_back(h);
        vertices_.push_back(mesh_.to_vertex(h));
    } while ((h = mesh_.next_halfedge(h)) != h0);

    // do we have at least four vertices?
    const int n = halfedges_.size();
    if (n <= 3)
        return;

    // compute minimal triangulation by dynamic programming
    weight_.clear();
    weight_.resize(n,
                   std::vector<Scalar>(n, std::numeric_limits<Scalar>::max()));
    index_.clear();
    index_.resize(n, std::vector<int>(n, 0));

    int i, j, m, k, imin;
    Scalar w, wmin;

    // initialize 2-gons
    for (i = 0; i < n - 1; ++i)
    {
        weight_[i][i + 1] = 0.0;
        index_[i][i + 1] = -1;
    }

    // n-gons with n>2
    for (j = 2; j < n; ++j)
    {
        // for all n-gons [i,i+j]
        for (i = 0; i < n - j; ++i)
        {
            k = i + j;
            wmin = std::numeric_limits<Scalar>::max();
            imin = -1;

            // find best split i < m < i+j
            for (m = i + 1; m < k; ++m)
            {
                switch (objective_)
                {
                    case Objective::MIN_AREA:
                        w = weight_[i][m] + compute_weight(i, m, k) +
                            weight_[m][k];
                        break;
                    case Objective::MAX_ANGLE:
                        w = std::max(
                            weight_[i][m],
                            std::max(compute_weight(i, m, k), weight_[m][k]));
                        break;
                    default:
                        // should never happen
                        exit(1);
                        break;
                }

                if (w < wmin)
                {
                    wmin = w;
                    imin = m;
                }
            }

            weight_[i][k] = wmin;
            index_[i][k] = imin;
        }
    }

    // now add triangles to mesh
    std::vector<ivec2> todo;
    todo.reserve(n);
    todo.push_back(ivec2(0, n - 1));
    while (!todo.empty())
    {
        ivec2 tri = todo.back();
        todo.pop_back();
        int start = tri[0];
        int end = tri[1];
        if (end - start < 2)
            continue;
        int split = index_[start][end];

        insert_edge(start, split);
        insert_edge(split, end);

        todo.push_back(ivec2(start, split));
        todo.push_back(ivec2(split, end));
    }

    // clean up
    weight_.clear();
    index_.clear();
    halfedges_.clear();
    vertices_.clear();
}

Scalar SurfaceTriangulation::compute_weight(int i, int j, int k) const
{
    const Vertex a = vertices_[i];
    const Vertex b = vertices_[j];
    const Vertex c = vertices_[k];

    // If one of the potential edges already exists this would result in an
    // invalid triangulation. This happens for suzanne.obj. Prevent this by
    // giving infinite weight.
    if (is_edge(a, b) && is_edge(b, c) && is_edge(c, a))
        return std::numeric_limits<Scalar>::max();

    const Point& pa = points_[a];
    const Point& pb = points_[b];
    const Point& pc = points_[c];

    Scalar w = std::numeric_limits<Scalar>::max();
    switch (objective_)
    {
        // compute squared triangle area
        case Objective::MIN_AREA:
            w = (pb - pa).cross(pc - pa).squaredNorm();
            break;

        // compute one over minimum angle
        // or cosine of minimum angle
        // maximum cosine (which should then be minimized)
        case Objective::MAX_ANGLE:
            Scalar cosa = (pb - pa).normalized().dot((pc - pa).normalized());
            Scalar cosb = (pa - pb).normalized().dot((pc - pb).normalized());
            Scalar cosc = (pa - pc).normalized().dot((pb - pc).normalized());
            w = std::max(cosa, std::max(cosb, cosc));
            break;
    }

    return w;
}

bool SurfaceTriangulation::is_edge(Vertex a, Vertex b) const
{
    return mesh_.find_halfedge(a, b).is_valid();
}

bool SurfaceTriangulation::insert_edge(int i, int j)
{
    Halfedge h0 = halfedges_[i];
    Halfedge h1 = halfedges_[j];
    Vertex v0 = vertices_[i];
    Vertex v1 = vertices_[j];

    // does edge already exist?
    if (mesh_.find_halfedge(v0, v1).is_valid())
    {
        return false;
    }

    // can we reach v1 from h0?
    {
        Halfedge h = h0;
        do
        {
            h = mesh_.next_halfedge(h);
            if (mesh_.to_vertex(h) == v1)
            {
                mesh_.insert_edge(h0, h);
                return true;
            }
        } while (h != h0);
    }

    // can we reach v0 from h1?
    {
        Halfedge h = h1;
        do
        {
            h = mesh_.next_halfedge(h);
            if (mesh_.to_vertex(h) == v0)
            {
                mesh_.insert_edge(h1, h);
                return true;
            }
        } while (h != h1);
    }

    return false;
}

} // namespace pmp
