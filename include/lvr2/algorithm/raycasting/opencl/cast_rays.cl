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

#pragma OPENCL EXTENSION cl_intel_printf : enable

#define EPSILON 0.0000001
#define PI 3.14159265

typedef struct {
    bool intersects;
    float3 intersection;
} intersection_result;

typedef struct {
    float3 vertex0;
    float3 vertex1;
    float3 vertex2;
} triangle_t;

/**
 * Credits go to: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
 */
intersection_result ray_intersects_triangle(
    float3 ray,
    float3 ray_origin,
    triangle_t triangle
)
{
    float3 edge1, edge2, h, s, q;
    float a, f, u, v;
    intersection_result out;

    edge1 = triangle.vertex1 - triangle.vertex0;
    edge2 = triangle.vertex2 - triangle.vertex0;
    h = cross(ray, edge2);
    a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON)
    {
        out.intersects = false;
        return out;
    }

    f = 1/a;
    s = ray_origin - triangle.vertex0;
    u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
    {
        out.intersects = false;
        return out;
    }

    q = cross(s, edge1);
    v = f * dot(ray, q);

    if (v < 0.0 || u + v > 1.0)
    {
        out.intersects = false;
        return out;
    }

    float t = f * dot(edge2, q);
    if (t > EPSILON)
    {
        out.intersects = true;
        out.intersection = ray_origin + ray * t;
        return out;
    }

    out.intersects = false;
    return out;
}

/**
 * Credits go to: http://www.geeks3d.com/20141201/how-to-rotate-a-vertex-by-a-quaternion-in-glsl/
 */
float4 quat_from_axis_angle(float3 axis, float angle)
{
    float4 qr;

    float half_angle = (angle * 0.5) * PI / 180.0;
    qr.x = axis.x * sin(half_angle);
    qr.y = axis.y * sin(half_angle);
    qr.z = axis.z * sin(half_angle);
    qr.w = cos(half_angle);

    return qr;
}

float4 quat_conj(float4 q)
{
    return (float4)(-q.x, -q.y, -q.z, q.w);
}

float4 quat_mult(float4 q1, float4 q2)
{
    float4 qr;

    qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

    return qr;
}

float3 rotate_vertex_position(float3 position, float3 axis, float angle)
{
    //float4 qr = quat_from_axis_angle(axis, angle);
    //float4 qr_conj = quat_conj(qr);
    //float4 q_pos = (float4)(position.x, position.y, position.z, 0);
    //float4 q_tmp = quat_mult(qr, q_pos);
    //qr = quat_mult(q_tmp, qr_conj);
    //return (float3)(qr.x, qr.y, qr.z);

    // the above optimized
    float4 q = quat_from_axis_angle(axis, angle);
    return position + (2.0f * cross(q.xyz, cross(q.xyz, position) + q.w * position));
}

__kernel void cast_rays(
    __global float* vertices,
    __global int* faces,
    __global float* rays,
    __global float* ray_origin,
    __global int* num_faces,
    __global float* result,
    __global uchar* result_hits
)
{
    int id = get_global_id(0);

    // current ray: rays[id*3], rays[id*3+1], rays[id*3+2]
    float3 ray = (float3)(rays[0], rays[1], rays[2]);
    float3 ray_o = (float3)(ray_origin[0], ray_origin[1], ray_origin[2]);

    float3 y_axis = (float3)(0, 1, 0);
    float3 x_axis = (float3)(1, 0, 0);

    ray = rotate_vertex_position(ray, y_axis, id * ((180.0/1000.0) - 90));
    ray = rotate_vertex_position(ray, x_axis, id * ((90.0/16.0) - 45));

    uchar ray_hit = 0;

    // initialize result memory with zeros
    result[id*3] = 0;
    result[id*3+1] = 0;
    result[id*3+2] = 0;

    for (int i = 0; i < *num_faces; ++i)
    {
        // current face: faces[i*3], faces[i*3+1], faces[i*3+2]
        int3 face = (int3)(faces[i*3], faces[i*3+1], faces[i*3+2]);

        // vertices of face: vertices[faces[i*3]], vertices[faces[i*3]+1], vertices[faces[i*3]+2] etc.
        triangle_t triangle;
        triangle.vertex0 = (float3)(vertices[face.x*3], vertices[face.x*3+1], vertices[face.x*3+2]);
        triangle.vertex1 = (float3)(vertices[face.y*3], vertices[face.y*3+1], vertices[face.y*3+2]);
        triangle.vertex2 = (float3)(vertices[face.z*3], vertices[face.z*3+1], vertices[face.z*3+2]);

        intersection_result i_result = ray_intersects_triangle(ray, ray_o, triangle);
        if (ray_hit == 0 && i_result.intersects)
        {
            result[id*3] = i_result.intersection.x;
            result[id*3+1] = i_result.intersection.y;
            result[id*3+2] = i_result.intersection.z;
            ray_hit = 1;
        }
    }

    result_hits[id] = ray_hit;
}
