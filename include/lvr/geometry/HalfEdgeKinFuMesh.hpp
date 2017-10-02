/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * HalfEdgeKinFuMesh.hpp
 *
 *  @date 13.11.2015
 *  @author Tristan Igelbrink (Tristan@Igelbrink.com)
 */

#ifndef HalfEdgeKinFuMesh_H_
#define HalfEdgeKinFuMesh_H_

#include <boost/unordered_map.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>
#include <stack>
#include <set>
#include <list>
#include <map>
#include <sstream>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <queue>

#ifndef __APPLE__
#include <GL/glu.h>
#include <GL/glut.h>
#else
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#endif

//mein zeugs
#include <opencv2/opencv.hpp>
#include <kfusion/types.hpp>


using namespace std;
#include "HalfEdgeMesh.hpp"

namespace lvr
{

/**
 * @brief A implementation of a half edge triangle mesh.
 */
template<typename VertexT, typename NormalT>
class HalfEdgeKinFuMesh : public HalfEdgeMesh<VertexT, NormalT>
{
public:

    typedef HalfEdgeFace<VertexT, NormalT> HFace;
    typedef HalfEdgeVertex<VertexT, NormalT> HVertex;

    typedef HalfEdge<HVertex, HFace> HEdge;

    typedef HFace*                              FacePtr;
    typedef HVertex*                            VertexPtr;
    typedef Region<VertexT, NormalT>*           RegionPtr;
    typedef HEdge*                              EdgePtr;

    typedef vector<HFace*>  FaceVector;
    typedef vector<Region<VertexT, NormalT>* >  RegionVector;
    typedef vector<HEdge*>   EdgeVector;
    typedef vector<HVertex* > VertexVector;


    HalfEdgeKinFuMesh();

    /**
     * @brief   Ctor.
     *
     * @param    pm    a pointer to the point cloud manager
     */
    HalfEdgeKinFuMesh( typename PointsetSurface<VertexT>::Ptr pm );

    /**
     * @brief   Creates a HalfEdgeMesh from the given mesh buffer
     */
    HalfEdgeKinFuMesh( MeshBufferPtr model);

    /**
     * @brief   Dtor.
     */
    virtual ~HalfEdgeKinFuMesh();

    void addMesh(HalfEdgeKinFuMesh* slice, bool texture);



    /**
     * @brief    Applies region growing and regression plane algorithms and deletes small
     *             regions
     *
     * @param iterations        The number of iterations to use
     *
     * @param normalThreshold   The normal threshold
     *
     * @param minRegionSize        The minimum size of a region
     *
     * @param smallRegionSize    The size up to which a region is considered as small
     *
     * @param remove_flickering    Whether to remove flickering faces or not
     */
    virtual void optimizePlanes(int iterations, float normalThreshold, int minRegionSize = 50, int smallRegionSize = 0, bool remove_flickering = true);


    virtual HalfEdgeKinFuMesh<VertexT, NormalT>*  retesselateInHalfEdge(float fusionThreshold = 0.01, bool textured = false, int start_texture_index=0);

    void mergeVertex(VertexPtr merge_vert, VertexPtr erase_vert);

    void setFusionVertex(uint v);

    void setFusionNeighborVertex(uint v);

    void setOldFusionVertex(uint v);

    unordered_map<size_t, size_t> m_slice_verts;
    unordered_map<size_t, size_t> m_fused_verts;
    unordered_map<VertexPtr, VertexPtr> m_fusion_verts;
    size_t m_fusionNeighbors;
    //size_t m_fusionVertices;
    size_t m_fusionOldNeighbors;
    vector<FacePtr> m_fusionFaces;
    VertexVector                                m_fusionVertices;
    VertexVector                                m_oldFusionVertices;

    int projectAndMapNewImage(kfusion::ImgPose img_pose, const char* texture_output_dir="");

    std::vector<std::vector<cv::Point3f> > getBoundingRectangles(int& size);
    std::vector<std::pair<cv::Mat,float> > getTextures();

     std::vector<std::vector<cv::Point3f> > bounding_rectangles_3D;
     //first version with one texture each bounding box
     //texture + float saving best angle area was seen
    std::vector<std::pair<cv::Mat,float> > textures;

    //global bounding_rectangles

    size_t num_cams,b_rect_size,end_texture_index,start_texture_index;


protected:

    /**
     * @brief    Starts a region growing wrt the angle between the faces and returns the
     *             number of connected faces. Faces are connected means they share a common
     *             edge - a point is not a connection in this context
     *
     * @param    start_face    The face from which the region growing is started
     *
     * @param    normal        The normal to refer to
     *
     * @param    angle        the maximum angle allowed between two faces
     *
     * @param    region        The region number to apply to the faces of the found region
     *
     * @param   leafs       A vector to store the faces from which the region growing needs to start again
     *
     * @param   depth       The maximum recursion depth
     *
     * @return    Returns the size of the region - 1 (the start face is not included)
     */
    virtual int regionGrowing(FacePtr start_face, NormalT &normal, float &angle, RegionPtr region, vector<FacePtr> &leafs, unsigned int depth);

    /** TEXTURE STUFF
     * @brief getBoundingRectangles
     */
    std::vector<cv::Point3f> getBoundingRectangle(std::vector<VertexT> act_contour, NormalT normale);

    void createInitialTexture(std::vector<cv::Point3f> b_rect, int texture_index, const char* output_dir="",float pic_size_factor=1000.0);

    void getInitialUV(float x,float y,float z,std::vector<cv::Point3f> b_rect,float& u, float& v);
    void getInitialUV_b(float x,float y,float z,std::vector<std::vector<cv::Point3f> > b_rects,size_t b_rect_number,float& u, float& v);

    void fillInitialTextures(std::vector<std::vector<cv::Point3f> > b_rects,
           kfusion::ImgPose img_pose, int image_number,
           const char* texture_output_dir="");

    void fillInitialTexture(std::vector<std::vector<cv::Point3f> > b_rects,
           kfusion::ImgPose img_pose, int image_number,
           const char* texture_output_dir="");

    int fillNonPlanarColors(kfusion::ImgPose img_pose);

    void fillImageWithBlackPolygon( cv::Mat& img , cv::Point* pointarr, int size);
    void firstBehindSecondImage(cv::Mat first, cv::Mat second, cv::Mat& dst);
    void firstBehindSecondImage(cv::Mat first, cv::Mat second, cv::Mat& dst, cv::Mat mask);

    cv::Rect calcCvRect(std::vector<cv::Point2f> rect);

    std::pair<int, std::vector<int> > calcShadowTupel(std::vector<cv::Point3f> base_rect, std::vector<cv::Point3f> shadow_rect, int shadow_rect_index);

    bool calcShadowInliers(std::vector<cv::Point3f> base_rect, std::vector<cv::Point3f> shadow_rect, std::vector<int>& inliers);

    //second version with one clipped texture
    cv::Mat texture;
    std::vector< pair<size_t, cv::Size > > texture_stats;

    class sort_indices
    {
        private:
        std::vector<std::pair<cv::Mat,float> > textures;
        public:
        sort_indices(std::vector<std::pair<cv::Mat,float> > textures) : textures(textures) {}
        bool operator()(int i, int j) { return (textures[i].first.cols*textures[i].first.rows)<(textures[j].first.cols*textures[j].first.rows); }
    };

    ///texturing end


    friend class ClassifierFactory<VertexT, NormalT>;



    set<EdgePtr>        m_garbageEdges;
    set<HFace*>         m_garbageFaces;
    set<RegionPtr>      m_garbageRegions;

};

} // namespace lvr


#include "HalfEdgeKinFuMesh.tcc"

#endif /* HalfEdgeKinFuMesh_H_ */
