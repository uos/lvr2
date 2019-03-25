//
// Created by patrick on 10.02.19.
//

#ifndef LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
#define LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/config/BaseOption.hpp>

namespace lvr2{

    template <typename BaseVecT, typename NormalT>
    class GrowingCellStructure {
    public:

        /**
         * Construct a GCS instance
         * @param surface pointsetsurface to get pointcloud information from
         */
        GrowingCellStructure(PointsetSurfacePtr<BaseVecT> surface);

        /**
         * Public method of the Reconstruction Class, calling all the other methods, generating the mesh
         * approximating the pointcloud's surface
         * @param mesh pointer to the mesh
         */
        void getMesh(HalfEdgeMesh<BaseVecT> &mesh);


        //TODO: add functions to make gcs possible, such as laplacian smoothing

        /**
         *
         * GETTER AND SETTER
         *
         */

        int getRuntime() const {
            return m_runtime;
        }

        int getBasicSteps() const {
            return m_basicSteps;
        }

        int getNumSplits() const {
            return m_numSplits;
        }

        float getBoxFactor() const {
            return m_boxFactor;
        }

        bool isWithCollapse() const {
            return m_withCollapse;
        }

        float getLearningRate() const {
            return m_learningRate;
        }

        float getNeighborLearningRate() const {
            return m_neighborLearningRate;
        }

        float getDecreaseFactor() const {
            return m_decreaseFactor;
        }

        int getAllowMiss() const {
            return m_allowMiss;
        }

        float getCollapseThreshold() const {
            return m_collapseThreshold;
        }

        bool isFilterChain() const {
            return m_filterChain;
        }

        int getDeleteLongEdgesFactor() const {
            return m_deleteLongEdgesFactor;
        }

        void setRuntime(int m_runtime) {
            GrowingCellStructure::m_runtime = m_runtime;
        }

        void setBasicSteps(int m_basicSteps) {
            GrowingCellStructure::m_basicSteps = m_basicSteps;
        }

        void setNumSplits(int m_numSplits) {
            GrowingCellStructure::m_numSplits = m_numSplits;
        }

        void setBoxFactor(float m_boxFactor) {
            GrowingCellStructure::m_boxFactor = m_boxFactor;
        }

        void setWithCollapse(bool m_withCollapse) {
            GrowingCellStructure::m_withCollapse = m_withCollapse;
        }

        void setLearningRate(float m_learningRate) {
            GrowingCellStructure::m_learningRate = m_learningRate;
        }

        void setNeighborLearningRate(float m_neighborLearningRate) {
            GrowingCellStructure::m_neighborLearningRate = m_neighborLearningRate;
        }

        void setDecreaseFactor(float m_decreaseFactor) {
            GrowingCellStructure::m_decreaseFactor = m_decreaseFactor;
        }

        void setAllowMiss(int m_allowMiss) {
            GrowingCellStructure::m_allowMiss = m_allowMiss;
        }

        void setCollapseThreshold(float m_collapseThreshold) {
            GrowingCellStructure::m_collapseThreshold = m_collapseThreshold;
        }

        void setFilterChain(bool m_filterChain) {
            GrowingCellStructure::m_filterChain = m_filterChain;
        }

        void setDeleteLongEdgesFactor(int m_deleteLongEdgesFactor) {
            GrowingCellStructure::m_deleteLongEdgesFactor = m_deleteLongEdgesFactor;
        }

    private:
        PointsetSurfacePtr<BaseVecT> m_surface; //helper-surface
        HalfEdgeMesh<BaseVecT> *m_mesh;

        int m_runtime; //how many steps?
        int m_basicSteps; //how many steps until collapse?
        int m_numSplits; //how many splits
        float m_boxFactor; //for initial mesh
        bool m_withCollapse; //should we collapse?
        float m_learningRate; //learning rate of the algorithm
        float m_neighborLearningRate;
        float m_decreaseFactor; //for sc calc
        int m_allowMiss;
        float m_collapseThreshold; //threshold for the collapse - when does it make sense
        bool m_filterChain; //should a filter chain be applied?
        int m_deleteLongEdgesFactor;

        /**
         * Getting the initial Polyhedron Mesh and placing it in the center of the pointcloud (Tetrahedron)
         * @param mesh
         */
        //void getInitialMesh(HalfEdgeMesh<BaseVecT> &mesh);

        //execute basic step
        void executeBasicStep();

        //split vertex with .. -- get handle in function or pass it?
        void executeVertexSplit();

        //collapse edge with ... -- get handle in function or pass it?
        void executeEdgeCollapse();

        /**
         * Getting the initial Tetrahedron, which will be used to approx the surface
         * @param mesh: pointer to a mesh
         * @return nothing
         */
        void getInitialMesh();

        void initTestMesh();

        void performLaplacianSmoothing(VertexHandle vertex);

    };
}


#include <lvr2/reconstruction/gs2/GrowingCellStructure.tcc>

#endif //LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
