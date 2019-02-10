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
        GrowingCellStructure(PointsetSurfacePtr<BaseVecT> surface){
            m_surface = surface;
            //std::cout << "Center:" << m_surface.get()->getBoundingBox().getCentroid() << std::endl;
        }


        //execute basic step
        void executeBasicStep();

        //split vertex with .. -- get handle in function or pass it?
        void executeVertexSplit(HalfEdgeHandle handle);

        //collapse edge with ... -- get handle in function or pass it?
        void executeEdgeCollapse(VertexHandle handle);

        // TODO: add gcs construction with calls to above three functions
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
        PointsetSurfacePtr<BaseVecT> m_surface;

        int m_runtime;
        int m_basicSteps;
        int m_numSplits;
        float m_boxFactor;
        bool m_withCollapse;
        float m_learningRate;
        float m_neighborLearningRate;
        float m_decreaseFactor;
        int m_allowMiss;
        float m_collapseThreshold;
        bool m_filterChain;
        int m_deleteLongEdgesFactor;
    };
}




#endif //LAS_VEGAS_GROWINGCELLSTRUCTURE_HPP
