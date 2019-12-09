#pragma once


#ifndef LAS_VEGAS_SCANIO_HPP
#define LAS_VEGAS_SCANIO_HPP

#include "lvr2/types/Scan.hpp"

namespace lvr2 {

    namespace hdf5features {

        template<typename Derived>
        class ScanIO {
        public:
            void save(std::string name, const ScanPtr& buffer);
            void save(HighFive::Group& group, const ScanPtr& buffer);

            ScanPtr load(std::string name);
            ScanPtr load(HighFive::Group& group);
            ScanPtr loadScan(std::string name);

        protected:

            bool isScan(HighFive::Group& group);

            Derived* m_file_access = static_cast<Derived*>(this);
            // dependencies
            VariantChannelIO<Derived>* m_vchannel_io = static_cast<VariantChannelIO<Derived>*>(m_file_access);

            static constexpr const char* ID = "ScanIO";
            static constexpr const char* OBJID = "Scan";
        };


    } // hdf5features

} // namespace lvr2

#include "ScanIO.tcc"

#endif //LAS_VEGAS_SCANIO_HPP
