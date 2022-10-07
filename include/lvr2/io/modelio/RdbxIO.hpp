//
// Created by Nikolas on 13.09.22.
//

#ifndef LAS_VEGAS_RDBXIO_HPP
#define LAS_VEGAS_RDBXIO_HPP

#include "lvr2/io/modelio/ModelIOBase.hpp"
#include <iostream>

#include <riegl/rdb.hpp>
#include <riegl/rdb/default.hpp>
//#include <riegl/rdb/default/attributes.hpp>

namespace lvr2
{

    class RdbxIO : public ModelIOBase
    {
    public:
        RdbxIO();

        explicit RdbxIO(const std::vector<std::string>& attributes);

        ~RdbxIO();

        /**
         * @brief Save the loaded elements to a rdbx file
         * TODO:
         *
         * @param filename
         */
        void save(std::string filename);

        /**
         * @brief Set the model and save the loaded elements to a rdbx file
         * TODO:
         *
         * @param model
         * @param filename
         */
        void save(ModelPtr model, std::string filename);

        /**
         * @brief Parse the rdbx file and load the supported elements
         * TODO:
         *
         * @param filename
         */
        ModelPtr read(std::string filename);

        /**
         * @brief Maybe not needed
         * TODO:
         *
         * @param filename
         * @param n
         * @param reduction
         * @return
         */
        ModelPtr read(std::string filename, int n, int reduction);

    private:
        std::vector<uint64_t> m_id;
        std::vector<std::array<double, 3>> m_coordinates;
        std::vector<float> m_amplitude;
        std::vector<float> m_reflectance;
        std::vector<std::array<uint8_t, 4>> m_rgb;
        std::vector<uint16_t> m_class;
        std::unordered_set<std::string> m_attributes;           //String Vector of wanted Attributes, with the attributes Named as in the rdblib given by RIEGL.
    };

} // lvr2

#endif //LAS_VEGAS_RDBXIO_HPP
