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
 * @file       BaseIO.hpp
 * @brief      Base interface for all I/O related classes.
 * @details    This file introduces a pure virtual class specifying some basic
 *             methods which must be implemented by all I/O classes in the lvr
 *             toolkit.
 * @author     Thomas Wiemann (twiemann), twiemann@uos.de
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version    110929
 * @date       Created:       2011-08-03
 * @date       Last modified: 2011-09-29 20:40:14
 */

#ifndef BASEIO_HPP_
#define BASEIO_HPP_

#include <string>
#include <map>

#include "lvr2/io/Model.hpp"

namespace lvr2
{


/**
 * @brief Interface specification for low-level io. All read
 *        elements are stored in linear arrays.
 */
class BaseIO
{
    public:
        BaseIO() {}
        virtual ~BaseIO() {};

        /**
         * \brief Parse the given file and load supported elements.
         *
         * @param filename  The file to read.
         */
        virtual ModelPtr read(std::string filename ) = 0;


        /**
         * \brief Save the loaded elements to the given file.
         *
         * @param filename Filename of the file to write.
         */
        virtual void save(std::string filename) = 0;


        /**
         * \brief Set the model and save the loaded elements to the given
         *        file.
         *
         * @param filename Filename of the file to write.
         */
        virtual void save(ModelPtr model, std::string filename);



        /**
         * \brief  Set the model for io operations to use.
         * \param m  Shared pointer to model.
         **/
        virtual void setModel(ModelPtr m);


        /**
         * \brief  Get the model for io operations.
         * \return  Shared pointer to model.
         **/
        virtual ModelPtr getModel();


    protected:
        ModelPtr m_model;

};

} // namespace lvr2


#endif /* BASEIO_HPP_ */
