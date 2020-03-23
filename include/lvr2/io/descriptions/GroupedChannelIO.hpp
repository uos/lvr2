/**
 * Copyright (c) 2019, University Osnabrück
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

#ifndef GROUPEDCHANNELIO_HPP
#define GROUPEDCHANNELIO_HPP

#include <string>
#include "lvr2/types/BaseBuffer.hpp"

namespace lvr2
{

/**
 * @brief   Interface class for io classes that support annotation channels
 *          organized in different groups
 * 
 */
class GroupedChannelIO
{
public:
    /**
   * @brief getChannel  Reads a float attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the float channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel) = 0;

  /**
   * @brief getChannel  Reads an index attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the index channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel) = 0;

  /**
   * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the unsigned char channel
   * @return            true if the channel has been loaded successfully, false otherwise
   */
  virtual bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel) = 0;


  /**
   * @brief addChannel  Writes a float attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the float channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const FloatChannel& channel) = 0;

  /**
   * @brief addChannel  Writes an index attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the index channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const IndexChannel& channel) = 0;

  /**
   * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
   * @param group       The associated attribute group
   * @param name        The associated attribute name
   * @param channel     The pointer to the unsigned char channel which should be written
   * @return            true if the channel has been written successfully, false otherwise
   */
  virtual bool addChannel(const std::string group, const std::string name, const UCharChannel& channel) = 0;
};

} // namespace lvr2

#endif