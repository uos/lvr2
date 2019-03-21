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
 *  Panic.hpp
 *
 */

#ifndef LVR2_UTIL_PANIC_H_
#define LVR2_UTIL_PANIC_H_

#include <exception>
#include <iostream>
#include <string>

namespace lvr2
{

/**
 * @brief An exception denoting an internal bug.
 */
struct PanicException : public std::exception
{
    PanicException(std::string msg) : m_msg(msg) {}

    virtual const char* what() const noexcept
    {
        return m_msg.c_str();
    }

private:
    std::string m_msg;
};


struct VertexLoopException : public std::exception
{
    VertexLoopException(std::string msg) : m_msg(msg) {}

    virtual const char* what() const noexcept
    {
        return m_msg.c_str();
    }
private:
    std::string m_msg;
};

/**
 * @brief Throws a panic exception with the given error message.
 */
inline void panic(std::string msg)
{
    throw PanicException("Program panicked: " + msg);
}

/**
 * @brief Throws a panic exception with the given error message and denotes
 *        that the exception was thrown due to a missing implementation.
 */
inline void panic_unimplemented(std::string msg)
{
    throw PanicException("Program panicked due to missing implementation: " + msg);
}


} // namespace lvr2


#endif // LVR2_UTIL_PANIC_H_
