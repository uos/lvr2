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
