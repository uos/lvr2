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
 * Random.hpp
 *
 *  @date 14.07.2017
 *  @author Johan M. von Behren <johan@vonbehren.eu>
 */

#ifndef LVR2_UTIL_RANDOM_H_
#define LVR2_UTIL_RANDOM_H_

#include  <random>
#include  <iterator>

namespace lvr2
{

/**
 * Select a random element between start and end
 *
 * @tparam Iter Type for the start and end iterator
 * @tparam RandomGenerator Type for the random generator
 *
 * @param start Start of the range to pick random element between
 * @param end End of the range to pick random element between
 * @param g random generator for selecting process
 *
 * @return iterator at random position between start and end
 */
template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g);

/**
 * Select a random element between start and end using the std::mt19937 random generator
 *
 * @tparam Iter Iter Type for the start and end iterator
 *
 * @param start Start of the range to pick random element between
 * @param end End of the range to pick random element between
 *
 * @return iterator at random position between start and end
 */
template<typename Iter>
Iter select_randomly(Iter start, Iter end);

} // namespace lvr2

#include <lvr2/util/Random.tcc>

#endif // LVR2_UTIL_RANDOM_H_
