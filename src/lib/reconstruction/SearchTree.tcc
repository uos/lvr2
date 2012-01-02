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
*  SearchTree.tcc
*
*       Created on: 02.01.2012
*           Author: Florian Otte
 */

namespace lssr {

   template<typename VertexT, typename NormalT>
      void SearchTree< VertexT, NormalT >::setKn( int kn ) {
         m_kn = kn;
      }


   template<typename VertexT, typename NormalT>
      void SearchTree< VertexT, NormalT >::setKi( int ki ) {
         m_ki = ki;
      }


   template<typename VertexT, typename NormalT>
      void SearchTree< VertexT, NormalT >::setKd( int kd ) {
         m_kd = kd;
      }

   template<typename VertexT, typename NormalT>
      int SearchTree< VertexT, NormalT >::getKn() {
         return m_kn;
      }


   template<typename VertexT, typename NormalT>
      int SearchTree< VertexT, NormalT >::getKi() {
         return m_ki;
      }

   template<typename VertexT, typename NormalT>
      int SearchTree< VertexT, NormalT >::getKd() {
         return m_kd;
      }
} // namespace
