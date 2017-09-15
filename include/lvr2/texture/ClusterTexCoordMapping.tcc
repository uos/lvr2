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
* Texturizer.tcc
*
*  @date 23.07.2017
*  @author Jan Philipp Vogtherr <jvogtherr@uni-osnabrueck.de>
*  @author Kristin Schmidt <krschmidt@uni-osnabrueck.de>
*/


namespace lvr2
{

void ClusterTexCoordMapping::push(ClusterHandle handle, TexCoords tex)
{
    if (m_len == m_mapping.size())
    {
        cout << "Error: Overflow in ClusterTexCoordMapping" << endl;
    }
    else
    {
        m_mapping[m_len] = make_pair(handle, tex);
        m_len++;
    }
}

TexCoords ClusterTexCoordMapping::getTexCoords(ClusterHandle clusterH) const
{
    for (size_t i = 0; i < m_len; i++)
    {
        if (m_mapping[i]->first == clusterH)
        {
            return m_mapping[i]->second;
        }
    }
}




} // namespace lvr2
