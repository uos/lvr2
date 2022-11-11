#include "lvr2/algorithm/UtilAlgorithms.hpp"

namespace lvr2
{

void rasterize_line(Eigen::Vector2f p0, Eigen::Vector2f p1, std::function<void(Eigen::Vector2i)> cb)
{
    using Eigen::Vector2i;
    float x_dist = std::abs( p1.x() - p0.x());
    float y_dist = std::abs( p1.y() - p0.y());
    // Plot along x
    if (x_dist > y_dist)
    {
        Eigen::Vector2f left, right;
        left = p0.x() < p1.x() ? p0 : p1;
        right = p0.x() < p1.x() ? p1 : p0;

        // Check for div by 0
        if (x_dist == 0.0f) 
        {
            cb(left.cast<int>());
            return;
        }

        float slope = ((float) right.y() - left.y()) / ((float) right.x() - left.x());
        // Left point
        cb(left.cast<int>());
        // y at the right edge of the pixel containing the left point (use floor(x + 1) in case x is a whole number)
        float y_0 = left.y() + slope * (std::floor(left.x() + 1.0f) - left.x());
        // Check if y_0 is between floor(left.y()) and ceil(left.y())
        if (((int) left.y()) != ((int) y_0))
        {
            cb(Vector2i(left.x(), y_0));
        }

        // Right point
        cb(right.cast<int>());
        // y at the left edge of the pixel containing the right point
        float y_1 = right.y() - slope * (right.x() - std::floor(right.x()));
        // Check if y_1 is between floor(right.y()) and ceil(right.y())
        if (((int) right.y()) != ((int) y_1))
        {
            cb(Vector2i(right.x(), y_1));
        }

        // The line
        float y = y_0;
        for (int x = floor(left.x() + 1.0f); x < floor(right.x()); x++)
        {
            float next_y = y + slope;
            cb(Vector2i(x, y));

            if (((int) y) != ((int) next_y))
            {
                cb(Vector2i(x, next_y));
            }

            y = next_y;
        }
  
    }
    // Plot along y
    else
    {
        Eigen::Vector2f down, up;
        down = p0.y() < p1.y() ? p0 : p1;
        up = p0.y() < p1.y() ? p1 : p0;

        // Check for div by 0
        if (y_dist == 0.0f) 
        {
            cb(down.cast<int>());
            return;
        }

        float slope = ((float) up.x() - down.x()) / ((float) up.y() - down.y());

        // low point
        cb(down.cast<int>());
        // x at the upper edge of the pixel containing the lower point
        float x_0 = down.x() + slope * (std::floor(down.y() + 1) - down.y());
        // Check if x_0 is between floor(down.x()) and ceil(down.x())
        if (((int) down.x()) != ((int) x_0))
        {
            cb(Vector2i(x_0, down.y()));
        }

        // high point
        cb(up.cast<int>());
        // x at the lower edge of the pixel containing the high point
        float x_1 = up.x() - slope * (up.y() - std::floor(up.y()));
        // Check if x_1 is between floor(up.x()) and ceil(up.x())
        if (((int) up.x()) != ((int) x_1))
        {
            cb(Vector2i(x_1, up.y()));
        }
        
        // The line
        float x = x_0;
        for (int y = floor(down.y() + 1.0f); y < floor(up.y()); y++)
        {
            float next_x = x + slope;
            cb(Vector2i(x, y));

            if (((int) x) != ((int) next_x))
            {
                cb(Vector2i(next_x, y));
            }

            x = next_x;
        }
    }
}

} // namespace lvr2