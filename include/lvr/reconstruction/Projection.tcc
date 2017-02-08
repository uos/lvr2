namespace lvr
{

float Projection::toPolar(const float _cartesian[], float polar[])
{
    float phi, theta, rho;

    float x_sqr, y_sqr, z_sqr;
    x_sqr = _cartesian[0] * _cartesian[0];
    y_sqr = _cartesian[1] * _cartesian[1];
    z_sqr = _cartesian[2] * _cartesian[2];

    rho = std::sqrt(x_sqr + y_sqr + z_sqr);

    float cartesian[3];
    cartesian[0] = _cartesian[0] / rho;
    cartesian[1] = _cartesian[1] / rho;
    cartesian[2] = _cartesian[2] / rho;

    phi = std::acos(cartesian[2]);

    float theta0;

    if(std::abs(phi) < 0.0001)
    {
        theta = 0.0;
    }
    else if(std::abs(M_PI - phi) < 0.0001)
    {
        theta = 0.0;
    }
    else
    {
        if(std::abs(cartesian[0]/std::sin(phi)) > 1.0)
        {
            if(cartesian[0]/std::sin(phi) < 0)
            {
                theta0 = M_PI;
            }
            else
            {
                theta0 = 0.0;
            }
        }
        else
        {
            theta0 = acos(cartesian[0]/std::sin(phi));

        }

        float sintheta = cartesian[1]/std::sin(phi);
        float EPS = 0.0001;

        if(std::abs(std::sin(theta0) - sintheta) < EPS)
        {
            theta = theta0;
        }
        else if(std::abs( std::sin( 2*M_PI - theta0 ) - sintheta ) < EPS)
        {
            theta = 2*M_PI - theta0;
        }
        else
        {
            theta = 0;
        }
    }
    polar[0] = phi;
    polar[1] = theta;
    polar[2] = rho;
}

}
