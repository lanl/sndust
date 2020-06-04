#pragma once

#include "definitions.hpp"
#include "params.hpp"

#ifdef DO_VARSTAT
  #include "varstat.hpp"
#endif

struct system_f
{
  inline void operator()( const vector_type &x , vector_type &dxdt , Real /* t */ )
  {
    // using explicit array ordering
    // may be used to reorder formula for rate
    Real T[] = {  
      [0] = k1A * x[1],
      [1] = k2 * x[0] * x[1],
      [2] = k3A * x[0],
      [3] = two * k4 * x[0] * x[0],
      [4] = f * k5 * x[2]
    };

    // summation of forward and backward rates  
    // dxdt[ 0 ] = k1 * A * x[1] - k2 * x[0] * x[1] + k3 * A * x[0] - 2.0 * k4 * x[0] * x[0];
    // dxdt[ 1 ] = -k1 * A * x[1] - k2 * x[0] * x[1] + 0.5 * f * kc * B * x[2];
    //dxdt[ 0 ] = T[0] - T[1] + T[2] - T[3];
    //dxdt[ 1 ] = -T[0] - T[1] + T[4];
    dxdt[0] = (T[0]+((-T[1]+-T[3])+T[2])); dxdt[1] = (-T[0]+(-T[1]+T[4]));
    dxdt[ 2 ] = k3A * x[0] - k5 * x[2];
      
#ifdef DO_VARSTAT
    {
    for(size_t i = 0; i < varstat::Tstore.size(); ++i)
      varstat::Tstore[i].push_back(T[i]);
    }
#endif
  }

 };

struct system_J
{
    inline void operator()( const vector_type &  x , matrix_type &J , const Real & /* t */ , vector_type &dfdt ) const
    {
      // differentiation coeffiecents
      // fill in Jacobian
      J( 0, 0 ) = -k2 * x[1] + k3A - four * k4 * x[0];
      J( 0, 1 ) = k1A - k2 * x[0];
      J( 0, 2 ) = 0.0;
      J( 1, 0 ) = -k2 * x[1];
      J( 1, 1 ) = -k1A - k2 * x[0];
      J( 1, 2 ) = f * k5;
      J( 2, 0 ) = k3A;
      J( 2, 1 ) = 0.0;
      J( 2, 2 ) = -k5;

      // the rates have no explicit dependence on time
      dfdt[0] = 0.0;
      dfdt[1] = 0.0;
      dfdt[2] = 0.0;
    }
};


