/*
 * modified stiff_system.cpp from boost libraries
 * https://www.boost.org/doc/libs/1_72_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/tutorial/stiff_systems.html
 */

#include <iostream>
#include <fstream>
#include <utility>

#include <boost/numeric/odeint.hpp>
#include <cstdio>

#include "definitions.hpp"
#include "obs.hpp"
#include "system.hpp"

#ifdef DO_VARSTAT
  #include "varstat.hpp"
#endif

using namespace boost::numeric::odeint;

/*
 * Looking at the 'Oregonator', a cyclic chemical reaction
 *
 *  A + Y -> X + P
 *  X + Y -> 2P
 *  A + X -> 2X + 2Z
 *  X + X -> A + P
 *  B + Z -> (0.5)f Y
 *
 * species A, B are fixed, with X,Y,Z varying
 *
 * the system is modelled by the ODE x = x[N],
 *
 *   dx 
 *  ---- = f(x, t)
 *   dt
 *
 *   where f(x, t) is the rate of change, and must 
 *   be provided so that the integrator can solve for x
 *
 *   The Jacobian
 *
 *                  df
 *   J = J(f, x) = ----
 *                  dx
 *
 *   provides for implicit solutions
 *
 *   The convention here is:
 *
 *    x[0] == X
 *    x[1] == Y
 *    x[2] == Z
 *
 * For more information, see https://en.wikipedia.org/wiki/Oregonator
*/
int main( int argc , char **argv )
{
    if (argc != 2)
    {
      std :: cerr << "pass FP value!\n";
      std :: exit(1);
    }

//    f = static_cast<Real>(std::atof(argv[1]));
    std::sscanf(argv[1], "%lf", &f);
    vector_type x( 3 );

    // initial conditions
    x[0] = 1.0E3;
    x[1] = 1.0E0;
    x[2] = 1.0E0;

//    Real _b = k3A - k2*x[1];
//    x[0] = (_b + std::sqrt((_b*_b) + 8. * k1A * k4 * x[1])) / (four*k4);

    Real t_init = 0.0;
    Real t_final = 1.0E5;
    Real t = t_init, dt = 1.0E-6;

    // setup the output file
    std::string outf="f";
    outf += std::to_string(f) + ".dat"; 

    gnuplot_obs<vector_type> obs(outf);

    // for stability analysis
    std::vector<Real> Xstore;

    // integration counters
    size_t n_step_attempt = 0;
    size_t n_step_fail = 0;
    size_t n_step_ok = 0;

    // integration stepper and system model
    auto stepper = make_controlled< rosenbrock4< Real > >(abs_tol, rel_tol, max_dt);
    auto system = std::make_pair( system_f(), system_J() );

    // step through time until done
    while(t < t_final)
    {
      n_step_attempt++;

      // try a step, if failed the x, t are unchanged
      // and dt is reduced.
      if(stepper.try_step( system, x, t, dt))
      {
        n_step_ok++;
      }
      else
      {
        n_step_fail++;
      }
      obs(x, t, dt);
    }

    // write any remaining state data
    obs.flush();

    // simple runtime information
    auto percent_val = [&](const auto& n){ return static_cast<Real>(n)/static_cast<Real>(n_step_attempt) * 100.0; };
    std::clog << "[RUN] fp precision: ";
    if constexpr(std::is_same_v<float, Real>)
      std::clog << "32\n";
    if constexpr(std::is_same_v<double, Real>)
      std::clog << "64\n";
    std::clog << "[RUN] num_ok: " << n_step_ok << "(" << percent_val(n_step_ok) << " %)\n";
    std::clog << "[RUN] num_failed: " << n_step_fail << "(" << percent_val(n_step_fail) << " %) \n";


#ifdef DO_VARSTAT
    {
      for (size_t i = 0; i < varstat::Tstore.size(); ++i)
      {
        auto [mean, stddev] = varstat::run_stats(varstat::Tstore[i]);
        std::clog << "[STAT: T" << i + 1 << "] mean = " << mean << " stdev = " << stddev << "\n";
      }
    }
#endif

    return 0;
}
