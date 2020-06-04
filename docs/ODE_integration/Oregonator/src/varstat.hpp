#pragma once 

#include <algorithm>
#include <cmath>

#include "definitions.hpp"

namespace varstat
{
  std::array<std::vector<Real>, 5> Tstore;

  inline auto run_stats(const std::vector<Real>& v)
  {
    Real mean, stddev;
    mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<Real>(v.size());
    stddev = std::sqrt(std::accumulate(v.begin(), v.end(), 0.0, [&](auto s, auto x){ return std::move(s) + std::pow(x - mean, 2.0); } ) / static_cast<Real>(v.size()));
 
    return std::make_pair(mean, stddev);
  }

}
