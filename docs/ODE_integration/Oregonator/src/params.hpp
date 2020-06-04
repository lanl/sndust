#pragma once

#include <utility>

#include "definitions.hpp"

// constant numbers
constexpr Real two = 2.0;
constexpr Real four = 4.0;
constexpr Real half = 0.5;

constexpr Real k1A = 0.2; 
constexpr Real k2 = 2.0E9;
constexpr Real k3A = 1.0E3;
constexpr Real k4 = 5.0E7;
constexpr Real k5 = 1.0;

//constexpr Real f = 10.538;
Real f;

// this model is senstive to these tolerances
constexpr Real abs_tol = 1.0E-8; //solution absolute tolerance (when y ~ 0)
constexpr Real rel_tol = 1.0E-8; //solution relative tolerance

// limit timesteps to get usable data
constexpr Real max_dt = 1.0E0;

