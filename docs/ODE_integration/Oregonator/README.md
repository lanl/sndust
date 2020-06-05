# Oregonator simulation

This application runs a generalized Oregonator, an equilibrium kinetics problem with stable oscillations. It is meant to demonstrate the necessity of accuracy of rate evaluation and integrator method.

# Prereq

- C++17 compiler
- CMake 3.15 or later
- Boost 1.68 or later (only requires headers)

Optional requirements to use bundled analysis scripts:

- Python3+ with numpy, scipy

# Building

Use the canonical cmake procedure. There are no project-defined CMake options.

```{engine=sh}
$> mkdir build && cd build
$> cmake -DCMAKE_BUILD_TYPE=Release ..
$> cmake --build .
```

The build will result in a single executable, `oregonator_double`.

# Running

The binary takes one argument. This argument is the adjustible parameter `f`.

## Parameters of the model

All parameters in `src/params.hpp` can effect result. However, we only vary the parameter `f`, defined in `src/params.hpp`.

To particular values of interest are:

- `f = 1.9352`
- `f = 0.500315`

```{engine=sh}
$> oregonator_double 1.945
```

## Output

Run statistics (number of integration steps failed/succeed) are printed on `stderr`.

Simulation data is produced data file `f<f_value>.dat`

To check for stability, run 
```{engine=sh}
python check_stable.py -i f1.945000.dat -t 1.0E-8
python check_stable.py -i f0.500315.dat -t 1.0E-8
```

The `-t` parameter is the tolerance; `1.0E-8` is taken from the tolerance of the integration, but you may vary this value if you wish.
