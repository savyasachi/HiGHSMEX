
# HiGHSMEX

MATLAB mex interface to the [HiGHS optimization library.](https://github.com/ERGO-Code/HiGHS)\
*HiGHSMEX is not a part of the official HiGHS distribution.*\
\
HiGHSMEX is built with HiGHS v1.15.1.
## Pre-compiled mex file

For 64-bit Windows, Linux and macOS users the pre-compiled mex files *highsmex.mexw64*, *highsmex.mexa64* and *highsmex.mexmaca64* are provided so you do not have to install HiGHS on your system or, compile the mex file. \
Thanks to [Ray Zimmerman](https://github.com/rdzman) for providing help with the compilation of the mex file on macOS platform. \
Thanks to [GenosseFlosse](https://github.com/GenosseFlosse) for providing help with the compilation of the mex file on Linux platform.
## Instructions for compiling from source

1. Download or clone the HiGHS code. The *highsmex.cpp* file needs to be compiled with C++20 switch, hence it would be advisable to compile HiGHS with C++20 switch. To do so modify the CMakeLists.txt of HiGHS by changing the line 29 ```set(CMAKE_CXX_STANDARD 11)``` to ```set(CMAKE_CXX_STANDARD 20)``` and line 301 ```set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")``` to ```set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")```. Install HiGHS as described [here.](https://github.com/ERGO-Code/HiGHS/tree/master/cmake) This should create a static library named *highs* e.g., *highs.lib* on Windows.\
**Note for macOS users** Your cmake command should look like the following \
`cmake -S <path-to-source> -B <path-to-build> -DBUILD_SHARED_LIBS=OFF -DZLIB=OFF -DCMAKE_OSX_DEPLOYMENT_TARGET=13.4`
2. Open script named *make_highsmex.m* in MATLAB and specify the inputs. Then execute the script. This builds both mex files of the project, named highsmex.mex* and highsmex_iis.mex* e.g., highsmex.mexw64 and highsmex_iis.mexw64 on 64-bit Windows. The two are independent of each other, so a single one can be built by naming its source file, e.g. ```make_highsmex('1.15.1', 'highsmex.cpp')```.\
The code shared by the two, i.e. the MATLAB type helpers, the conversion of the HiGHS options and info structs, the matrix conversion, and the mex logging and error handling, lives in the header *highsmex_common.hpp*. It is header-only, so no extra source file has to be passed to *mex*.


## Documentation

HiGHS documentation is available [here.](https://ergo-code.github.io/HiGHS/stable/) A MATLAB function named *callhighs* is the interface to the HiGHS library. Run ```help callhighs``` on MATLAB command prompt to see the help on the input and output parameters of the callhighs function. Also, see the MATLAB script *example_callhighs.m* for various examples of usage.

**Caution:** Do not call the *highsmex* function directly, and only use the *callhighs* function.

### Irreducible infeasible subsystem (IIS)

A second MATLAB function named *callhighs_iis* is the interface to the [IIS facility](https://ergo-code.github.io/HiGHS/stable/guide/advanced/) of HiGHS, which explains *why* a model is infeasible. It takes the same leading input arguments as *callhighs* (c, A, L, U, l, u), so an infeasible model can be handed over unchanged, and it returns the rows and columns that form the IIS, the bounds of those rows and columns that are in conflict, and the IIS itself as a standalone model that can be passed straight back to *callhighs*. Run ```help callhighs_iis``` on the MATLAB command prompt, and see the MATLAB script *example_callhighs_iis.m* for examples of usage.

Unlike HiGHS, whose ```iis_strategy``` default performs only the cheap trivial checks, *callhighs_iis* defaults to a full IIS calculation (```iis_strategy = 6```). This requires the solution of multiple LPs and is therefore considerably more expensive than a single solve; use the ```iis_time_limit``` option to bound the effort.

The HiGHS IIS facility is available for **linear** programs only, hence *callhighs_iis* accepts neither an integrality vector nor a Hessian.

**Caution:** Do not call the *highsmex_iis* function directly, and only use the *callhighs_iis* function.

### Tests

Two test functions are provided, each returning the number of failed checks.

+ ```test_callhighs``` checks the interface of *callhighs* and then calls ```validatehighsmex```, which solves a set of LP, MILP and QP models and verifies each solution directly against the model it came from: the variable bounds, the rows and the integrality are checked, and the objective is recomputed from the returned solution.
+ ```test_callhighs_iis``` checks the interface of *callhighs_iis* and verifies every IIS against the definition of an IIS using ```validateiis```.

Both additionally cross-check against [Gurobi](https://www.gurobi.com/) when it is available. *validatehighsmex* compares the model status and the optimal objective value, and *test_callhighs_iis* compares against ```gurobi_iis```. Gurobi is located by ```findgurobi``` via the ```GUROBI_HOME``` environment variable, e.g. ```GUROBI_HOME=C:\gurobi1302\win64```, or from the MATLAB search path. Gurobi is entirely optional; without it the comparisons are skipped and the remaining checks still run.

Note that neither comparison requires the two solvers to return the same solution. An optimal point need not be unique, and neither need an IIS, so only quantities that are well defined are compared: the model status, the optimal objective value, and, for an IIS, whether the result actually satisfies the definition.

```matlab
assert(test_callhighs() == 0)
assert(test_callhighs_iis() == 0)
```

HiGHSMEX provides access to almost all the capabilities of HiGHS library except the following
+ Reading problem data from a model file. 
+ Setting names for the rows and columns of the model, or setting name for the objective.
+ Advanced features described [here](https://ergo-code.github.io/HiGHS/stable/guide/advanced/), apart from the IIS facility which is available through *callhighs_iis*.
+ HiGHS callbacks.

Newer releases of MATLAB from R2024 include HiGHS as the LP and MILP solver. The advantages of HiGHSMEX are
+ It can be used by MATLAB users with versions older than R2024.
+ It can be used by MATLAB users with no access to MATLAB's Optimization Toolbox.
+ It is quick and easy to integrate the new releases of HiGHS as they become available.
+ Most of the features of HiGHS are available including QP, multi-objective LP, and, hot-starting.
+ Provides access to all the options of HiGHS.
## License
HiGHSMEX is covered by the [MIT](https://choosealicense.com/licenses/mit/) license.

