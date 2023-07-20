#ifndef HPC_HELPERS_HPP
#define HPC_HELPERS_HPP

#include <iostream>
#include <cstdint>

    #include <chrono>


    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();

    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << "s" << std::endl;



    // transfer constants
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

#endif