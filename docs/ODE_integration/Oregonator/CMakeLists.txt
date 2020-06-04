####
# precision experiment
####

project(oregonator LANGUAGES CXX)
cmake_minimum_required(VERSION 3.15)
###
# options
###

###
# dependencies
###
find_package(Boost REQUIRED)

###
# executable target
###
function(gen_exec fp_type)
  set(BIN_NAME "oregonator_${fp_type}")
  add_executable(${BIN_NAME})

  target_sources(${BIN_NAME}
    PUBLIC
      "src/main.cpp"
      "src/definitions.hpp"
      "src/params.hpp"
      "src/system.hpp"
      "src/obs.hpp"
  )

  target_compile_options(${BIN_NAME}
    PUBLIC
    "-Wall"
  )
  target_compile_features(${BIN_NAME}
    PUBLIC
      cxx_std_17
  )

  target_compile_definitions(${BIN_NAME}
    PUBLIC
      "DEFAULT_PREC=${fp_type}"
  )

  target_link_libraries(${BIN_NAME}
    PUBLIC
      "Boost::boost"
  )
endfunction()

gen_exec("double")

configure_file(${CMAKE_SOURCE_DIR}/check_stable.py ${CMAKE_BINARY_DIR}/check_stable.py)



