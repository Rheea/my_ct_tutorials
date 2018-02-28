cmake_minimum_required(VERSION 2.8.3)

#include(${CMAKE_CURRENT_SOURCE_DIR}/../ct/cmake/compilerSettings.cmake)
#include(/home/aradulescu/catkin_mockup/src/my_ct_project/ct/cmake/explicitTemplateHelpers.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/../ct/cmake/clang-cxx-dev-tools.cmake)


# reads the templates from file
function(ct_getDimensionsFromLine LineContent)
    string(REGEX REPLACE "," ";" LineContent ${LineContent})

    foreach(NameAndValue ${LineContent})    
      # Strip leading spaces
      string(REGEX REPLACE "^[ ]+" "" NameAndValue ${NameAndValue})
      # Find variable name
      string(REGEX MATCH "^[^=]+" Name ${NameAndValue})
      # Find the value
      string(REPLACE "${Name}=" "" Value ${NameAndValue})
      
      # Set the variable
      if(Name STREQUAL "STATE_DIM")
          set(STATE_DIM_PRESPEC "${Value}" PARENT_SCOPE)
      endif()
      
        # Set the variable
      if(Name STREQUAL "CONTROL_DIM")
          set(CONTROL_DIM_PRESPEC "${Value}" PARENT_SCOPE)
      endif()
      
      if(Name STREQUAL "SCALAR")
          set(SCALAR_PRESPEC "${Value}" PARENT_SCOPE)
      endif()
      
      if(Name STREQUAL "POS_DIM")
          set(POS_DIM_PRESPEC "${Value}" PARENT_SCOPE)
      endif()
      
      if(Name STREQUAL "VEL_DIM")
          set(VEL_DIM_PRESPEC "${Value}" PARENT_SCOPE)
      endif()
    endforeach()

endfunction()


# reads explicit templates from config file and gathers sources and libs
function(ct_configure_explicit_templates ConfigFile ConfigDir LibPrefix)

  FILE(READ "${ConfigFile}" contents)

  STRING(REGEX REPLACE ";" "\\\\;" contents "${contents}")
  STRING(REGEX REPLACE "\n" ";" contents "${contents}")
  
  #message(WARNING "file content: ${contents}")
  
  foreach(line ${contents})
      #message(WARNING "extracting variables from line: ${line}")
      set(STATE_DIM_PRESPEC "")
      set(CONTROL_DIM_PRESPEC "")
      set(SCALAR_PRESPEC "")
      set(POS_DIM_PRESPEC "0")
      set(VEL_DIM_PRESPEC "0")
      set(CURRENT_SRCS "")
  
      ct_getDimensionsFromLine(${line})
      
      #message(WARNING "extracted: STATE_DIM=${STATE_DIM_PRESPEC}, CONTROL_DIM=${STATE_DIM_PRESPEC}, SCALAR=${SCALAR_PRESPEC}")
      
      string(REGEX REPLACE "[^0-9a-zA-Z]+" "" SCALAR_PRESPEC_CLEAN ${SCALAR_PRESPEC})
      set(CURRENT_LIB_NAME "${LibPrefix}-${STATE_DIM_PRESPEC}-${CONTROL_DIM_PRESPEC}-${SCALAR_PRESPEC_CLEAN}-${POS_DIM_PRESPEC}-${VEL_DIM_PRESPEC}")
      
      if(STATE_DIM_PRESPEC AND CONTROL_DIM_PRESPEC AND SCALAR_PRESPEC)
          #message(WARNING "Will configure now")
          ct_configureFiles(${ConfigDir} ${STATE_DIM_PRESPEC}, ${CONTROL_DIM_PRESPEC}, ${SCALAR_PRESPEC} ${POS_DIM_PRESPEC} ${VEL_DIM_PRESPEC})
      elseif()
          #message(WARNING "Nothing to configure")
      endif()
      
      if(CURRENT_SRCS)
          set(${CURRENT_LIB_NAME}_SRCS ${CURRENT_SRCS} PARENT_SCOPE)
          list(APPEND LIB_NAMES "${CURRENT_LIB_NAME}")
      endif()
  endforeach()
  
  set(PRESPEC_LIB_NAMES ${LIB_NAMES} PARENT_SCOPE)

endfunction()


# finds cpp.in files and configures them
function(ct_configureFiles ConfigDir STATE_DIM_PRESPEC, CONTROL_DIM_PRESPEC, SCALAR_PRESPEC, POS_DIM_PREPSEC, VEL_DIM_PRESPEC)
    set(CURRENT_SRCS "")
    file(GLOB_RECURSE files "${ConfigDir}*.in")
    #message(WARNING "files to configure in directory ${ConfigDir}:\n ${files}")
    foreach(file ${files})
        string(REGEX REPLACE "[^0-9a-zA-Z]+" "" SCALAR_PRESPEC_CLEAN ${SCALAR_PRESPEC})
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" outputFile ${file})
        string(REPLACE ".cpp.in" "" outputFile ${outputFile})
        set(outputFile "${outputFile}-${STATE_DIM_PRESPEC}-${CONTROL_DIM_PRESPEC}-${SCALAR_PRESPEC_CLEAN}-${POS_DIM_PRESPEC}-${VEL_DIM_PRESPEC}.cpp")
        #message(WARNING "configuring file \n ${file} to \n ${outputFile} ")
        set(DOUBLE_OR_FLOAT false)
        if(SCALAR_PRESPEC STREQUAL "double" OR SCALAR_PRESPEC STREQUAL "float")
            set(DOUBLE_OR_FLOAT true)
        endif()
        configure_file(${file} ${outputFile})
        list(APPEND CURRENT_SRCS ${outputFile})
    endforeach()
   
    #(WARNING "CURRENT_SRCS: ${CURRENT_SRCS}")
    set(CURRENT_SRCS "${CURRENT_SRCS}" PARENT_SCOPE)
endfunction()


# creates a target for each explicit template lib and adds its sources to it
function(ct_add_explicit_template_libs)
    foreach(lib_name ${PRESPEC_LIB_NAMES})
      #get_filename_component(raw_filename ${file} NAME_WE)
      #message(WARNING "sources for lib ${lib_name}: \n ${${lib_name}_SRCS}")
      add_library(${lib_name}
           ${${lib_name}_SRCS}
      )
      target_link_libraries(${lib_name} ${catkin_LIBRARIES} ${PYTHON_LIBRARY})
    endforeach()
endfunction()


# link external library (for example to link optcon against lapack)
function(ct_link_external_library extLibs)
foreach(lib_name ${PRESPEC_LIB_NAMES})
      target_link_libraries(${lib_name} "${extLibs}")
    endforeach()
endfunction()



project(my_ct_project)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread -fopenmp -std=c++11 -Wall")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -pthread -std=c++11 -Wall")
SET(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  ct_core
  ct_models
  ct_optcon
  ct_rbd
  roscpp
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )


#########################################################################################
#todo export this properly from ct_core
find_package(PythonLibs QUIET)
if (PYTHONLIBS_FOUND)
   message(STATUS "Python found")
   include_directories(${PYTHON_INCLUDE_DIRS})
   set(PLOTTING_ENABLED true)
   add_definitions(-DPLOTTING_ENABLED)
else()
   message(STATUS "Python not found")
   set(PYTHON_LIBRARY "")
endif()

option(HPIPM "Compile with HPIPM support" ON)

## lapack libs
set(LAPACK_LIBS "")
find_package(LAPACK QUIET)
if(LAPACK_FOUND)
    message(STATUS "Found LAPACK library")
    set(LAPACK_LIBS lapack)
    add_definitions (-DCT_USE_LAPACK)
    set(USE_LAPACK ON)
else(LAPACK_FOUND)
    message(WARNING "Could not find LAPACK library")
endif(LAPACK_FOUND)

## dummy HPIPM libs
set(HPIPM_LIBS "")


if(MATLAB)
    message(STATUS "MATLAB support ON")
    find_package(matlab_cpp_interface REQUIRED)
    if(matlab_cpp_interface_FOUND)
        include_directories(${matlab_cpp_interface_INCLUDE_DIRS})
        set(MATLAB_LIB_DIR ${matlab_cpp_interface_LIBRARIES})
        add_definitions( -DMATLAB )
    endif()
endif(MATLAB)
set(MATLAB_LIBS ${MATLAB_LIB_DIR})


## include BLASFEO AND HPIPM
if(HPIPM)
    message(STATUS "Switching HPIPM ON")
    if(DEFINED ENV{BLASFEO_DIR})
        if(DEFINED ENV{HPIPM_DIR})
            message(WARNING "...Found HPIPM and BLASFEO environment variables")

            set(BLASFEO_INCLUDE_DIR $ENV{BLASFEO_DIR}/include)
            set(HPIPM_INCLUDE_DIR $ENV{HPIPM_DIR}/include)
            set(BLASFEO_DIR $ENV{BLASFEO_DIR})
            set(HPIPM_DIR $ENV{HPIPM_DIR})

            include_directories(${BLASFEO_INCLUDE_DIR})
            include_directories(${HPIPM_INCLUDE_DIR})
            link_directories(${BLASFEO_DIR}/lib)
            link_directories(${HPIPM_DIR}/lib)

            find_library(BLASFEO_LIBRARY blasfeo  ${BLASFEO_DIR}/lib REQUIRED)
            find_library(HPIPM_LIBRARY hpipm  ${HPIPM_DIR}/lib REQUIRED)

            set(HPIPM_LIBS hpipm  blasfeo)

            add_definitions (-DHPIPM)

        else(DEFINED ENV{HPIPM_DIR})
            message(WARNING "WARNING: Trying to build with HPIPM, but no ENV variable for HPIPM found.")
        endif(DEFINED ENV{HPIPM_DIR})
    else(DEFINED ENV{BLASFEO_DIR})
        message(WARNING "WARNING: Trying to build with HPIPM, but no ENV variable for Blasfeo found.")
    endif(DEFINED ENV{BLASFEO_DIR})
endif(HPIPM)


#message(${BLASFEO_DIR})
#message(${HPIPM_DIR})
#message(${HPIPM_INCLUDE_DIR}
#        ${BLASFEO_INCLUDE_DIR})
#######################################################################################


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need


#used later for clang-tidy
set(INC_DIRS ${catkin_INCLUDE_DIRS}  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${HPIPM_INCLUDE_DIR} ${BLASFEO_INCLUDE_DIR})

catkin_package(
  INCLUDE_DIRS include ${HPIPM_INCLUDE_DIR} ${BLASFEO_INCLUDE_DIR}
  LIBRARIES my_ct_project ${HPIPM_LIBRARY} ${BLASFEO_LIBRARY}
  CATKIN_DEPENDS ct_core ct_models ct_optcon ct_rbd roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########
if(USE_LAPACK)
    ct_link_external_library(${LAPACK_LIBS})
endif(USE_LAPACK)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${ct_model_libs}
  ${LAPACK_LIBS}

)


########################################################################################

# add libraries for explicit template
ct_add_explicit_template_libs()

if(HPIPM)
   ct_link_external_library(${BLASFEO_LIBRARY})
   ct_link_external_library(${HPIPM_LIBRARY})
endif(HPIPM)

#########################################################################################

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/my_ct_project.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated bpkg_check_modules(libcmaes REQUIRED libcmaes)efore libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
#add_executable(${PROJECT_NAME}_node src/my_ct_project_node.cpp)
add_executable(${PROJECT_NAME}_node src/my_ct_project_main.cpp)
add_executable(${PROJECT_NAME}_controller_node src/my_ct_controller_main.cpp)
add_executable(${PROJECT_NAME}_hyqtest_node src/my_ct_project_HyQ_test.cpp)
add_executable(${PROJECT_NAME}_hyatest_node src/my_ct_project_HyA_test.cpp)

add_executable(${PROJECT_NAME}_hyqlqr_node src/LQR_hyq.cpp)


target_link_libraries(${PROJECT_NAME}_hyatest_node ${ct_model_libs} ${catkin_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_hyqtest_node ${ct_model_libs} ${catkin_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_hyqlqr_node  ${LAPACK_LIBS} ${ct_model_libs} ${catkin_LIBRARIES})


if(HPIPM)
 add_executable(${PROJECT_NAME}_NLOC_node src/my_NLOC_generalConstrained.cpp)
# target_link_libraries(my_NLOC_generalConstrained ${catkin_LIBRARIES} ${HPIPM_LIBS})
 add_executable(${PROJECT_NAME}_hyqnloc_node src/NLOC_hyq.cpp)
 target_link_libraries(${PROJECT_NAME}_hyqnloc_node ${LAPACK_LIBS} ${ct_model_libs} ${catkin_LIBRARIES} ${HPIPM_LIBS} ${MATLAB_LIB_DIR})
endif(HPIPM)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
 target_link_libraries(${PROJECT_NAME}_NLOC_node ${catkin_LIBRARIES})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_my_ct_project.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
