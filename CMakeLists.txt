cmake_minimum_required(VERSION 3.20)

project(surface_vectorizer_pipeline_test LANGUAGES CXX)

# -----------------------------------------------------------------------------
# C++ standard
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------------
# Optimization flags
# -----------------------------------------------------------------------------

if(MSVC)

    # максимальная оптимизация
    add_compile_options(/O2 /Ob2 /Oi /Ot /GL /MP)

    # link-time optimization
    add_link_options(/LTCG)

else()

    add_compile_options(-O3 -march=native -ffast-math)

endif()

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------

add_library(surface_vectorizer_core STATIC)

target_sources(surface_vectorizer_core PRIVATE

    # image
    src/image/oklab_convert.cpp
    src/image/image_io_stb.cpp
    src/image/scanline_integral_stats.cpp
    src/image/integral_map.cpp

    # surface
    src/surface/mesh.cpp
    src/surface/mesh_topology.cpp
    src/surface/mesh_builders.cpp
    src/surface/triangle_plane.cpp

    # rendering
    src/render/reference_renderer.cpp
    src/render/surface_plane_renderer.cpp

    # error
    src/error/error_estimator.cpp
    src/error/hierarchical_surface_error.cpp

    # edges
    src/edge/edge_map.cpp

    # pyramid
    src/pyramid/error_pyramid.cpp

    # refinement
    src/refine/adaptive_refinement.cpp
    src/refine/pyramid_refinement.cpp

    # mesh optimization
    src/meshopt/mesh_optimization.cpp

    # stb implementation
    third_party/stb_impl.cpp
)

target_include_directories(surface_vectorizer_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# -----------------------------------------------------------------------------
# Warnings
# -----------------------------------------------------------------------------

if(MSVC)

    target_compile_options(surface_vectorizer_core PRIVATE
        /W4
        /permissive-
    )

else()

    target_compile_options(surface_vectorizer_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

endif()

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------

add_executable(pipeline_battle_test
    apps/pipeline_battle_test/main.cpp
)

target_link_libraries(pipeline_battle_test PRIVATE surface_vectorizer_core)

# -----------------------------------------------------------------------------
# LTO / IPO (whole program optimization)
# -----------------------------------------------------------------------------

include(CheckIPOSupported)

check_ipo_supported(RESULT ipo_supported)

if(ipo_supported)

    set_property(TARGET surface_vectorizer_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET pipeline_battle_test PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

endif()

# -----------------------------------------------------------------------------
# Visual Studio settings
# -----------------------------------------------------------------------------

if(MSVC)

    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY VS_STARTUP_PROJECT pipeline_battle_test
    )

    set_target_properties(pipeline_battle_test PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

endif()