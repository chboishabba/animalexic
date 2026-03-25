cmake_minimum_required(VERSION 3.16)
project(animalexic_shaders LANGUAGES NONE)

find_program(GLSLC glslc REQUIRED)

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BUILD_DIR ${SHADER_DIR}/build)
file(MAKE_DIRECTORY ${SHADER_BUILD_DIR})

set(SHADER_SOURCES
    ${SHADER_DIR}/frame_diff.comp
    ${SHADER_DIR}/stereo_roi.comp
)

set(SPV_OUTPUTS "")
foreach(SRC ${SHADER_SOURCES})
    get_filename_component(NAME_WE ${SRC} NAME_WE)
    set(SPV ${SHADER_BUILD_DIR}/${NAME_WE}.spv)
    add_custom_command(
        OUTPUT ${SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BUILD_DIR}
        COMMAND ${GLSLC} -O -std=450 -o ${SPV} ${SRC}
        DEPENDS ${SRC}
        COMMENT "Compiling ${SRC} -> ${SPV}"
        VERBATIM
    )
    list(APPEND SPV_OUTPUTS ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})

message(STATUS "Use: cmake -S . -B build && cmake --build build --target shaders")
