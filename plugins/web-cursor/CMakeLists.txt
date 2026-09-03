cmake_minimum_required(VERSION 3.19)

project(caelestia-webcursor LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
add_compile_options(-fexceptions)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EPOXY REQUIRED IMPORTED_TARGET epoxy)

set(ULTRALIGHT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty"
    CACHE PATH "Path to an extracted Ultralight SDK")
set(ULTRALIGHT_REPO "LuYishan-4/ThirdParty"
    CACHE STRING "GitHub repository mirroring the Ultralight SDK used for the automatic download")
set(ULTRALIGHT_REF "main"
    CACHE STRING "Branch/tag/commit of ULTRALIGHT_REPO fetched when no local copy is found")
set(ULTRALIGHT_DOWNLOAD_URL ""
    CACHE STRING "Override the automatic Ultralight SDK download URL (full tarball URL, e.g. your own mirror)")
option(ULTRALIGHT_DOWNLOAD "Automatically download the Ultralight SDK when no local copy is found" ON)

if(NOT EXISTS "${ULTRALIGHT_ROOT}/include/AppCore/App.h")
    if(NOT ULTRALIGHT_DOWNLOAD)
        message(FATAL_ERROR
            "Ultralight SDK not found at ${ULTRALIGHT_ROOT}. "
            "Download/extract it there manually, point -DULTRALIGHT_ROOT=<path> at an "
            "existing SDK, or re-enable ULTRALIGHT_DOWNLOAD to fetch it automatically.")
    endif()

    if(NOT ULTRALIGHT_DOWNLOAD_URL)
        set(ULTRALIGHT_DOWNLOAD_URL
            "https://codeload.github.com/${ULTRALIGHT_REPO}/tar.gz/${ULTRALIGHT_REF}")
    endif()

    set(_sdk_dir "${CMAKE_BINARY_DIR}/ultralight-sdk")
    file(MAKE_DIRECTORY "${_sdk_dir}")
    set(_sdk_archive "${_sdk_dir}/thirdparty.tar.gz")
    message(STATUS "[web-cursor] Ultralight SDK not found; downloading ${ULTRALIGHT_REPO}@${ULTRALIGHT_REF} ...")
    file(DOWNLOAD "${ULTRALIGHT_DOWNLOAD_URL}" "${_sdk_archive}"
        SHOW_PROGRESS TIMEOUT 600 STATUS _dl_status LOG _dl_log)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR
            "[web-cursor] Failed to download the Ultralight SDK from:\n"
            "  ${ULTRALIGHT_DOWNLOAD_URL}\n"
            "${_dl_log}\n"
            "Install it manually and re-run cmake with -DULTRALIGHT_ROOT=<path> "
            "or -DULTRALIGHT_DOWNLOAD=OFF.")
    endif()

    file(ARCHIVE_EXTRACT INPUT "${_sdk_archive}" DESTINATION "${_sdk_dir}")

    # The archive contains a single top-level folder (e.g. ThirdParty-main);
    # locate it instead of assuming the archive's internal layout.
    set(_found_root "")
    file(GLOB _sdk_candidates "${_sdk_dir}/*")
    foreach(_cand IN LISTS _sdk_candidates)
        if(EXISTS "${_cand}/include/AppCore/App.h" AND EXISTS "${_cand}/bin/libUltralightCore.so")
            set(_found_root "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT _found_root)
        message(FATAL_ERROR
            "[web-cursor] Downloaded archive does not look like an Ultralight SDK: ${_sdk_archive}\n"
            "Point -DULTRALIGHT_ROOT=<path> at an extracted SDK and re-run cmake.")
    endif()

    set(ULTRALIGHT_ROOT "${_found_root}" CACHE PATH "Path to an extracted Ultralight SDK" FORCE)
    message(STATUS "[web-cursor] Ultralight SDK ready at ${ULTRALIGHT_ROOT}")
endif()

set(ULTRALIGHT_BIN "${ULTRALIGHT_ROOT}/bin")

function(ultralight_import_library target_name lib_basename)
    add_library(${target_name} SHARED IMPORTED)
    set_target_properties(${target_name} PROPERTIES
        IMPORTED_LOCATION "${ULTRALIGHT_BIN}/lib${lib_basename}.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ULTRALIGHT_ROOT}/include"
    )
endfunction()

ultralight_import_library(Ultralight::Ultralight Ultralight)
ultralight_import_library(Ultralight::UltralightCore UltralightCore)
ultralight_import_library(Ultralight::AppCore AppCore)
ultralight_import_library(Ultralight::WebCore WebCore)

add_library(Ultralight::SDK INTERFACE IMPORTED)
target_link_libraries(Ultralight::SDK INTERFACE
    Ultralight::Ultralight
    Ultralight::UltralightCore
    Ultralight::AppCore
    Ultralight::WebCore
)

set(CMAKE_BUILD_RPATH "${ULTRALIGHT_BIN}")
set(CMAKE_INSTALL_RPATH "/usr/lib/webkde_core")

find_package(Qt6 6.6 REQUIRED COMPONENTS Core Gui DBus Widgets)

find_package(ECM 6.0 REQUIRED NO_MODULE)
set(CMAKE_MODULE_PATH ${ECM_MODULE_PATH})
include(KDEInstallDirs)
include(KDECMakeSettings)
include(KDECompilerSettings NO_POLICY_SCOPE)

find_package(KF6 REQUIRED COMPONENTS CoreAddons Config)
find_package(KWin REQUIRED)

set(WEBCURSOR_CONTENTS "${CMAKE_CURRENT_SOURCE_DIR}/contents")

add_library(webkde_core OBJECT
    ${WEBCURSOR_CONTENTS}/config/CursorJSON.cpp
    ${WEBCURSOR_CONTENTS}/config/CursorJSON.hpp
    ${WEBCURSOR_CONTENTS}/config/UserConfig.cpp
    ${WEBCURSOR_CONTENTS}/config/UserConfig.hpp
    ${WEBCURSOR_CONTENTS}/lib/Quick/PluginPath/PluginPath.cpp
    ${WEBCURSOR_CONTENTS}/lib/Quick/PluginPath/PluginPath.hpp
)
set_target_properties(webkde_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(webkde_core PUBLIC
    ${WEBCURSOR_CONTENTS}
    ${WEBCURSOR_CONTENTS}/config
    ${WEBCURSOR_CONTENTS}/header
    ${WEBCURSOR_CONTENTS}/lib
    ${WEBCURSOR_CONTENTS}/lib/Quick/PluginPath

)
target_link_libraries(webkde_core PUBLIC Qt6::Core Qt6::DBus )

add_subdirectory(contents/kwin-plasma)

install(DIRECTORY ${WEBCURSOR_CONTENTS}/WebCursor/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/caelestia/webcursor")
install(DIRECTORY "${ULTRALIGHT_BIN}/"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/webkde_core"
    FILES_MATCHING PATTERN "*.so*")
install(DIRECTORY "${ULTRALIGHT_ROOT}/resources"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/caelestia/webcursor")
