kcoreaddons_add_plugin(
    ultralightwebcursor
    INSTALL_NAMESPACE "kwin/effects/plugins"
)


target_sources(ultralightwebcursor PRIVATE
    main.cpp
    ../lib/UltralightHtmlEffect.cpp
    KwinMouseProvider.cpp
    ../lib/BlackList/BlacklistManager.cpp
    ../header/KwinCursorEffect.hpp
    ../header/KwinMouseProvider.hpp
    ../header/UltralightHtmlEffect.hpp
    ../lib/BlackList/BlacklistManager.hpp
    ../lib/CrashReport/CrashReport.hpp
    #../../dep/glad/src/glad.c
    ../lib/UltralightPl/WebListener.hpp
    ../lib/WebCall/WebCall.hpp
    ../header/MainCursorStaff.hpp
     $<TARGET_OBJECTS:webkde_core>
)
target_include_directories(ultralightwebcursor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/contents
    ${CMAKE_SOURCE_DIR}/contents/config
    ${CMAKE_SOURCE_DIR}/contents/header
    ${CMAKE_SOURCE_DIR}/contents/lib
    /usr/include/kwin
    /usr/include/kwin/core
)


target_link_libraries(ultralightwebcursor
    PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::DBus
    KF6::CoreAddons
    KF6::ConfigCore
    KWin::kwin
    Ultralight::SDK
    PkgConfig::EPOXY
)
