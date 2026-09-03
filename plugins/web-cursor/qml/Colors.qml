// Palette bridge. Same pattern as the wallpaper-selector plugin: a plain
// object that maps the Caelestia shell's material-3 palette (provided by the
// shell runtime through `Caelestia.Config` + `qs.services`) onto short color
// names used by this plugin's QML UI.
import QtQuick
import Caelestia.Config
import qs.services

QtObject {
    id: colors

    property color primary: Colours.palette.m3primary
    property color primaryText: Colours.palette.m3onPrimary
    property color primaryContainer: Colours.palette.m3primaryContainer
    property color primaryContainerText: Colours.palette.m3onPrimaryContainer
    property color primaryForeground: Colours.palette.m3onPrimary

    property color secondary: Colours.palette.m3secondary
    property color secondaryText: Colours.palette.m3onSecondary
    property color secondaryContainer: Colours.palette.m3secondaryContainer
    property color secondaryContainerText: Colours.palette.m3onSecondaryContainer

    property color tertiary: Colours.palette.m3tertiary
    property color tertiaryText: Colours.palette.m3onTertiary
    property color tertiaryContainer: Colours.palette.m3tertiaryContainer
    property color tertiaryContainerText: Colours.palette.m3onTertiaryContainer

    property color background: Colours.palette.m3background
    property color backgroundText: Colours.palette.m3onBackground
    property color surface: Colours.palette.m3surface
    property color surfaceText: Colours.palette.m3onSurface
    property color surfaceVariant: Colours.palette.m3surfaceVariant
    property color surfaceVariantText: Colours.palette.m3onSurfaceVariant
    property color surfaceContainer: Colours.palette.m3surfaceContainer

    property color error: Colours.palette.m3error
    property color errorText: Colours.palette.m3onError
    property color errorContainer: Colours.palette.m3errorContainer
    property color errorContainerText: Colours.palette.m3onErrorContainer

    property color outline: Colours.palette.m3outline
    property color shadow: Colours.palette.m3shadow
    property color inverseSurface: Colours.palette.m3inverseSurface
    property color inverseSurfaceText: Colours.palette.m3inverseOnSurface
    property color inversePrimary: Colours.palette.m3inversePrimary
}
