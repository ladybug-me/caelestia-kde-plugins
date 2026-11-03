import QtQuick
import qs.services

// The shell's Material palette, under the names this widget uses.
QtObject {
    readonly property color surface: Colours.palette.m3surfaceContainer
    readonly property color track: Colours.palette.m3surfaceContainerHighest
    readonly property color subtle: Colours.palette.m3onSurfaceVariant
    readonly property color accent: Colours.palette.m3primary
    readonly property color secondary: Colours.palette.m3secondary
    readonly property color warn: Colours.palette.m3error
}
