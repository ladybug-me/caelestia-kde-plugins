pragma Singleton

import QtQuick
import Quickshell
import Quickshell.Io
import qs.services

// Frame the bar widget and the popout both draw from. The service that samples
// CPU usage writes it; anything else only reads.
//
// QtObject has no default property, so the font loader and the IPC handler are
// held as properties rather than declared as children.
QtObject {
    id: root

    signal panelRequested()

    // Font registration is process-wide and keyed by path, so the file name has
    // to change whenever the glyph set does, or the shell keeps the old face.
    readonly property FontLoader catFont: FontLoader {
        source: Qt.resolvedUrl("catwalk2.otf")
    }
    readonly property string fontFamily: root.catFont.status === FontLoader.Ready ? root.catFont.name : ""

    property string glyph: "a"
    property string pace: "idle"
    property int cpuPercent: 0
    property int catSize: 24
    property bool showCpuPercent: false
    property string colourMode: "theme"
    property color customColour: "#E8A24C"

    readonly property color colour: root.colourMode === "custom" ? root.customColour : Colours.palette.m3secondary

    readonly property IpcHandler ipc: IpcHandler {
        target: "cat"

        function status(): string {
            return JSON.stringify({
                pace: root.pace,
                glyph: root.glyph,
                cpuPercent: root.cpuPercent,
                catSize: root.catSize,
                showCpuPercent: root.showCpuPercent,
                colourMode: root.colourMode,
                fontFamily: root.fontFamily
            });
        }

        function panel(): void {
            root.panelRequested();
        }
    }
}
