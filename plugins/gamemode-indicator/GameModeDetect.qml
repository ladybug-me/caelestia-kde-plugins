import QtQuick
import Quickshell
import Quickshell.Io

// Feral GameMode (the `gamemode` / `gamemoded` daemon) does not tell the shell
// anything on its own, so poll its CLI status. Caelestia's own switch is read
// from GameMode.enabled by whoever uses this component.
Item {
    id: root

    readonly property bool feralActive: _active

    property bool _active: false

    visible: false
    implicitWidth: 0
    implicitHeight: 0

    Timer {
        interval: 4000
        running: true
        repeat: true
        triggeredOnStart: true

        onTriggered: status.running = true
    }

    Process {
        id: status

        command: ["gamemoded", "-s"]

        stdout: StdioCollector {
            onStreamFinished: {
                root._active = text.includes("is active");
            }
        }
    }
}
