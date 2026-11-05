import QtQuick
import Quickshell
import Caelestia
import qs.services

Item {
    id: root

    // The plugin only needs to watch for game mode and announce it with the
    // shell's own toast; the visual part lives in the bar widget.
    visible: false
    implicitWidth: 0
    implicitHeight: 0

    function announce(enabled) {
        if (enabled)
            Toaster.toast(qsTr("Game mode enabled"), qsTr("Detected by gamemoded"), "gamepad");
        else
            Toaster.toast(qsTr("Game mode disabled"), qsTr("gamemoded is no longer active"), "gamepad");
    }

    Connections {
        target: detect

        function onFeralActiveChanged() {
            // Caelestia's own game mode already has its own toast, so only
            // announce the external gamemoded case here.
            if (GameMode.enabled)
                return;
            Qt.callLater(root.announce, detect.feralActive);
        }
    }

    GameModeDetect {
        id: detect
    }
}
