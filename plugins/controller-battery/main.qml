import QtQuick
import qs.services

// Loaded by the shell whenever the plugin is enabled.
//
// The shell registers the desktop widget from metadata.json when it starts, but
// not when the plugin is switched off and on again in Nexus (it looks the
// manifest up in a ListModel, which drops the nested widget list). So the
// plugin registers its own widget shortly after being loaded. If the shell
// has already registered it, this does nothing.
Item {
    id: root

    readonly property string pluginId: "controller-battery"

    visible: false

    Timer {
        // Runs after the shell has finished its own (failing) bookkeeping.
        interval: 150
        running: true
        onTriggered: {
            if (BackgroundWidgetRegistry.isKnown("controllerBattery"))
                return;
            BackgroundWidgetRegistry.register(root.pluginId, [
                {
                    id: "controllerBattery",
                    name: "Controller Battery",
                    z: 4,
                    componentUrl: Qt.resolvedUrl("DesktopWidget.qml")
                }
            ]);
        }
    }

    Component.onDestruction: BackgroundWidgetRegistry.unregister(root.pluginId)
}
