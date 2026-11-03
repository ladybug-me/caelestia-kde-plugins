pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import Caelestia
import Caelestia.Config
import qs.components
import qs.services
import "."

// The bar cat, panel-sized. It draws the same frame CatState publishes, so the
// two run in step; with no bar widget loaded the service still samples, so the
// popout stays alive on its own.
Item {
    id: root

    property var popouts
    property var visibilities

    readonly property int panelCatSize: 120

    implicitWidth: 240
    implicitHeight: layout.implicitHeight

    function closePanel(): void {
        if (root.popouts)
            root.popouts.hasCurrent = false;
    }

    ColumnLayout {
        id: layout

        width: root.implicitWidth
        spacing: Tokens.spacing.small

        Text {
            Layout.alignment: Qt.AlignHCenter
            color: CatState.colour
            font.family: CatState.fontFamily
            font.pixelSize: root.panelCatSize
            horizontalAlignment: Text.AlignHCenter
            text: CatState.glyph
        }
        StyledText {
            Layout.alignment: Qt.AlignHCenter
            color: CatState.colour
            font: Tokens.font.title.small
            text: qsTr("CPU %1%").arg(CatState.cpuPercent)
        }
        StyledText {
            Layout.alignment: Qt.AlignHCenter
            color: Colours.palette.m3onSurfaceVariant
            font: Tokens.font.body.small
            text: CatState.pace === "idle" ? qsTr("Asleep") : CatState.pace === "walk" ? qsTr("Walking") : qsTr("Running")
        }
    }
    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: root.closePanel()
    }
}
