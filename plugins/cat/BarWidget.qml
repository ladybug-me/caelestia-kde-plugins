pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import Caelestia.Config
import qs.components
import qs.services
import "."

Item {
    id: root

    property var screen
    property var popouts

    readonly property string widgetId: "cat"
    readonly property bool active: root.popouts ? (root.popouts.currentName === root.widgetId && root.popouts.hasCurrent === true) : false
    readonly property bool vertical: Config.bar.position === "left" || Config.bar.position === "right"

    implicitWidth: (root.vertical ? verticalCat.implicitWidth : horizontalCat.implicitWidth) + Tokens.padding.small * 2
    implicitHeight: (root.vertical ? verticalCat.implicitHeight : horizontalCat.implicitHeight) + Tokens.padding.small * 2

    function centerAlongBar(): real {
        const p = root.mapToItem(null, root.width / 2, root.height / 2);
        return root.vertical ? p.y : p.x;
    }

    function setOpen(open: bool): void {
        if (!root.popouts)
            return;
        if (open)
            root.popouts.currentCenter = root.centerAlongBar();
        root.popouts.currentName = open ? root.widgetId : "";
        root.popouts.hasCurrent = open;
    }

    function handleHover(_pos, _isHorizontal, shellPopouts): void {
        const target = shellPopouts ?? root.popouts;
        if (!target || target.currentName !== root.widgetId || target.hasCurrent !== true)
            return;
        target.currentCenter = root.centerAlongBar();
    }

    component CatGlyph: Text {
        color: CatState.colour
        font.family: CatState.fontFamily
        font.pixelSize: CatState.catSize
        text: CatState.glyph
    }

    component CpuLabel: Text {
        color: CatState.colour
        font: Tokens.font.body.small
        text: `${CatState.cpuPercent}%`
        visible: CatState.showCpuPercent
    }

    Connections {
        function onPanelRequested(): void {
            root.setOpen(!root.active);
        }

        target: CatState
    }

    StyledRect {
        anchors.fill: parent
        color: Colours.palette.m3onSurface
        opacity: root.active ? 0.14 : (mouse.containsMouse ? 0.08 : 0)
        radius: Tokens.rounding.extraLarge

        Behavior on opacity {
            NumberAnimation {
                duration: 120
            }
        }
    }
    RowLayout {
        id: horizontalCat

        anchors.centerIn: parent
        spacing: 4
        visible: !root.vertical

        CatGlyph {
            Layout.alignment: Qt.AlignVCenter
        }
        CpuLabel {
            Layout.alignment: Qt.AlignVCenter
        }
    }
    ColumnLayout {
        id: verticalCat

        anchors.centerIn: parent
        spacing: 4
        visible: root.vertical

        CatGlyph {
            Layout.alignment: Qt.AlignHCenter
        }
        CpuLabel {
            Layout.alignment: Qt.AlignHCenter
        }
    }
    MouseArea {
        id: mouse

        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        hoverEnabled: true
        onClicked: root.setOpen(!root.active)
    }
}
