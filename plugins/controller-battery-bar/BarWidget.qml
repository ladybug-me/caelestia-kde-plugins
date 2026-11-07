pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Shapes
import Quickshell.Services.UPower
import Caelestia.Config
import qs.components
import qs.components.controls
import qs.services
import "."

// Bar widget: a ring that empties as the controller battery drains, with the
// level inside. Green above 30%, amber down to 10%, red below. Clicking opens
// a popout with the details of every connected controller.
StyledRect {
    id: root

    property var bar: null
    property var screen: null
    property var popouts: null

    // Same thresholds the desktop widget uses, plus the green band the bar
    // wants for "plenty of battery left".
    readonly property real warnLevel: 0.3
    readonly property real lowLevel: 0.1

    readonly property bool horizontal: bar ? bar.isHorizontal !== false : true
    readonly property int thickness: bar && bar.barThickness ? bar.barThickness : Tokens.sizes.bar.innerWidth
    // Sized against the bar's other icons (about 0.42 of the thickness) instead
    // of using the whole bar height, which made it look oversized next to them.
    readonly property real ringSize: Math.max(16, Math.round(root.thickness * 0.55))
    readonly property real ringThickness: Math.max(1.5, root.ringSize / 11)
    readonly property int digits: String(Math.round(root.level * 100)).length
    // The number has to fit inside the ring, so it shrinks with the digit count
    // ("100" is three characters) instead of using a fixed size that overflows.
    readonly property real ringFontSize: Math.max(5, Math.min(Math.round(root.ringSize * 0.40), Math.floor((root.ringSize - root.ringThickness * 2 - 2) / (root.digits * 0.75))))
    readonly property var controllers: devices.controllers
    readonly property var worst: controllers.length > 0 ? controllers.reduce((a, b) => (b.percentage ?? 0) < (a.percentage ?? 0) ? b : a) : null
    readonly property real level: worst ? (worst.percentage ?? 0) : 1
    readonly property bool pluggedIn: controllers.some(d => devices.pluggedInFor(d))
    readonly property color levelColour: root.pluggedIn ? "#4ade80" : root.level > root.warnLevel ? "#4ade80" : root.level > root.lowLevel ? "#fbbf24" : Colours.palette.m3error
    readonly property bool popoutOpen: popouts ? (popouts.currentName === "controllerBatteryBar" && popouts.hasCurrent === true) : false

    // Eased copy of the level so the ring glides between readings.
    property real shownLevel: level

    function centerAlongBar(): real {
        const p = root.mapToItem(null, root.width / 2, root.height / 2);
        const vertical = Config.bar.position === "left" || Config.bar.position === "right";
        return vertical ? p.y : p.x;
    }

    function setPopout(open: bool): void {
        if (!popouts)
            return;
        if (open)
            popouts.currentCenter = root.centerAlongBar();
        popouts.currentName = open ? "controllerBatteryBar" : "";
        popouts.hasCurrent = open;
    }

    // Controllers are off most of the time, so the widget only takes a slot
    // while one is actually connected.
    visible: controllers.length > 0
    implicitWidth: root.horizontal ? (root.ringSize + Tokens.padding.small) : root.thickness
    implicitHeight: root.horizontal ? root.thickness : (root.ringSize + Tokens.padding.small)
    radius: Tokens.rounding.full
    color: Qt.alpha(Colours.tPalette.m3surfaceContainer, 0)

    Behavior on shownLevel {
        NumberAnimation {
            duration: 400
            easing.type: Easing.OutCubic
        }
    }

    Devices {
        id: devices
    }

    StateLayer {
        color: root.levelColour

        onClicked: root.setPopout(!root.popoutOpen)
    }

    Shape {
        id: ring

        anchors.centerIn: parent
        width: root.ringSize
        height: root.ringSize
        preferredRendererType: Shape.CurveRenderer

        ShapePath {
            strokeColor: Colours.palette.m3surfaceContainerHighest
            strokeWidth: root.ringThickness
            fillColor: "transparent"

            PathAngleArc {
                centerX: ring.width / 2
                centerY: ring.height / 2
                radiusX: (ring.width - root.ringThickness) / 2
                radiusY: (ring.height - root.ringThickness) / 2
                startAngle: -90
                sweepAngle: 360
            }
        }

        ShapePath {
            strokeColor: root.levelColour
            strokeWidth: root.ringThickness
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap

            PathAngleArc {
                centerX: ring.width / 2
                centerY: ring.height / 2
                radiusX: (ring.width - root.ringThickness) / 2
                radiusY: (ring.height - root.ringThickness) / 2
                startAngle: -90
                sweepAngle: Math.max(0.5, Math.min(359.9, 360 * root.shownLevel))
            }
        }
    }

    StyledText {
        id: label

        anchors.centerIn: parent
        width: root.ringSize - root.ringThickness * 2 - 2
        horizontalAlignment: Text.AlignHCenter
        text: Math.round(root.level * 100)
        color: root.levelColour
        font.pixelSize: root.ringFontSize
        font.weight: Font.DemiBold
    }
}
