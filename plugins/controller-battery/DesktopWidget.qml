pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Shapes
import QtQuick.Effects
import QtCore
import Quickshell
import Quickshell.Services.UPower
import "."

// Desktop widget: battery level of connected game controllers, one ring per
// controller with an icon in the middle.
//
// - Drag the card to move it; the position is remembered.
//   Double-click the card to put it back to its default corner.
// - Replace the SVG files in the "icons" folder with your own artwork
//   (same file names), or point the rules below at other files.
// - Devices come from UPower, so a controller only shows up while the kernel
//   exposes a battery for it (the Steam controller disappears while the Steam
//   client is running, because Steam takes the device over).
Item {
    id: root

    // Filled in by Background.qml for the properties declared here.
    property ShellScreen screen: null

    // The desktop layer only passes input to this region when the Plasma
    // wallpaper is shown instead of Caelestia's.
    property var inputRegion: Region {
        item: root.visible ? card : null
    }

    // ---- Tweak these ----------------------------------------------------
    // Default position, used until the card has been dragged somewhere.
    readonly property bool alignRight: true
    readonly property bool alignBottom: false
    readonly property int edgeMargin: 32          // distance from the left/right edge
    readonly property int verticalMargin: 72      // distance from top/bottom (clears the bar)

    readonly property int padding: 10             // same space on all four sides of the card
    readonly property int ringSize: 76
    readonly property int ringThickness: 7
    readonly property int labelWidth: ringSize + 20 // wide enough for the text under a ring
    readonly property int itemSpacing: 20         // gap between controllers
    readonly property bool showLabels: true       // controller name under the percentage
    readonly property bool tintIcons: true        // recolour icons to the theme; turn off for coloured SVGs
    readonly property real lowThreshold: 0.2      // fraction at or below which the ring turns red
    readonly property bool onlyFirstScreen: true  // show on one monitor only
    readonly property bool showAllPeripherals: false // also list mice, keyboards, headsets...

    // First matching rule wins. "match" is tested against the device's sysfs
    // name and model; "icon" is relative to this folder. These rules also
    // decide what counts as a controller, so a device matching none of them is
    // ignored unless showAllPeripherals is on.
    readonly property var iconRules: [
        { match: /steam/i, name: "Steam Controller", icon: "icons/steam-controller.svg" },
        { match: /ps-controller|dualsense|dualshock/i, name: "PS5 Controller", icon: "icons/ps5-controller.svg" },
        { match: /controller|gamepad|joystick/i, name: "Controller", icon: "icons/controller.svg" }
    ]
    // For peripherals pulled in by showAllPeripherals, which no rule names.
    readonly property string fallbackIcon: "icons/controller.svg"
    // ---------------------------------------------------------------------

    Theme {
        id: theme
    }

    function isController(d: var): bool {
        if (d.isLaptopBattery || d.type === UPowerDeviceType.LinePower || !d.isPresent)
            return false;
        return showAllPeripherals || d.type === UPowerDeviceType.GamingInput || ruleFor(d) !== null;
    }

    function ruleFor(d: var): var {
        const id = (d.nativePath ?? "") + " " + (d.model ?? "");
        return iconRules.find(r => r.match.test(id)) ?? null;
    }

    function labelFor(d: var): string {
        return ruleFor(d)?.name ?? (d.model || "Controller");
    }

    function iconFor(d: var): url {
        return Qt.resolvedUrl(ruleFor(d)?.icon ?? fallbackIcon);
    }

    readonly property var controllers: UPower.devices.values.filter(d => isController(d))

    readonly property bool onThisScreen: {
        if (!onlyFirstScreen || !screen)
            return true;
        return screen.name === Quickshell.screens[0]?.name;
    }

    // ---- Position (draggable, remembered) -------------------------------

    Settings {
        id: store

        // -1,-1 means "never moved": use the default corner.
        property point cardPos: Qt.point(-1, -1)

        category: "Plugin_controller-battery"

        onCardPosChanged: root.place()
    }

    readonly property real maxX: Math.max(0, root.width - card.width)
    readonly property real maxY: Math.max(0, root.height - card.height)

    function clampX(v: real): real {
        return Math.max(0, Math.min(root.maxX, v));
    }

    function clampY(v: real): real {
        return Math.max(0, Math.min(root.maxY, v));
    }

    function place(): void {
        if (dragHandler.active)
            return;
        const dx = alignRight ? root.maxX - edgeMargin : edgeMargin;
        const dy = alignBottom ? root.maxY - verticalMargin : verticalMargin;
        card.x = clampX(store.cardPos.x >= 0 ? store.cardPos.x : dx);
        card.y = clampY(store.cardPos.y >= 0 ? store.cardPos.y : dy);
    }

    function resetPosition(): void {
        store.cardPos = Qt.point(-1, -1);
    }

    onWidthChanged: place()
    onHeightChanged: place()
    Component.onCompleted: Qt.callLater(place)

    visible: onThisScreen && controllers.length > 0

    Rectangle {
        id: card

        width: row.implicitWidth + root.padding * 2
        height: row.implicitHeight + root.padding * 2
        radius: 22
        color: Qt.alpha(theme.surface, 0.92)

        onWidthChanged: root.place()
        onHeightChanged: root.place()

        HoverHandler {
            cursorShape: Qt.OpenHandCursor
        }

        DragHandler {
            id: dragHandler

            target: card
            xAxis.minimum: 0
            xAxis.maximum: root.maxX
            yAxis.minimum: 0
            yAxis.maximum: root.maxY
            cursorShape: Qt.ClosedHandCursor

            onActiveChanged: {
                if (!active)
                    store.cardPos = Qt.point(card.x, card.y);
            }
        }

        TapHandler {
            acceptedButtons: Qt.LeftButton
            onDoubleTapped: root.resetPosition()
        }

        Row {
            id: row

            x: root.padding
            y: root.padding
            spacing: root.itemSpacing

            Repeater {
                model: ScriptModel {
                    values: root.controllers
                }

                Column {
                    id: entry

                    required property var modelData

                    readonly property real level: modelData.percentage
                    readonly property bool charging: modelData.state === UPowerDeviceState.Charging
                        || modelData.state === UPowerDeviceState.PendingCharge
                    // Finished charging (e.g. a controller resting on its dock).
                    readonly property bool full: modelData.state === UPowerDeviceState.FullyCharged
                    // A docked controller is not charging but is still plugged in.
                    readonly property bool pluggedIn: charging || full
                    readonly property bool low: level <= root.lowThreshold && !pluggedIn
                    readonly property color ringColor: low ? theme.warn : theme.accent

                    // Resolved once per entry rather than inside each binding below.
                    readonly property string label: root.labelFor(modelData)
                    readonly property url icon: root.iconFor(modelData)

                    // Eased copy of the level so the ring glides between readings.
                    property real shownLevel: level

                    spacing: 8

                    Behavior on shownLevel {
                        NumberAnimation {
                            duration: 400
                            easing.type: Easing.OutCubic
                        }
                    }

                    Item {
                        width: root.labelWidth
                        height: root.ringSize

                        Shape {
                            id: ring

                            width: root.ringSize
                            height: root.ringSize
                            anchors.horizontalCenter: parent.horizontalCenter
                            preferredRendererType: Shape.CurveRenderer

                            ShapePath {
                                strokeColor: theme.track
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
                                strokeColor: entry.ringColor
                                strokeWidth: root.ringThickness
                                fillColor: "transparent"
                                capStyle: ShapePath.RoundCap

                                PathAngleArc {
                                    centerX: ring.width / 2
                                    centerY: ring.height / 2
                                    radiusX: (ring.width - root.ringThickness) / 2
                                    radiusY: (ring.height - root.ringThickness) / 2
                                    startAngle: -90
                                    sweepAngle: Math.max(0.5, Math.min(359.9, 360 * entry.shownLevel))
                                }
                            }
                        }

                        Image {
                            anchors.centerIn: ring
                            width: root.ringSize * 0.6
                            height: width
                            source: entry.icon
                            sourceSize: Qt.size(width * 2, height * 2)
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            mipmap: true

                            layer.enabled: root.tintIcons
                            layer.effect: MultiEffect {
                                colorization: 1.0
                                colorizationColor: entry.low ? theme.warn : theme.secondary
                            }
                        }
                    }

                    Text {
                        width: root.labelWidth
                        horizontalAlignment: Text.AlignHCenter
                        text: Math.round(entry.level * 100) + "%" + (entry.pluggedIn ? " 🗲" : "")
                        color: entry.low ? theme.warn : theme.secondary
                        font.pixelSize: 15
                        font.bold: true
                        lineHeight: 20
                        lineHeightMode: Text.FixedHeight
                        verticalAlignment: Text.AlignVCenter
                    }

                    Text {
                        visible: root.showLabels
                        width: root.labelWidth
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideRight
                        text: entry.label
                        color: theme.subtle
                        font.pixelSize: 12
                    }
                }
            }
        }
    }
}
