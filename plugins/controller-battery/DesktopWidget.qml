import QtQuick
import QtQuick.Shapes
import QtQuick.Effects
import QtCore
import Quickshell
import Quickshell.Services.UPower

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
    property var screen
    property var modelData
    property var background

    // The desktop layer only passes input to this region when the Plasma
    // wallpaper is shown instead of Caelestia's.
    property var inputRegion: Region {
        item: card
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
    readonly property int itemSpacing: 20         // gap between controllers
    readonly property bool showLabels: true       // controller name under the percentage
    readonly property bool tintIcons: true        // recolour icons to the theme; turn off for coloured SVGs
    readonly property real lowThreshold: 0.2      // fraction at or below which the ring turns red
    readonly property bool onlyFirstScreen: true  // show on one monitor only
    readonly property bool showAllPeripherals: false // also list mice, keyboards, headsets...

    // First matching rule wins. "match" is tested against the device's sysfs
    // name and model; "icon" is relative to this folder.
    readonly property var iconRules: [
        { match: /steam/i, name: "Steam Controller", icon: "icons/steam-controller.svg" },
        { match: /ps-controller|dualsense|dualshock/i, name: "PS5 Controller", icon: "icons/ps5-controller.svg" }
    ]
    readonly property string fallbackIcon: "icons/controller.svg"
    // ---------------------------------------------------------------------

    // Use the shell's colour palette when it can be imported from here;
    // otherwise fall back to fixed colours below.
    readonly property var theme: {
        try {
            return Qt.createQmlObject(`
                import QtQuick
                import qs.services
                QtObject {
                    readonly property color surface: Colours.palette.m3surfaceContainer
                    readonly property color track: Colours.palette.m3surfaceContainerHighest
                    readonly property color text: Colours.palette.m3onSurface
                    readonly property color subtle: Colours.palette.m3onSurfaceVariant
                    readonly property color accent: Colours.palette.m3primary
                    readonly property color secondary: Colours.palette.m3secondary
                    readonly property color warn: Colours.palette.m3error
                }
            `, root, "controllerBatteryTheme");
        } catch (e) {
            return null;
        }
    }

    readonly property color surfaceColor: theme ? theme.surface : "#1e2327"
    readonly property color trackColor: theme ? theme.track : "#38414a"
    readonly property color textColor: theme ? theme.text : "#e6e6e6"
    readonly property color subtleColor: theme ? theme.subtle : "#a0a8b0"
    readonly property color accentColor: theme ? theme.accent : "#8ab4f8"
    readonly property color secondaryColor: theme ? theme.secondary : "#5d8bf4"
    readonly property color warnColor: theme ? theme.warn : "#f28b82"

    function isController(d: var): bool {
        if (d.isLaptopBattery || d.type === UPowerDeviceType.LinePower || !d.isPresent)
            return false;
        if (showAllPeripherals)
            return true;
        if (d.type === UPowerDeviceType.GamingInput)
            return true;
        return /controller|dualsense|dualshock|gamepad|joystick|steam/i.test((d.model ?? "") + " " + (d.nativePath ?? ""));
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

        // -1 means "never moved": use the default corner.
        property real cardX: -1
        property real cardY: -1

        category: "Plugin_controller-battery"

        onCardXChanged: root.place()
        onCardYChanged: root.place()
    }

    function clampX(v: real): real {
        return Math.max(0, Math.min(root.width - card.width, v));
    }

    function clampY(v: real): real {
        return Math.max(0, Math.min(root.height - card.height, v));
    }

    // True while a finished drag is being written to the settings. Saving x
    // fires its change handler before y has been saved; without this guard
    // that handler would snap the card back to the default height.
    property bool saving: false

    function place(): void {
        if (dragHandler.active || saving)
            return;
        const dx = alignRight ? root.width - card.width - edgeMargin : edgeMargin;
        const dy = alignBottom ? root.height - card.height - verticalMargin : verticalMargin;
        card.x = clampX(store.cardX >= 0 ? store.cardX : dx);
        card.y = clampY(store.cardY >= 0 ? store.cardY : dy);
    }

    function resetPosition(): void {
        saving = true;
        store.cardX = -1;
        store.cardY = -1;
        saving = false;
        place();
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
        color: Qt.alpha(root.surfaceColor, 0.92)

        onWidthChanged: root.place()
        onHeightChanged: root.place()

        HoverHandler {
            cursorShape: Qt.OpenHandCursor
        }

        DragHandler {
            id: dragHandler

            target: card
            xAxis.minimum: 0
            xAxis.maximum: Math.max(0, root.width - card.width)
            yAxis.minimum: 0
            yAxis.maximum: Math.max(0, root.height - card.height)
            cursorShape: Qt.ClosedHandCursor

            onActiveChanged: {
                if (!active) {
                    root.saving = true;
                    store.cardX = card.x;
                    store.cardY = card.y;
                    root.saving = false;
                }
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

                    readonly property int cellWidth: root.ringSize + 20
                    readonly property real level: modelData.percentage
                    readonly property bool charging: modelData.state === UPowerDeviceState.Charging
                        || modelData.state === UPowerDeviceState.PendingCharge
                    // Finished charging (e.g. a controller resting on its dock).
                    readonly property bool full: modelData.state === UPowerDeviceState.FullyCharged
                    readonly property bool low: level <= root.lowThreshold && !charging
                    readonly property color ringColor: low ? root.warnColor : root.accentColor

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
                        width: entry.cellWidth
                        height: root.ringSize

                        Shape {
                            id: ring

                            width: root.ringSize
                            height: root.ringSize
                            anchors.horizontalCenter: parent.horizontalCenter
                            preferredRendererType: Shape.CurveRenderer

                            ShapePath {
                                strokeColor: root.trackColor
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
                            source: root.iconFor(entry.modelData)
                            sourceSize: Qt.size(width * 2, height * 2)
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            mipmap: true

                            layer.enabled: root.tintIcons
                            layer.effect: MultiEffect {
                                colorization: 1.0
                                colorizationColor: entry.low ? root.warnColor : root.secondaryColor
                            }
                        }
                    }

                    Text {
                        width: entry.cellWidth
                        horizontalAlignment: Text.AlignHCenter
                        text: Math.round(entry.level * 100) + "%" + ((entry.charging || entry.full) ? " 🗲" : "")
                        color: entry.low ? root.warnColor : root.secondaryColor
                        font.pixelSize: 15
                        font.bold: true
                        lineHeight: 20
                        lineHeightMode: Text.FixedHeight
                        verticalAlignment: Text.AlignVCenter
                    }

                    Text {
                        visible: root.showLabels
                        width: entry.cellWidth
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideRight
                        text: root.labelFor(entry.modelData)
                        color: root.subtleColor
                        font.pixelSize: 12
                    }
                }
            }
        }
    }
}
