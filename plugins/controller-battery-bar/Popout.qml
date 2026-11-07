pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import QtQuick.Shapes
import Quickshell
import Quickshell.Services.UPower
import Caelestia.Config
import qs.components
import qs.components.controls
import qs.services
import "."

// Popout listing every connected controller with the same data the desktop
// widget shows, plus a few extras UPower exposes (energy, draw, health, sysfs).
ColumnLayout {
    id: root

    // Same thresholds the bar widget uses.
    readonly property real warnLevel: 0.3
    readonly property real lowLevel: 0.1

    readonly property var controllers: devices.controllers

    width: Math.max(340, cardLayout.implicitWidth)
    spacing: Tokens.spacing.medium

    function has(d: var, key: string): bool {
        const v = d ? d[key] : undefined;
        return typeof v === "number" && isFinite(v) && v > 0;
    }

    function watt(v: double): string {
        return (Math.round(v * 10) / 10) + " W";
    }

    function wattHour(v: double): string {
        return (Math.round(v * 10) / 10) + " Wh";
    }

    function duration(seconds: double): string {
        return Units.formatDurationShort(seconds, qsTr("Calculating..."));
    }

    // The kernel names the sysfs node after the device's MAC
    // (hid-aa:bb:cc:dd:ee:ff-battery), so it is the only place UPower gives it.
    function macFor(d: var): string {
        const m = String(d?.nativePath ?? "").match(/([0-9a-f]{2}:){5}[0-9a-f]{2}/i);
        return m ? m[0].toUpperCase() : "";
    }

    function stateFor(d: var): string {
        if (devices.chargingFor(d))
            return qsTr("Charging");
        if (d.state === UPowerDeviceState.FullyCharged)
            return qsTr("Fully charged");
        if (d.state === UPowerDeviceState.Discharging)
            return qsTr("Discharging");
        return qsTr("Unknown");
    }

    function colourFor(d: var): color {
        if (devices.pluggedInFor(d))
            return "#4ade80";
        const level = d.percentage ?? 0;
        return level > root.warnLevel ? "#4ade80" : level > root.lowLevel ? "#fbbf24" : Colours.palette.m3error;
    }

    component DetailRow: RowLayout {
        id: row

        required property string label
        required property string value

        Layout.fillWidth: true
        spacing: Tokens.spacing.small

        StyledText {
            text: row.label
            color: Colours.palette.m3onSurfaceVariant
            font: Tokens.font.body.small
        }

        Item {
            Layout.fillWidth: true
        }

        StyledText {
            text: row.value
            font: Tokens.font.body.small
            horizontalAlignment: Text.AlignRight
            elide: Text.ElideMiddle
            Layout.maximumWidth: root.width * 0.55
        }
    }

    Devices {
        id: devices
    }

    StyledText {
        visible: root.controllers.length === 0
        text: qsTr("No controller connected")
        color: Colours.palette.m3onSurfaceVariant
        font: Tokens.font.body.medium
    }

    ColumnLayout {
        id: cardLayout

        Layout.fillWidth: true
        spacing: Tokens.spacing.medium

        Repeater {
            model: ScriptModel {
                values: root.controllers
            }

            StyledRect {
                id: card

                required property var modelData

                readonly property real level: modelData.percentage ?? 0
                readonly property color levelColour: root.colourFor(card.modelData)
                readonly property string mac: root.macFor(card.modelData)

                Layout.fillWidth: true
                implicitWidth: details.implicitWidth + Tokens.padding.large * 2
                implicitHeight: details.implicitHeight + Tokens.padding.large * 2
                color: Colours.palette.m3surfaceContainerHigh
                radius: Tokens.rounding.large

                ColumnLayout {
                    id: details

                    anchors.fill: parent
                    anchors.margins: Tokens.padding.large
                    spacing: Tokens.spacing.small

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: Tokens.spacing.medium

                        Shape {
                            id: ring

                            Layout.preferredWidth: 44
                            Layout.preferredHeight: 44
                            preferredRendererType: Shape.CurveRenderer

                            ShapePath {
                                strokeColor: Colours.palette.m3surfaceContainerHighest
                                strokeWidth: 4
                                fillColor: "transparent"

                                PathAngleArc {
                                    centerX: ring.width / 2
                                    centerY: ring.height / 2
                                    radiusX: (ring.width - 4) / 2
                                    radiusY: (ring.height - 4) / 2
                                    startAngle: -90
                                    sweepAngle: 360
                                }
                            }

                            ShapePath {
                                strokeColor: card.levelColour
                                strokeWidth: 4
                                fillColor: "transparent"
                                capStyle: ShapePath.RoundCap

                                PathAngleArc {
                                    centerX: ring.width / 2
                                    centerY: ring.height / 2
                                    radiusX: (ring.width - 4) / 2
                                    radiusY: (ring.height - 4) / 2
                                    startAngle: -90
                                    sweepAngle: Math.max(0.5, Math.min(359.9, 360 * card.level))
                                }
                            }

                            StyledText {
                                anchors.centerIn: parent
                                text: Math.round(card.level * 100)
                                color: card.levelColour
                                font: Tokens.font.label.builders.medium.size(13).build()
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 0

                            StyledText {
                                Layout.fillWidth: true
                                text: devices.labelFor(card.modelData)
                                font: Tokens.font.title.small
                                elide: Text.ElideRight
                            }

                            StyledText {
                                Layout.fillWidth: true
                                text: root.stateFor(card.modelData)
                                color: card.levelColour
                                font: Tokens.font.body.small
                                elide: Text.ElideRight
                            }
                        }

                        MaterialIcon {
                            text: devices.pluggedInFor(card.modelData) ? "battery_charging_full" : "sports_esports"
                            color: card.levelColour
                            fontStyle: Tokens.font.icon.medium
                        }
                    }

                    DetailRow {
                        visible: root.has(card.modelData, "energy") && root.has(card.modelData, "energyCapacity")
                        label: qsTr("Energy")
                        value: root.wattHour(card.modelData.energy ?? 0) + " / " + root.wattHour(card.modelData.energyCapacity ?? 0)
                    }

                    DetailRow {
                        visible: root.has(card.modelData, "changeRate")
                        label: qsTr("Power draw")
                        value: root.watt(card.modelData.changeRate ?? 0)
                    }

                    DetailRow {
                        visible: devices.chargingFor(card.modelData) && root.has(card.modelData, "timeToFull")
                        label: qsTr("Until full")
                        value: root.duration(card.modelData.timeToFull ?? 0)
                    }

                    DetailRow {
                        visible: !devices.pluggedInFor(card.modelData) && root.has(card.modelData, "timeToEmpty")
                        label: qsTr("Remaining")
                        value: root.duration(card.modelData.timeToEmpty ?? 0)
                    }

                    DetailRow {
                        visible: card.modelData.healthSupported === true && root.has(card.modelData, "healthPercentage")
                        label: qsTr("Health")
                        value: Math.round(card.modelData.healthPercentage ?? 0) + "%"
                    }

                    DetailRow {
                        visible: card.mac !== ""
                        label: qsTr("MAC address")
                        value: card.mac
                    }

                    DetailRow {
                        visible: (card.modelData.nativePath ?? "") !== ""
                        label: qsTr("Device")
                        value: String(card.modelData.nativePath ?? "").replace("/sys/class/power_supply/", "")
                    }
                }
            }
        }
    }
}
