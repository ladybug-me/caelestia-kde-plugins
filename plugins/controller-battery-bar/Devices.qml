pragma ComponentBehavior: Bound

import QtQuick
import Quickshell.Services.UPower

// Controller detection for the bar widget.
//
// UPower only exposes a controller while the kernel has a battery for it: a
// wireless Steam controller disappears from the list while the Steam client is
// running, because Steam takes the device over.
QtObject {
    id: root

    // Also list mice, keyboards, headsets...
    property bool showAllPeripherals: false

    // First matching rule wins. "match" is tested against the device's sysfs
    // path and model. These rules also decide what counts as a controller, so a
    // device matching none of them is ignored unless showAllPeripherals is on.
    property var nameRules: [
        { match: /steam/i, name: "Steam Controller" },
        { match: /ps-controller|dualsense|dualshock/i, name: "PS5 Controller" },
        { match: /controller|gamepad|joystick/i, name: "Controller" }
    ]

    readonly property var controllers: UPower.devices.values.filter(d => root.isController(d))

    function isController(d: var): bool {
        if (d.isLaptopBattery || d.type === UPowerDeviceType.LinePower || !d.isPresent)
            return false;
        return root.showAllPeripherals || d.type === UPowerDeviceType.GamingInput || root.ruleFor(d) !== null;
    }

    function ruleFor(d: var): var {
        const id = (d.nativePath ?? "") + " " + (d.model ?? "");
        return root.nameRules.find(r => r.match.test(id)) ?? null;
    }

    function labelFor(d: var): string {
        return root.ruleFor(d)?.name ?? (d.model || qsTr("Controller"));
    }

    function percentFor(d: var): int {
        return Math.round((d.percentage ?? 0) * 100);
    }

    function chargingFor(d: var): bool {
        return d.state === UPowerDeviceState.Charging || d.state === UPowerDeviceState.PendingCharge;
    }

    // A docked controller is not charging but is still plugged in.
    function pluggedInFor(d: var): bool {
        return root.chargingFor(d) || d.state === UPowerDeviceState.FullyCharged;
    }
}
