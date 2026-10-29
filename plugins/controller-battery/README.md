This is a plugin for [caelestia-kde](https://github.com/ladybug-me/caelestia-kde) to show the battery level of PS5, Steam or any other controllers.

*Not affiliated with or endorsed by Sony Interactive Entertainment or Valve. PlayStation, PS5, DualSense and Steam are trademarks of their respective owners.*

<img width="3840" height="2160" alt="image" src="https://github.com/user-attachments/assets/08858252-2199-4fab-81d9-9fc9a56f5df1" />

## Installation

Requires [caelestia-kde](https://github.com/ladybug-me/caelestia-kde) (Plasma 6, Quickshell).

Either install from the git URL in Nexus > Plugins, or manually:

```bash
git clone https://github.com/verbrannter-toast/controller-battery.git ~/.config/caelestia/plugins/controller-battery
caelestia shell -d
```

Then enable the plugin in Nexus > Plugins if it isn't already.

## Customizing

The panel is movable across the whole desktop and saves its position across reboots. You can reset the position to the default by double clicking it.
[DesktopWidget.qml](https://github.com/verbrannter-toast/controller-battery/blob/main/DesktopWidget.qml) features a ton of customizing options between line 32 and 56 which should make adjusting to your preference easy. This is also where you can adjust the default position that double clicking resets to. If you want to use different icons to the default ones just replace them with the exact file name in `/icons`. 
When just one or no controller is connected the panel simply resizes or hides completely.

A 🗲 icon next to the percentage means the controller is charging. If the controller finished charging, the 🗲 icon stays for as long as it is still plugged in.

## Known limitations

### Steam controller only shows up while Steam is closed

The widget reads battery levels from UPower, which gets them from the kernel. For a wireless Steam controller (tested with the Steam Controller Puck, `28de:1304`), the kernel's `hid-steam` driver exposes a battery only while the Steam client is **not** running. As far as I can tell, the driver steps aside when Steam opens the controller directly (Steam Input), and the battery entry disappears until Steam is closed.

- **Steam running:** the Steam controller is missing from the widget (and from `upower -e`). Steam's own UI still shows its battery level.
- **Steam closed:** the controller appears with its battery level as soon as it is awake and connected.
- **PS5 (DualSense):** not affected; it is exposed by the `hid-playstation` driver whether or not Steam is running.

To check what your system exposes:

```bash
upower -e
ls /sys/class/power_supply/
```

If a controller is not listed there, the widget cannot show it. This is a limitation of the kernel/UPower side, not of the widget.
