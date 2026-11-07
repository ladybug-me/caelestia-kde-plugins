# Controller Battery Bar

A plugin for [caelestia-kde](https://github.com/ladybug-me/caelestia-kde) that
shows the battery level of connected game controllers (PS5, Steam, any gamepad
the kernel exposes) as a small ring in the bar.

The ring empties as the battery drains, the percentage sits in the middle, and
the colour tells you at a glance how much is left:

| Level | Colour |
| --- | --- |
| above 30%, or plugged in | green |
| 10% to 30% | amber |
| below 10% | error (red) |

Clicking the ring opens a popout with one card per controller: name, charge
state, energy and capacity (Wh), power draw (W), time until full or empty,
battery health, MAC address and the sysfs device name.

The widget only lives in the panel and hides itself when no controller is
connected. It is meant as a more graphical, colour-coded alternative to the
generic battery indicator the system shows, and it is visible while playing
fullscreen games, when the desktop is not.

*Not affiliated with or endorsed by Sony Interactive Entertainment or Valve.
PlayStation, PS5, DualSense and Steam are trademarks of their respective
owners.*

## How it finds controllers

Battery data comes from UPower, which reads it from the kernel, so a controller
only shows up while the kernel exposes a battery for it. A wireless Steam
controller can disappear from the list while the Steam client is running,
because Steam takes the device over.
