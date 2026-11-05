# Game mode indicator

A green gamepad appears in the bar while game mode is active, and every change
is announced with the shell's own toast (same style and position as the
built-in game mode toast).

It reacts to both game mode sources:

- Caelestia's own game mode (the quick toggle). In that case the shell already
  shows its toast, so the plugin only draws the icon.
- Feral GameMode (`gamemoded`), i.e. anything launched with `gamemoderun`.
  Here the plugin also announces when it starts and stops.

## Installation

Add the **Game mode indicator** widget to a bar section from the bar editor
(Nexus, then panels, then the bar you want). No configuration needed.

`gamemoded` is optional: without it the plugin simply follows Caelestia's game
mode.
