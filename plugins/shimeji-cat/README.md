# Shimeji Cat

A shimeji-style desktop pet for Caelestia. A small cat lives on your desktop,
wanders around, sits on the taskbar, and can be picked up and tossed with the
mouse.

Ported from the shimeji module in
[ladybug-me/caelestia-kde](https://github.com/ladybug-me/caelestia-kde).

## Requirements

- Caelestia shell 2.3 or newer.

## Sprites

The plugin bundles the classic shimeji sprite set (`shime1.png` through
`shime46.png`, 128x128 px) in `sprites/` and uses them by default. The default
`spritePath` is `sprites/`, resolved relative to this plugin folder.

To use your own frames, edit `spritePath` in `ShimejiCat.qml`:

- An absolute path like `/home/you/sprites/`.
- A home-relative path like `~/sprites/`.
- A shell-root-relative path like `root:/assets/shimeji/pusheen/`.

## Configuration

Edit the constants at the top of `ShimejiCat.qml`:

- `spritePath` - directory containing the sprite frames.
- `petCount` - number of cats per screen (default 1).

## License

GPL-3.0-or-later. See `LICENSE`. Original code by the Caelestia contributors.
