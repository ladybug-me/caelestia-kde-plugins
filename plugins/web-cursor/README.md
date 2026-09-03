># Ultralight Web Cursor

A KWin effect that replaces the system cursor with an animated HTML/CSS/JS
cursor rendered by [Ultralight](https://ultralig.ht/), plus a Quickshell
settings UI to manage it. The cursor is a real HTML page, so themes are plain
folders with an `index.html` (see `contents/WebCursor/`).

The plugin is **source-only** and ships two halves in the same folder:

| Half | Purpose |
| --- | --- |
| `contents/` + `CMakeLists.txt` | The KWin effect (`ultralightwebcursor`), built from C++ with CMake. |
| `main.qml` + `qml/` | A Quickshell settings UI (Nexus-style page + theme/config service), ported from [`caelestia-dots-kde`](https://github.com/LuYishan-4/caelestia-dots-kde). |

Both halves share one config document: `webCursor` in
`~/.config/caelestia/shell.json`. The effect reads it from its own process; the
UI writes it and then pokes the effect over D-Bus.

The store manifest is `type: quickshell`, which is what makes the Caelestia
shell's plugin loader instantiate `main.qml` and the settings UI (the folder
keeps the KWin package files — `metadata.desktop` and `kwineffect.kpluginId`
`ultralightwebcursor` — so the effect can still be built/installed from the
same source via its CMakeLists).

## Dependencies

**Building / running the effect** (Arch package names in parentheses):

- KDE Frameworks 6 / KWin development headers (`kwin6-devel`)
- Qt 6 (`qt6-base`), Extra CMake Modules (`extra-cmake-modules`)
- `epoxy`
- The **Ultralight SDK** (https://ultralig.ht/) — *not* bundled on purpose:
  this store is source-only and the `.so` binaries are too large to ship.
  `CMakeLists.txt` looks for an extracted SDK at `ULTRALIGHT_ROOT` (default
  `./ThirdParty`, layout `bin/*.so` + `include` + `resources`). If none is
  found there, it downloads the SDK automatically from the ThirdParty GitHub
  mirror (<https://github.com/LuYishan-4/ThirdParty>, branch `main` by
  default) into the build directory.

  Automatic download can be tuned or disabled with cache variables:

  ```sh
  cmake -B build -S . \
    -DULTRALIGHT_REPO=LuYishan-4/ThirdParty \
    -DULTRALIGHT_REF=main \
    -DULTRALIGHT_DOWNLOAD_URL="https://..." \
    -DULTRALIGHT_DOWNLOAD=OFF -DULTRALIGHT_ROOT=/path/to/ultralight-sdk
  ```

  Anything you keep inside this folder (e.g. a copy of the SDK at
  `ThirdParty/`) is git-ignored and must never be committed.

**Running the settings UI** (the shell side):

- A Caelestia shell with Quickshell, plus the shell modules the QML imports:
  `Quickshell.Io`/`Wayland`, `qs.services(.api)`, and `Caelestia.Config`
  (used only for the Material-3 palette in `qml/Colors.qml`). No KDE/Qt dev
  headers are needed to run this half.

## Getting the effect installed

The effect can be built in two ways:

### 1. Automatically, from the settings UI

When `main.qml` starts it checks whether the bundled C++ project was already
built (`<plugin dir>/build` with a compiled `ultralightwebcursor.so`). If not,
it runs `cmake -S . -B build` and `cmake --build build` in the background
(first run also fetches the Ultralight SDK). Build progress/errors are shown
at the top of the settings panel. This only *builds*; to make KWin load the
effect you still need to install it once:

```sh
sudo cmake --install build
```

### 2. Manually, from a checkout

```sh
cmake -B build -S .          # fetches the Ultralight SDK automatically if needed
cmake --build build
sudo cmake --install build
```

`cmake --install` copies:

- the effect plugin `ultralightwebcursor` into KWin's effect plugin dir,
- the Ultralight runtime libraries into `/usr/lib/webkde_core`,
- the built-in cursor themes and Ultralight resources into `/usr/share/caelestia/webcursor`.

## Settings UI (Quickshell)

The QML half mirrors the web cursor UI/service/config from
[`caelestia-dots-kde`](https://github.com/LuYishan-4/caelestia-dots-kde):

| Source (caelestia-dots-kde) | Port in this folder |
| --- | --- |
| `shell/plugin/src/Caelestia/Config/webcursorconfig.hpp` | `qml/Config.qml` (`WebCursorConfig`) |
| `shell/services/WebCursor.qml` | `qml/WebCursorManager.qml` |
| `shell/modules/nexus/pages/desktop/WebCursorPage.qml` | `qml/settings/WebCursorSettingsPanel.qml` |

QML layout follows the `wallpaper-selector` plugin: `main.qml` is the entry
`Scope`, `qml/` is the module (with `qmldir`, `singleton Style`, `Colors`,
`StyledToolTip`), and shell-agnostic controls live in `qml/components/`.

What the UI can do:

- **Enable / disable** the effect and **apply themes**, sizes and the
  blacklist — everything is written to `shell.json` first, then pushed to the
  running effect over D-Bus (`busctl`), never by reconfiguring `kwinrc`
  (which would reload/unload effect plugins and crash KWin).
- **Theme management**: on load, built-in themes under
  `/usr/share/caelestia/webcursor` are symlinked into the user themes dir; the
  UI lists them, shows `CursorData.json` metadata (`IconPath`, `Author`,
  `describe`, minimum size), and can **upload a theme folder** (copied to the
  user dir), open a theme folder, or remove uploaded themes.
- One settings overlay per screen, toggled with the `webcursor_settings`
  shortcut (default `Meta+Shift+C`, override with the `webCursor.shortcut`
  key in `shell.json`); close it with `Esc` or by clicking the backdrop.

## Configuration

Everything lives in the `webCursor` section of
`~/.config/caelestia/shell.json`:

```jsonc
{
  "webCursor": {
    "shortcut": "Meta+Shift+C",   // toggle key for the settings overlay
    "build": { "auto": true },    // auto cmake-build the effect on UI start
    "cursor": {
      "enabled": true,
      "width": 128,
      "height": 128,
      "selectTheme": "variant4-ciallo",
      "themesDir": "~/.config/caelestia/webcursor",
      "blacklist": []
    }
  }
}
```

`webCursor.cursor` is the effect's own schema (defaults mirror
`webcursorconfig.hpp`). Missing keys are created with the defaults above;
unrelated sections of `shell.json` are preserved. The UI watches the file, so
manual edits are picked up automatically.

Settings can also be applied live over D-Bus:

```sh
busctl --user call org.kde.KWin /UltralightCursor org.kde.kwin.KWin.KwinCursorEffect reloadHtml
```

## Enable

After the effect is installed, either enable it in KWin's config:

```sh
kwriteconfig6 --file kwinrc --group Plugins --key ultralightwebcursorEnabled true
qdbus org.kde.KWin /Effects org.kde.kwin.Effects.reconfigure
```

or at runtime through D-Bus (what the UI's enable switch does):

```sh
busctl --user call org.kde.KWin /Effects org.kde.kwin.Effects loadEffect s ultralightwebcursor
busctl --user call org.kde.KWin /UltralightCursor org.kde.kwin.KWin.KwinCursorEffect enable
```

## Store layout

- `metadata.json` - Caelestia plugin store manifest (`type: quickshell`; keeps `kwineffect.kpluginId` for the bundled effect).
- `metadata.desktop` - KWin effect metadata.
- `CMakeLists.txt` - effect build script (source-only; `build/` is git-ignored).
- `contents/` - effect source (KPackage `contents/` layout) and bundled cursor themes.
- `main.qml` + `qml/` - Quickshell settings UI (`main.qml` is the entry `Scope`).

Validate the store locally with:

```sh
python scripts/validate.py
```
