# Ultralight Web Cursor

A KWin effect that replaces the system cursor with an animated HTML/CSS/JS
cursor rendered by [Ultralight](https://ultralig.ht/). The cursor is a real
HTML page, so themes are plain folders with an `index.html` (see
`contents/WebCursor/`).

The plugin is source-only and ships both halves in one folder:

- the KWin effect (`ultralightwebcursor`), built from C++ with CMake, and
- the Quickshell settings UI (`main.qml` + `qml/`), which the shell loads
  because the manifest sets `"ui": "main.qml"`, and which registers the
  `Meta+Ctrl+C` shortcut.

Both halves share the `webCursor` section in
`~/.config/caelestia/shell.json`.

The store manifest is `type: kwineffect`, matching the KWin effect package and
its `kwineffect.kpluginId` (`ultralightwebcursor`).

## Dependencies

**Building / running the effect** (Arch package names in parentheses):

- KDE Frameworks 6 / KWin development headers (`kwin6-devel`)
- Qt 6 (`qt6-base`), Extra CMake Modules (`extra-cmake-modules`)
- `epoxy`
- The Ultralight SDK ([ultralig.ht](https://ultralig.ht/)) - not bundled on purpose:
  this store is source-only and the `.so` binaries are too large to ship.
  Install and extract a compatible SDK yourself. Its directory must be named
  **`ThirdParty`** and live directly inside this plugin folder. The required
  layout is `ThirdParty/bin/*.so`, `ThirdParty/include`, and
  `ThirdParty/resources`. Use the provided script to copy an already extracted
  SDK into that exact location; it never downloads anything:

  ```sh
  ./install-sdk.sh /path/to/extracted-ultralight-sdk
  ```

  The build never downloads an SDK or accesses the network.

**Running the settings UI** (the shell side):

- A Caelestia shell with Quickshell, plus the shell modules the QML imports:
  `Quickshell.Io`/`Wayland`, `qs.services(.api)`, and `Caelestia.Config`
  (used only for the Material-3 palette in `qml/Colors.qml`). No KDE/Qt dev
  headers are needed to run this half.

## Getting the effect installed

The effect can be built in two ways:

### 1. Automatically, from the settings UI

When `main.qml` starts it checks whether the bundled C++ project was already
built (`<plugin dir>/build` with a compiled `ultralightwebcursor.so`). When it
is missing and `webCursor.build.auto` is true (the default), it runs
`cmake -S . -B build` and `cmake --build build` in the background. Install the
Ultralight SDK yourself before using this option. By default CMake expects it
in `ThirdParty/`; if it is stored elsewhere, configure the build with
`-DULTRALIGHT_ROOT=/path/to/ultralight-sdk`. Build progress/errors are shown at
the top of the settings panel.

The install that follows is automatic too, unless `webCursor.build.autoInstall`
is false. Once the build output exists the plugin runs `pkexec cmake --install`
for you (one polkit password prompt), remembers the installed artifact in
`$XDG_CACHE_HOME/caelestia/webcursor/installed-fingerprint`, and enables the
effect in KWin. Because the fingerprint records which artifact is live, later
shell starts do not prompt again; a rebuilt artifact (e.g. after a plugin
update) installs once more. Dismissing the password prompt turns
auto-install off and leaves the panel's Install button for retries.

### 2. Manually, from a checkout

```sh
cmake -B build -S .
cmake --build build
sudo cmake --install build
```

This effect is a build-required KWin plugin rather than a prebuilt KPackage.
The store's source-only installer must not pass it directly to
`kpackagetool6`; use the built-in settings UI or the CMake commands above to
build and install the compiled effect.

`cmake --install` copies:

- the effect plugin `ultralightwebcursor` into KWin's effect plugin dir,
- the Ultralight runtime libraries into `/usr/lib/webkde_core`,
- the built-in cursor themes and Ultralight resources into `/usr/share/caelestia/webcursor`.

## Settings UI (Quickshell)

The settings UI ships in this folder. The shell loads `main.qml` because
`metadata.json` sets `"ui": "main.qml"`; there is no separate settings plugin
to install.

The QML half mirrors the web cursor UI/service/config from
[`caelestia-kde`](https://github.com/LuYishan-4/caelestia-dots-kde):

| Source (caelestia-kde) | Port in this folder |
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
  shortcut (default `Meta+Ctrl+C`, override with the `webCursor.shortcut`
  key in `shell.json`); close it with `Esc` or by clicking the backdrop.

## Configuration

Everything lives in the `webCursor` section of
`~/.config/caelestia/shell.json`:

```jsonc
{
  "webCursor": {
    "shortcut": "Meta+Ctrl+C",   // toggle key for the settings overlay
    "build": {
      "auto": true,               // cmake-build the effect when it is missing
      "autoInstall": true         // pkexec-install + enable it once built
    },
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

- `metadata.json` - Caelestia plugin store manifest (`type: kwineffect`, `ui: main.qml`).
- `metadata.desktop` - KWin effect metadata.
- `CMakeLists.txt` - effect build script (source-only; `build/` is git-ignored).
- `install-sdk.sh` - manually copies an already extracted SDK into `ThirdParty/`; it never downloads a file.
- `contents/` - effect source (KPackage `contents/` layout) and bundled cursor themes.
- `main.qml`, `qml/` - the Quickshell settings UI the shell loads.
- `assets/` - icon and banner artwork for the store listing and the panel.

Validate the store locally with:

```sh
python scripts/validate.py
```
