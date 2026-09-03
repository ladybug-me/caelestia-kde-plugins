// QML port of `caelestia-dots-kde/shell/services/WebCursor.qml`.
//
// Adaptations for the plugin environment:
//   * the shell's `GlobalConfig.webCursor.cursor` became the `config`
//     property (a `WebCursorConfig` instance, see Config.qml) injected by the
//     plugin entry point,
//   * user/system theme detection is folded into a single listing pass
//     (instead of one `readlink` subprocess per theme).
//
// Everything else mirrors the original: system themes are symlinked into the
// user themes dir, uploads copy folders, and the KWin effect is controlled
// through its D-Bus interface (busctl), never through `kwinrc` reconfigure.
import QtQuick
import Quickshell
import Quickshell.Io

QtObject {
    id: root

    // Instance of WebCursorConfig (Config.qml); assigned by main.qml.
    property var config: null

    // Callback used by main.qml to hide the fullscreen settings overlay before
    // a pkexec (polkit) prompt is shown, otherwise the password dialog ends up
    // covered and unreachable.
    property var windowHider: null

    function _hideWindow() {
        if (typeof root.windowHider === "function") root.windowHider()
    }

    signal themeUploadFinished(bool success, string error)
    signal themeRemoveFinished(bool success, string error)

    property var themeList: []
    property var _themeKinds: ({})
    property var _themeDetails: ({})
    property string statusMessage: ""

    readonly property string systemThemesDir: "/usr/share/caelestia/webcursor"

    function userThemesDir(): string {
        return root.config ? root.config.themesDir : ""
    }

    // ---- built-in themes -> user dir linking ------------------------------
    function ensureInitialized() {
        linkProc.command = ["sh", "-c",
            'user="$1"; sys="$2"; ' +
            'mkdir -p "$user" || exit 1; ' +
            'for f in "$user"/*; do ' +
            '  [ -e "$f" ] && continue; ' +
            '  [ -L "$f" ] && rm -f -- "$f"; ' +
            'done; ' +

            '[ -d "$sys" ] || exit 0; ' +
            'for d in "$sys"/*/; do ' +
            '  [ -d "$d" ] || continue; ' +
            '  n=$(basename "$d"); ' +
            '  [ -f "$d/CursorData.json" ] && [ -f "$d/index.html" ] || continue; ' +
            '  target="$user/$n"; ' +
            '  [ -e "$target" ] || [ -L "$target" ] || ln -s -- "$d" "$target"; ' +
            'done',
            "--", root.userThemesDir(), root.systemThemesDir]
        linkProc.running = true
    }

    property Process linkProc: Process {
        id: linkProc
        command: []
        onExited: () => { root.refreshThemes() }
    }

    // ---- theme listing -----------------------------------------------------
    // Lists every valid theme folder in the user themes dir. Each output line
    // carries the theme kind plus its CursorData.json (base64), so details and
    // icons can be shown without XMLHttpRequest on file:// (disabled in the
    // shell runtime):
    //   <name>|<kind>|<base64-CursorData.json>
    function refreshThemes(): void {
        listProc.command = ["sh", "-c",
            'dir="$1"; [ -d "$dir" ] || exit 0; ' +
            'for d in "$dir"/*/; do ' +
            '  [ -d "$d" ] || continue; ' +
            '  n=$(basename "$d"); ' +
            '  [ -f "$d/CursorData.json" ] && [ -f "$d/index.html" ] || continue; ' +
            '  if [ -L "$d" ]; then k="system"; else k="user"; fi; ' +
            '  b64=$(base64 -w0 "$d/CursorData.json" 2>/dev/null); ' +
            '  printf "%s|%s|%s\\n" "$n" "$k" "$b64"; ' +
            'done',
            "--", root.userThemesDir()]
        listProc.running = true
    }

    property Process listProc: Process {
        id: listProc
        stdout: StdioCollector {
            id: listStdout
            onStreamFinished: {
                const names = []
                const kinds = {}
                const details = {}
                const base = root.userThemesDir()
                const lines = (listStdout.text || "").split("\n")
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim()
                    if (!line) continue
                    const parts = line.split("|")
                    if (parts.length < 2) continue
                    const name = parts[0]
                    const kind = parts[1]
                    const b64 = parts.length > 2 ? parts.slice(2).join("|") : ""
                    names.push(name)
                    kinds[name] = kind

                    const meta = { iconPath: "", author: qsTr("Unknown"), describe: "", minWidth: 128, minHeight: 128 }
                    try {
                        const raw = root._fromBase64(b64)
                        const obj = raw ? JSON.parse(raw) : null
                        if (obj) {
                            if (obj.IconPath) meta.iconPath = "file://" + base + "/" + name + "/" + obj.IconPath
                            meta.author = obj.Author || meta.author
                            meta.describe = obj.describe || ""
                            meta.minWidth = parseInt(obj.minWidth, 10) || 128
                            meta.minHeight = parseInt(obj.minHeight, 10) || 128
                        }
                    } catch (e) { /* keep defaults */ }
                    details[name] = meta
                }
                root._themeDetails = details
                root._themeKinds = kinds
                root.themeList = names
            }
        }
    }

    function themePath(name: string): string {
        if (!name || name.indexOf("/") !== -1 || name.indexOf("\\") !== -1)
            return ""
        return `${root.userThemesDir()}/${name}`
    }

    // true when the theme was uploaded by the user (a real folder); false for
    // symlinked built-in themes.
    function isUserTheme(name: string): bool {
        return root._themeKinds[name] === "user"
    }

    function _fromBase64(b64) {
        // Minimal base64 -> byte string decoder (CursorData.json is ASCII in
        // practice). Avoids the QML XMLHttpRequest file:// restriction.
        if (!b64) return ""
        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        let out = ""
        let buffer = 0
        let bits = 0
        for (let i = 0; i < b64.length; i++) {
            if (b64[i] === "=") break
            const val = chars.indexOf(b64[i])
            if (val < 0) continue
            buffer = (buffer << 6) | val
            bits += 6
            if (bits >= 8) {
                bits -= 8
                out += String.fromCharCode((buffer >> bits) & 0xFF)
            }
        }
        return out
    }

    function getThemeDetails(name: string): var {
        return root._themeDetails[name] || { iconPath: "", author: qsTr("Unknown"), describe: "", minWidth: 128, minHeight: 128 }
    }

    function openThemeFolder(name: string) {
        const path = root.themePath(name)
        if (path) Quickshell.execDetached(["xdg-open", path])
    }

    // ---- upload / remove ---------------------------------------------------
    function uploadTheme(srcPath: string, themeName: string) {
        const src = String(srcPath || "").replace(/^file:\/\//, "")
        if (!src) {
            root.statusMessage = qsTr("Invalid folder")
            root.themeUploadFinished(false, root.statusMessage)
            return
        }
        let name = String(themeName || "").trim()
        if (!name) {
            const parts = src.split("/").filter(p => p.length > 0)
            name = parts.length > 0 ? parts[parts.length - 1] : ""
        }
        name = name.replace(/[/\\]/g, "") // prevent path traversal
        if (!name) {
            root.statusMessage = qsTr("Invalid folder or theme name")
            root.themeUploadFinished(false, root.statusMessage)
            return
        }

        const dst = `${root.userThemesDir()}/${name}`
        const script =
            'src="$1"; dst="$2"; ' +
            '[ -d "$src" ] || { echo "source is not a directory" >&2; exit 1; }; ' +
            '[ -f "$src/CursorData.json" ] && [ -f "$src/index.html" ] || { echo "selected folder is not a valid cursor theme" >&2; exit 1; }; ' +
            '[ -L "$dst" ] && rm -f -- "$dst"; ' +
            'rm -rf -- "$dst" || exit 1; mkdir -p -- "$dst" || exit 1; ' +
            'cp -r -- "$src"/. "$dst"/ || exit 1'
        uploadProc.command = ["sh", "-c", script, "--", src, dst]
        uploadProc._themeName = name
        uploadProc.running = true
    }

    property Process uploadProc: Process {
        id: uploadProc
        property string _themeName: ""
        stderr: StdioCollector {
            id: uploadStderr
        }
        onExited: code => {
            if (code === 0) {
                // Apply the uploaded theme first (so the refreshed list shows it
                // as active), then rebuild the list and reload the effect once.
                root.setTheme(uploadProc._themeName)
                root.refreshThemes()
                root._requestReload()
                root.statusMessage = qsTr("Theme uploaded and applied")
                root.themeUploadFinished(true, "")
                if (root.config && root.config.installGlobal) {
                    root._installGlobalTheme(uploadProc._themeName)
                }
            } else {
                root.statusMessage = (uploadStderr.text || "").trim() || qsTr("Theme upload failed")
                root.themeUploadFinished(false, root.statusMessage)
            }
        }
    }

    // Optional second copy of an uploaded theme into the system theme dir.
    // Uses pkexec (polkit) so the password prompt is shown by the desktop, not
    // by this plugin.
    function _installGlobalTheme(themeName: string) {
        const src = `${root.userThemesDir()}/${themeName}`
        const script =
            'dir="$1"; name="$2"; src="$3"; ' +
            'mkdir -p -- "$dir" || exit 1; ' +
            'if [ -e "$dir/$name" ]; then echo "already-exists" >&2; exit 2; fi; ' +
            'cp -r -- "$src"/. "$dir/$name" 2>/dev/null || { echo "copy-failed" >&2; exit 3; }'
        sysInstallProc._themeName = themeName
        // Hide the overlay so the polkit password dialog is reachable.
        root._hideWindow()
        sysInstallProc.command = ["pkexec", "sh", "-c", script, "--", root.systemThemesDir, themeName, src]
        sysInstallProc.running = true
    }

    property Process sysInstallProc: Process {
        id: sysInstallProc
        property string _themeName: ""
        stderr: StdioCollector {
            id: sysInstallStderr
        }
        onExited: code => {
            const reason = (sysInstallStderr.text || "").trim()
            if (code === 0) {
                root.statusMessage = qsTr("Theme uploaded and installed system-wide")
            } else if (reason.indexOf("already-exists") !== -1) {
                root.statusMessage = qsTr("Theme uploaded; system copy skipped (already exists)")
            } else {
                root.statusMessage = qsTr("Theme uploaded; system copy %1").arg(
                    reason.length > 0 ? qsTr("failed") : qsTr("cancelled"))
            }
        }
    }
    function removeTheme(themeName: string) {
        if (!themeName) {
            root.themeRemoveFinished(false, qsTr("Empty theme name"))
            return
        }
        const dst = `${root.userThemesDir()}/${themeName}`
        const script =
            'dst="$1"; ' +
            '[ -e "$dst" ] || [ -L "$dst" ] || exit 2; ' +
            '[ -L "$dst" ] && { echo "cannot remove a built-in theme" >&2; exit 3; }; ' +
            'rm -rf -- "$dst"'
        removeProc.command = ["sh", "-c", script, "--", dst]
        removeProc._themeName = themeName
        removeProc.running = true
    }

    property Process removeProc: Process {
        id: removeProc
        property string _themeName: ""
        stderr: StdioCollector {
            id: removeStderr
        }
        onExited: code => {
            if (code === 0) {
                if (root.config && root.config.selectTheme === removeProc._themeName) {
                    root.config.selectTheme = ""
                    root._requestReload()
                }
                root.refreshThemes()
                root.statusMessage = qsTr("Theme removed successfully")
                root.themeRemoveFinished(true, "")
                // A theme installed system-wide needs root to be deleted; only
                // prompt when a copy actually exists there.
                root._maybeRemoveGlobalCopy(removeProc._themeName)
            } else if (code === 3) {
                root.statusMessage = qsTr("Built-in themes cannot be removed")
                root.themeRemoveFinished(false, root.statusMessage)
            } else {
                root.statusMessage = qsTr("Theme not found")
                root.themeRemoveFinished(false, root.statusMessage)
            }
        }
    }

    // If a system-wide copy of the removed theme exists, delete it too. This
    // needs root, so it goes through pkexec; we only prompt when the copy is
    // actually present (checked as the unprivileged user first).
    function _maybeRemoveGlobalCopy(themeName: string) {
        globalCheckProc._themeName = themeName
        globalCheckProc.command = ["sh", "-c",
            '[ -e "$1/$2" ] && echo yes || true', "--", root.systemThemesDir, themeName]
        globalCheckProc.running = true
    }

    property Process globalCheckProc: Process {
        id: globalCheckProc
        property string _themeName: ""
        stdout: StdioCollector {
            id: globalCheckOut
            onStreamFinished: {
                if ((globalCheckOut.text || "").trim() === "yes") {
                    root._removeGlobalCopy(globalCheckProc._themeName)
                }
            }
        }
    }

    function _removeGlobalCopy(themeName: string) {
        const script =
            'dir="$1"; name="$2"; ' +
            '[ -e "$dir/$name" ] || exit 0; ' +
            'rm -rf -- "$dir/$name"'
        globalRemoveProc._themeName = themeName
        // Hide the overlay so the polkit password dialog is reachable.
        root._hideWindow()
        globalRemoveProc.command = ["pkexec", "sh", "-c", script, "--", root.systemThemesDir, themeName]
        globalRemoveProc.running = true
    }

    property Process globalRemoveProc: Process {
        id: globalRemoveProc
        property string _themeName: ""
        stderr: StdioCollector {
            id: globalRemoveErr
        }
        onExited: code => {
            if (code === 0) {
                root.statusMessage = qsTr("Theme removed; system copy also removed")
            } else {
                root.statusMessage = qsTr("Theme removed; system copy %1").arg(
                    (globalRemoveErr.text || "").trim().length > 0 ? qsTr("failed") : qsTr("cancelled"))
            }
        }
    }

    // ---- apply / control the KWin effect -----------------------------------
    function setTheme(themeName: string) {
        if (!root.config) return
        root.config.selectTheme = themeName
    }

    function useTheme(themeName: string) {
        root.setTheme(themeName)
        root._requestReload()
        root.statusMessage = qsTr("Theme applied successfully")
    }

    function reload() {
        root.refreshThemes()
        root.statusMessage = qsTr("Reloaded successfully")
    }

    function enable() {
        if (!root.config) return
        root.config.enabled = true
        kwinToggleProc.command = ["sh", "-c",
            'busctl --user call org.kde.KWin /Effects org.kde.kwin.Effects loadEffect s ultralightwebcursor 2>/dev/null; ' +
            'busctl --user call org.kde.KWin /UltralightCursor org.kde.kwin.KWin.KwinCursorEffect enable 2>/dev/null; ' +
            'true']
        kwinToggleProc.running = true
        root.save()
        root.statusMessage = qsTr("Enabled")
    }

    function disable() {
        if (!root.config) return
        root.config.enabled = false
        kwinToggleProc.command = ["sh", "-c",
            'busctl --user call org.kde.KWin /UltralightCursor org.kde.kwin.KWin.KwinCursorEffect disable 2>/dev/null; ' +
            'true']
        kwinToggleProc.running = true
        // No reloadHtml needed while the effect is off; just persist the state.
        root.config.saveNow()
        root.statusMessage = qsTr("Disabled")
    }

    property Process kwinToggleProc: Process {
        id: kwinToggleProc
        command: []
    }

    // ---- Blacklist (delegates to the config object) ------------------------
    function addBlacklist(app: string) {
        if (root.config) root.config.addBlacklist(app)
    }

    function removeBlacklist(app: string) {
        if (root.config) root.config.removeBlacklist(app)
    }

    // ---- Apply size / config changes to KWin -------------------------------
    // Only reloadHtml: /KWin reconfigure makes KWin re-read kwinrc and
    // unload/reload effect plugins, which destroys and re-creates the
    // Ultralight renderer and crashes KWin.
    //
    // reloadHtml is debounced: rapid changes (stepper drags, toggling, theme
    // switching) collapse into a single D-Bus call, and the config file is
    // flushed first so the effect reads the final values.
    function _requestReload() {
        if (!root.config) return
        root.config.saveNow()
        reloadTimer.restart()
    }

    function _fireReload() {
        reconfigureProc.running = true
    }

    property Timer reloadTimer: Timer {
        id: reloadTimer
        interval: 400
        repeat: false
        onTriggered: root._fireReload()
    }

    function save() {
        if (!root.config) return
        root._requestReload()
        root.statusMessage = qsTr("Saved")
    }

    property Process reconfigureProc: Process {
        id: reconfigureProc
        command: ["sh", "-c",
            'busctl --user call org.kde.KWin /UltralightCursor org.kde.kwin.KWin.KwinCursorEffect reloadHtml 2>/dev/null; ' +
            'true']
    }
}
