// QML port of `caelestia-dots-kde`'s `Caelestia/Config/webcursorconfig.hpp`.
//
// The KWin cursor effect runs as a separate process and keeps its whole
// user-facing state in the `webCursor` section of the shared Caelestia config
// file (`~/.config/caelestia/shell.json`). This object mirrors that section
// (see `WebCursorMain` / `WebCursorConfig` in the header) so both Nexus and
// this plugin UI manipulate the exact same JSON document the effect reads.
//
// All persisted values are kept as observable properties here; every change is
// flushed back to `shell.json` (debounced). Unrelated sections of the file are
// preserved.
import QtQuick
import Quickshell
import Quickshell.Io

QtObject {
    id: config

    // ---- paths -----------------------------------------------------------
    readonly property string homeDir: Quickshell.env("HOME") || ""
    readonly property string configHome: Quickshell.env("XDG_CONFIG_HOME")
        || (homeDir.length > 0 ? homeDir + "/.config" : homeDir)
    readonly property string configDir: (configHome.length > 0 ? configHome : "") + "/caelestia"
    readonly property string configPath: configDir + "/shell.json"
    readonly property string themesDirDefault: configDir + "/webcursor"

    // ---- config file views ----------------------------------------------
    // One FileView for reading (auto-reload on external edits), one for
    // writing; same idiom as the wallpaper-selector plugin's Config.qml.
    property FileView _reader: FileView {
        id: reader
        path: config.configPath
        preload: true
        watchChanges: true
        onLoaded: config._reparse()
        onFileChanged: { reload(); config._reparse() }
    }
    property FileView _writer: FileView {
        id: writer
        path: config.configPath
        preload: true
    }

    // ---- mirrors `WebCursorMain` (webcursorconfig.hpp) -------------------
    property bool enabled: true
    property int width: 128
    property int height: 128
    property string selectTheme: "variant4-ciallo"
    property var blacklist: []
    property string themesDir: config.themesDirDefault

    // UI convenience; not part of the effect schema.
    property string shortcut: "Meta+Shift+C"

    // Bootstrap: build the bundled C++ effect (contents/ CMake project) with
    // cmake when its artifacts are missing from the plugin folder.
    property bool autoBuild: true

    // Uploads: additionally copy uploaded themes to /usr/share/caelestia/webcursor
    // through pkexec (asks for the system password via polkit).
    property bool installGlobal: false

    property bool _loading: true
    property var _data: ({})

    readonly property var defaultConfig: ({
        webCursor: {
            shortcut: "Meta+Shift+C",
            build: {
                auto: true
            },
            upload: {
                installGlobal: false
            },
            cursor: {
                enabled: true,
                width: 128,
                height: 128,
                selectTheme: "variant4-ciallo",
                blacklist: [],
                themesDir: config.themesDirDefault
            }
        }
    })

    function _setPath(obj, parts, value) {
        var target = obj
        for (var i = 0; i < parts.length - 1; i++) {
            if (typeof target[parts[i]] !== "object" || target[parts[i]] === null)
                target[parts[i]] = {}
            target = target[parts[i]]
        }
        target[parts[parts.length - 1]] = value
    }

    function _sameArray(a, b) {
        if (a.length !== b.length) return false
        for (var i = 0; i < a.length; i++)
            if (a[i] !== b[i]) return false
        return true
    }

    function _mergeDefaults(data) {
        var changed = false
        if (typeof data.webCursor !== "object" || data.webCursor === null) {
            data.webCursor = {}
            changed = true
        }
        var cur = data.webCursor
        var def = defaultConfig.webCursor
        for (var key in def) {
            if (key === "cursor" || key === "build" || key === "upload") continue
            if (cur[key] === undefined) { cur[key] = def[key]; changed = true }
        }
        if (typeof cur.build !== "object" || cur.build === null) {
            cur.build = {}
            changed = true
        }
        if (cur.build.auto === undefined) { cur.build.auto = true; changed = true }
        if (typeof cur.upload !== "object" || cur.upload === null) {
            cur.upload = {}
            changed = true
        }
        if (cur.upload.installGlobal === undefined) { cur.upload.installGlobal = false; changed = true }
        if (typeof cur.cursor !== "object" || cur.cursor === null) {
            cur.cursor = {}
            changed = true
        }
        var cursorDef = def.cursor
        for (var ckey in cursorDef) {
            if (cur.cursor[ckey] === undefined) { cur.cursor[ckey] = cursorDef[ckey]; changed = true }
        }
        return changed
    }

    function _reparse() {
        var raw = _reader.text() || ""
        var parsed = {}
        if (raw.length > 0) {
            try { parsed = JSON.parse(raw) } catch (e) { parsed = {} }
        }
        _mergeDefaults(parsed)

        _loading = true
        config._data = parsed
        var cursor = parsed.webCursor.cursor
        config.enabled = !!cursor.enabled
        config.width = parseInt(cursor.width, 10) || 128
        config.height = parseInt(cursor.height, 10) || 128
        config.selectTheme = cursor.selectTheme || "variant4-ciallo"
        const incomingBlacklist = Array.isArray(cursor.blacklist) ? cursor.blacklist.slice() : []
        if (!config._sameArray(incomingBlacklist, config.blacklist))
            config.blacklist = incomingBlacklist
        config.themesDir = cursor.themesDir || config.themesDirDefault
        config.shortcut = parsed.webCursor.shortcut || "Meta+Shift+C"
        config.autoBuild = !parsed.webCursor.build || parsed.webCursor.build.auto !== false
        config.installGlobal = !parsed.webCursor.upload || parsed.webCursor.upload.installGlobal === true
        _loading = false
    }

    // ---- persistence -----------------------------------------------------
    property var _pending: []
    property Timer _saveTimer: Timer {
        id: saveTimer
        interval: 200
        repeat: false
        onTriggered: config._flush()
    }

    function saveKey(path, value) {
        if (config._loading) return
        config._pending.push({ path: path, value: value })
        saveTimer.restart()
    }

    function saveNow() {
        saveTimer.stop()
        config._flush()
    }

    function _flush() {
        if (config._pending.length === 0) return
        _writer.reload()
        var data = {}
        try { data = JSON.parse(_writer.text() || "{}") } catch (e) { data = {} }
        _mergeDefaults(data)
        var pending = config._pending
        config._pending = []
        for (var i = 0; i < pending.length; i++) {
            var parts = pending[i].path.split(".")
            config._setPath(data, parts, pending[i].value)
        }
        config._data = data
        const newText = JSON.stringify(data, null, 2) + "\n"
        // Skip no-op writes so watchChanges does not trigger a reparse loop.
        var equal = false
        try {
            equal = (_writer.text() || "").trim().length > 0
                && JSON.stringify(JSON.parse(_writer.text()), null, 2) + "\n" === newText
        } catch (e) { /* treat unparsable as different */ }
        if (!equal) _writer.setText(newText)
    }

    function resetCursorToDefault() {
        if (config._loading) return
        _writer.reload()
        var data = {}
        try { data = JSON.parse(_writer.text() || "{}") } catch (e) { data = {} }
        data.webCursor = JSON.parse(JSON.stringify(defaultConfig.webCursor))
        // Re-resolve the dynamic default path.
        data.webCursor.cursor.themesDir = config.themesDirDefault
        data.webCursor.shortcut = "Meta+Shift+C"
        _loading = true
        config._data = data
        config.enabled = data.webCursor.cursor.enabled
        config.width = data.webCursor.cursor.width
        config.height = data.webCursor.cursor.height
        config.selectTheme = data.webCursor.cursor.selectTheme
        config.blacklist = []
        config.themesDir = data.webCursor.cursor.themesDir
        config.shortcut = data.webCursor.shortcut
        config.autoBuild = !data.webCursor.build || data.webCursor.build.auto !== false
        config.installGlobal = !data.webCursor.upload || data.webCursor.upload.installGlobal === true
        _loading = false
        _writer.setText(JSON.stringify(data, null, 2) + "\n")
    }

    // ---- blacklist helpers ----------------------------------------------
    function addBlacklist(app) {
        const trimmed = String(app || "").trim()
        if (!trimmed) return
        const list = config.blacklist
        if (list.indexOf(trimmed) === -1)
            config.blacklist = [...list, trimmed]
    }

    function removeBlacklist(app) {
        const list = config.blacklist
        const next = list.filter(a => a !== app)
        if (next.length !== list.length)
            config.blacklist = next
    }

    // ---- value change handlers ------------------------------------------
    onEnabledChanged: saveKey("webCursor.cursor.enabled", config.enabled)
    onWidthChanged: saveKey("webCursor.cursor.width", config.width)
    onHeightChanged: saveKey("webCursor.cursor.height", config.height)
    onSelectThemeChanged: saveKey("webCursor.cursor.selectTheme", config.selectTheme)
    onThemesDirChanged: saveKey("webCursor.cursor.themesDir", config.themesDir)
    onShortcutChanged: saveKey("webCursor.shortcut", config.shortcut)
    onAutoBuildChanged: saveKey("webCursor.build.auto", config.autoBuild)
    onInstallGlobalChanged: saveKey("webCursor.upload.installGlobal", config.installGlobal)
    onBlacklistChanged: saveKey("webCursor.cursor.blacklist", config.blacklist)

    Component.onCompleted: {
        config._reparse()
    }
}
