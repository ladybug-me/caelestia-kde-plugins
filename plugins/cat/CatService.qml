import QtQuick
import QtCore
import Quickshell
import Quickshell.Io
import "."

// Reads the CPU counters the plugin needs and turns them into a cat: idle when
// the machine is quiet, walking as load picks up, sprinting near 100%. Each
// frame is published into CatState so the bar widget and the popout agree.
Item {
    id: root

    // Upstream's pacing, kept in milliseconds.
    readonly property int tickMs: 80
    readonly property int walkMaxMs: 380
    readonly property int walkMinMs: 180
    readonly property int runMaxMs: 160
    readonly property int runMinMs: 55
    readonly property var runGlyphs: ["b", "c", "d", "e", "f"]

    Settings {
        id: cfg

        category: "Plugin_cat"
    }

    function clampInt(value: var, fallback: int, minimum: int, maximum: int): int {
        const number = Number(value);
        if (!isFinite(number))
            return fallback;
        return Math.max(minimum, Math.min(maximum, Math.round(number)));
    }

    readonly property int catSize: root.clampInt(cfg.value("cat_size", 24), 24, 12, 48)
    readonly property bool showCpuPercent: String(cfg.value("show_cpu_percent", false)) === "true"
    readonly property int walkThreshold: root.clampInt(cfg.value("walk_threshold", 15), 15, 0, 100)
    readonly property int runThreshold: root.clampInt(cfg.value("run_threshold", 60), 60, 0, 100)
    readonly property int pollSeconds: root.clampInt(cfg.value("poll_interval", 2), 2, 1, 10)
    readonly property string colourMode: String(cfg.value("cat_color_mode", "theme")) === "custom" ? "custom" : "theme"
    readonly property color customColour: {
        const raw = String(cfg.value("cat_color", "#E8A24C")).trim();
        return /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(raw) ? raw : "#E8A24C";
    }

    property int cpuPercent: 0
    property real previousTotal: -1
    property real previousIdle: -1
    property int frameIndex: 1
    property int frameElapsed: 0

    function publishSettings(): void {
        CatState.catSize = root.catSize;
        CatState.showCpuPercent = root.showCpuPercent;
        CatState.colourMode = root.colourMode;
        CatState.customColour = root.customColour;
    }

    // /proc/stat, aggregate "cpu" line: user nice system idle iowait irq softirq
    // steal ... . Usage is the share of the delta that was not idle or waiting.
    function consumeStat(raw: string): void {
        const first = String(raw ?? "").split("\n")[0];
        if (!first.startsWith("cpu "))
            return;
        const fields = first.trim().split(/\s+/).slice(1).map(Number);
        if (fields.length < 4)
            return;
        let valid = true;
        let total = 0;
        for (const value of fields) {
            if (!isFinite(value))
                valid = false;
            total += value;
        }
        if (!valid || total <= 0)
            return;
        const idle = fields[3] + (fields.length > 4 ? fields[4] : 0);
        if (root.previousTotal > 0) {
            const totalDelta = total - root.previousTotal;
            const idleDelta = idle - root.previousIdle;
            if (totalDelta > 0)
                root.cpuPercent = Math.max(0, Math.min(100, Math.round((1 - idleDelta / totalDelta) * 100)));
        }
        root.previousTotal = total;
        root.previousIdle = idle;
    }

    function paceFor(cpu: real): var {
        const runFloor = Math.max(root.runThreshold, root.walkThreshold + 1);
        if (cpu < root.walkThreshold)
            return {
                pace: "idle",
                duration: root.walkMaxMs
            };
        if (cpu < runFloor) {
            const range = Math.max(runFloor - root.walkThreshold, 1);
            const t = Math.max(0, Math.min(1, (cpu - root.walkThreshold) / range));
            return {
                pace: "walk",
                duration: root.walkMaxMs - t * (root.walkMaxMs - root.walkMinMs)
            };
        }
        const range = Math.max(100 - runFloor, 1);
        const t = Math.max(0, Math.min(1, (cpu - runFloor) / range));
        return {
            pace: "run",
            duration: root.runMaxMs - t * (root.runMaxMs - root.runMinMs)
        };
    }

    function tick(): void {
        const pace = root.paceFor(root.cpuPercent);
        if (pace.pace === "idle") {
            root.frameElapsed = 0;
            root.frameIndex = 1;
        } else {
            root.frameElapsed += root.tickMs;
            if (root.frameElapsed >= pace.duration) {
                root.frameElapsed = 0;
                root.frameIndex = root.frameIndex % root.runGlyphs.length + 1;
            }
        }
        CatState.pace = pace.pace;
        CatState.glyph = pace.pace === "idle" ? "a" : root.runGlyphs[root.frameIndex - 1];
        CatState.cpuPercent = root.cpuPercent;
    }

    Component.onCompleted: {
        root.publishSettings();
        root.tick();
    }

    FileView {
        id: procStat

        path: "/proc/stat"
        printErrors: false
        onLoaded: root.consumeStat(procStat.text())
    }

    Timer {
        interval: root.tickMs
        repeat: true
        running: true
        onTriggered: root.tick()
    }
    Timer {
        interval: root.pollSeconds * 1000
        repeat: true
        running: true
        triggeredOnStart: true
        onTriggered: procStat.reload()
    }
    // Usage is a delta, so the first reading alone says nothing: take a second
    // one shortly after start instead of waiting a whole poll interval.
    Timer {
        interval: 600
        repeat: false
        running: true
        onTriggered: procStat.reload()
    }
}
