// Settings UI for the web cursor effect.
//
// Port of `caelestia-dots-kde/shell/modules/nexus/pages/desktop/WebCursorPage.qml`
// into the plugin's own QML architecture (mirroring the wallpaper-selector
// plugin): no dependency on shell-only `qs.components` / `nexus.common`
// controls — rows, switches and steppers are the plugin's own components and
// all state lives in the `WebCursorConfig` / `WebCursorManager` singletons.
import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import Quickshell
import Quickshell.Io
import qs.components.controls
import ".."
import "../components"

Item {
    id: root

    signal closeRequested()

    property bool showing: false
    property var colors: null
    property var config: null
    property var manager: null
    property string buildStatus: ""

    visible: showing || opacity > 0
    opacity: showing ? 1 : 0
    enabled: showing
    Behavior on opacity { NumberAnimation { duration: Style.animMedium; easing.type: Easing.InOutQuad } }

    onShowingChanged: {
        if (root.showing) {
            root.forceActiveFocus()
            if (root.manager) root.manager.refreshThemes()
        }
    }

    Keys.onPressed: event => {
        if (event.key === Qt.Key_Escape) {
            root.closeRequested()
            event.accepted = true
        }
    }

    // Dim background - swallows clicks outside the modal.
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        onClicked: root.closeRequested()

        Rectangle {
            anchors.fill: parent
            color: "black"
            opacity: 0.5
        }
    }

    // Modal card
    Rectangle {
        id: modal
        width: Math.max(340, Math.min(720, parent.width - 80))
        height: Math.max(360, Math.min(parent.height - 120, contentLayout.implicitHeight + 120))
        anchors.centerIn: parent
        radius: Style.radiusXLarge + 4
        clip: true
        color: colors ? colors.surfaceContainer : "transparent"
        scale: root.showing ? 1.0 : 0.95
        Behavior on scale { NumberAnimation { duration: Style.animEnter; easing.type: Easing.OutBack } }

        // Modal guard so clicks inside do not bubble to the dim layer.
        MouseArea { anchors.fill: parent; onClicked: {} }

        ColumnLayout {
            anchors.fill: parent
            spacing: 0

            // Header
            RowLayout {
                Layout.fillWidth: true
                Layout.margins: Style.paddingLarge
                Layout.leftMargin: Style.paddingXLarge

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    Text {
                        text: qsTr("Web Cursor")
                        font.family: Style.fontFamilyHeading
                        font.pixelSize: Style.fontTitleLarge
                        font.weight: Font.DemiBold
                        color: colors ? colors.surfaceText : "#e0e0e0"
                    }
                    Text {
                        text: qsTr("HTML/CSS cursor rendered through KWin by Ultralight")
                        font.family: Style.fontFamily
                        font.pixelSize: Style.fontCaption
                        color: colors ? colors.surfaceVariantText : "#b0b0b0"
                    }
                }

                IconButton {
                    icon: "close"
                    type: IconButton.Text
                    onClicked: root.closeRequested()
                }
            }

            // Scrollable content
            Flickable {
                id: contentFlick
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                contentWidth: width
                contentHeight: contentLayout.implicitHeight + Style.paddingXLarge
                boundsBehavior: Flickable.StopAtBounds

                ScrollBar.vertical: ScrollBar {
                    parent: contentFlick
                    anchors.top: contentFlick.top
                    anchors.right: contentFlick.right
                    anchors.bottom: contentFlick.bottom
                    active: contentFlick.contentHeight > contentFlick.height
                    visible: active
                    contentItem: Rectangle {
                        implicitWidth: 6
                        radius: 3
                        color: colors ? colors.primary : Style.fallbackAccent
                    }
                }

                ColumnLayout {
                    id: contentLayout
                    width: contentFlick.width - Style.paddingXLarge * 2
                    x: Style.paddingXLarge
                    y: Style.paddingMedium
                    spacing: Style.spacingMedium

                    // Effect bootstrap status (see main.qml bootstrapEffect).
                    Text {
                        Layout.fillWidth: true
                        visible: root.buildStatus.length > 0
                        text: root.buildStatus
                        color: colors ? colors.tertiary : "#8bceff"
                        font.family: Style.fontFamily
                        font.pixelSize: Style.fontCaption
                        wrapMode: Text.WordWrap
                    }

                    // ---- Enable -------------------------------------------------
                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        SwitchRow {
                            id: enableRow
                            anchors.fill: parent
                            text: qsTr("Enable Web Cursor")
                            subtext: qsTr("Render the selected HTML/CSS cursor through KWin")
                            checked: root.config ? root.config.enabled : false
                            colors: root.colors
                            onToggled: on => { on ? root.manager.enable() : root.manager.disable() }
                        }
                        implicitHeight: enableRow.implicitHeight + padding * 2
                    }

                    // ---- Theme picker -------------------------------------------
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.topMargin: Style.spacingLarge

                        SectionHeader {
                            Layout.fillWidth: true
                            text: qsTr("Cursor Theme")
                            colors: root.colors
                            first: true
                        }
                        IconButton {
                            icon: "add"
                            type: IconButton.Tonal
                            onClicked: root.openFolderPicker()
                        }
                    }

                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        SwitchRow {
                            id: installGlobalRow
                            anchors.fill: parent
                            text: qsTr("Install system-wide")
                            subtext: qsTr("Copy uploaded themes to /usr/share/caelestia/webcursor (asks for your password)")
                            checked: root.config ? root.config.installGlobal : false
                            colors: root.colors
                            onToggled: on => { if (root.config) root.config.installGlobal = on }
                        }
                        implicitHeight: installGlobalRow.implicitHeight + padding * 2
                    }

                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        RowLayout {
                            id: currentThemeRow
                            anchors.fill: parent
                            spacing: Style.spacingLarge

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 0
                                Text {
                                    text: qsTr("Current theme")
                                    font.family: Style.fontFamily
                                    font.pixelSize: Style.fontCaption
                                    color: colors ? colors.surfaceVariantText : "#b0b0b0"
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: root.config ? root.config.selectTheme : ""
                                    font.family: Style.fontFamily
                                    font.pixelSize: Style.fontBodyLarge
                                    color: colors ? colors.surfaceText : "#e0e0e0"
                                    elide: Text.ElideRight
                                }
                            }
                            IconButton {
                                icon: "refresh"
                                type: IconButton.Text
                                onClicked: root.manager.reload()
                            }
                        }
                        implicitHeight: currentThemeRow.implicitHeight + padding * 2
                    }

                    Repeater {
                        model: root.manager ? root.manager.themeList : []

                        delegate: RowCard {
                            required property string modelData
                            readonly property var details: root.manager.getThemeDetails(modelData)

                            Layout.fillWidth: true
                            colors: root.colors
                            implicitHeight: themeRow.implicitHeight + padding * 2

                            RowLayout {
                                id: themeRow
                                anchors.fill: parent
                                spacing: Style.spacingLarge

                                Image {
                                    readonly property real baseSize: 56
                                    Layout.preferredWidth: baseSize
                                    Layout.preferredHeight: baseSize
                                    Layout.alignment: Qt.AlignVCenter
                                    sourceSize.width: 112
                                    sourceSize.height: 112
                                    source: details.iconPath || ""
                                    fillMode: Image.PreserveAspectFit
                                    smooth: true
                                    cache: false
                                    visible: status === Image.Ready
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData
                                        font.family: Style.fontFamily
                                        font.pixelSize: Style.fontBody
                                        color: colors ? colors.surfaceText : "#e0e0e0"
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        visible: details.describe.length > 0
                                        text: details.describe
                                        font.family: Style.fontFamily
                                        font.pixelSize: Style.fontCaption
                                        color: colors ? colors.surfaceVariantText : "#b0b0b0"
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        text: qsTr("By %1 · minimum %2 × %3").arg(details.author).arg(details.minWidth).arg(details.minHeight)
                                        font.family: Style.fontFamily
                                        font.pixelSize: Style.fontTiny
                                        color: colors ? colors.surfaceVariantText : "#b0b0b0"
                                    }
                                }

                                IconButton {
                                    readonly property bool active: root.config && root.config.selectTheme === modelData
                                    icon: active ? "check" : "play_arrow"
                                    type: IconButton.Text
                                    onClicked: root.manager.useTheme(modelData)
                                }
                                IconButton {
                                    icon: "folder_open"
                                    type: IconButton.Text
                                    onClicked: root.manager.openThemeFolder(modelData)
                                }
                                IconButton {
                                    visible: root.manager.isUserTheme(modelData)
                                    icon: "delete"
                                    type: IconButton.Text
                                    onClicked: root.manager.removeTheme(modelData)
                                }
                            }
                        }
                    }

                    // ---- Size -----------------------------------------------------
                    SectionHeader {
                        Layout.fillWidth: true
                        Layout.topMargin: Style.spacingLarge
                        text: qsTr("Size")
                        colors: root.colors
                    }

                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        StepperRow {
                            id: widthRow
                            anchors.fill: parent
                            text: qsTr("Cursor width")
                            subtext: qsTr("Render width in pixels")
                            min: 1
                            max: 1920
                            value: root.config ? root.config.width : 128
                            colors: root.colors
                            onMoved: value => {
                                root.config.width = value
                                root.manager.save()
                            }
                        }
                        implicitHeight: widthRow.implicitHeight + padding * 2
                    }

                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        StepperRow {
                            id: heightRow
                            anchors.fill: parent
                            text: qsTr("Cursor height")
                            subtext: qsTr("Render height in pixels")
                            min: 1
                            max: 1080
                            value: root.config ? root.config.height : 128
                            colors: root.colors
                            onMoved: value => {
                                root.config.height = value
                                root.manager.save()
                            }
                        }
                        implicitHeight: heightRow.implicitHeight + padding * 2
                    }

                    // ---- Ignored applications --------------------------------------
                    SectionHeader {
                        Layout.fillWidth: true
                        Layout.topMargin: Style.spacingLarge
                        text: qsTr("Ignored Applications")
                        colors: root.colors
                    }

                    RowCard {
                        Layout.fillWidth: true
                        colors: root.colors
                        ColumnLayout {
                            id: blacklistColumn
                            anchors.fill: parent
                            spacing: Style.spacingSmall

                            Repeater {
                                model: root.config ? root.config.blacklist : []

                                delegate: RowLayout {
                                    required property string modelData
                                    Layout.fillWidth: true

                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData
                                        font.family: Style.fontFamily
                                        font.pixelSize: Style.fontBody
                                        color: colors ? colors.surfaceText : "#e0e0e0"
                                        elide: Text.ElideRight
                                    }
                                    IconButton {
                                        icon: "close"
                                        type: IconButton.Text
                                        onClicked: root.manager.removeBlacklist(modelData)
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: Style.spacingSmall

                                TextField {
                                    id: blacklistInput
                                    Layout.fillWidth: true
                                    placeholderText: qsTr("Window class or application name")
                                    color: colors ? colors.surfaceText : "#e0e0e0"
                                    placeholderTextColor: colors ? colors.surfaceVariantText : "#b0b0b0"
                                    selectByMouse: true
                                    font.family: Style.fontFamily
                                    font.pixelSize: Style.fontBody
                                    background: Rectangle {
                                        radius: Style.radiusMedium
                                        color: colors ? colors.surface : "transparent"
                                        border.color: colors ? colors.outline : "transparent"
                                        border.width: 1
                                    }
                                    onAccepted: addBlacklistFromInput()
                                }
                                IconButton {
                                    icon: "add"
                                    type: IconButton.Tonal
                                    onClicked: addBlacklistFromInput()
                                }
                            }
                        }
                        implicitHeight: blacklistColumn.implicitHeight + padding * 2
                    }

                    // ---- Status -----------------------------------------------------
                    Text {
                        Layout.fillWidth: true
                        visible: root.manager && root.manager.statusMessage.length > 0
                        text: root.manager ? root.manager.statusMessage : ""
                        color: colors ? colors.surfaceVariantText : "#b0b0b0"
                        font.family: Style.fontFamily
                        font.pixelSize: Style.fontCaption
                        wrapMode: Text.WordWrap
                    }
                }
            }
        }
    }

    // ---- built-in folder picker ------------------------------------------
    // QtQuick.Dialogs.FolderDialog segfaults inside Quickshell layer-shell
    // windows (and kdialog is not guaranteed to be installed), so theme
    // install uses this tiny in-plugin directory browser instead.
    property bool pickerOpen: false
    property string pickerPath: "/"
    property var pickerItems: []

    function openFolderPicker() {
        root.pickerPath = Quickshell.env("HOME") || "/"
        root.refreshPicker()
        root.pickerOpen = true
    }

    function _pickerParent(path) {
        const p = String(path || "").replace(/\/+$/, "")
        if (!p || p === "/") return "/"
        const i = p.lastIndexOf("/")
        if (i <= 0) return "/"
        return p.substring(0, i)
    }

    function _pickerJoin(path, name) {
        return String(path || "/").replace(/\/+$/, "") + "/" + name
    }

    function refreshPicker() {
        pickerListProc.command = ["sh", "-c",
            'p="$1"; [ -d "$p" ] || exit 0; ' +
            'for d in "$p"/*/; do ' +
            '  [ -d "$d" ] || continue; ' +
            '  n=$(basename "$d"); ' +
            '  if [ -f "$d/index.html" ] && [ -f "$d/CursorData.json" ]; then k="theme"; else k="dir"; fi; ' +
            '  printf "%s|%s\\n" "$n" "$k"; done',
            "--", root.pickerPath]
        pickerListProc.running = true
    }

    function pickerSelect(item) {
        if (!item) return
        if (item.kind === "up") {
            root.pickerPath = root._pickerParent(root.pickerPath)
            root.refreshPicker()
        } else if (item.kind === "dir") {
            root.pickerPath = root._pickerJoin(root.pickerPath, item.label)
            root.refreshPicker()
        } else if (item.kind === "theme") {
            root.pickerOpen = false
            root.manager.uploadTheme(root._pickerJoin(root.pickerPath, item.label), "")
        }
    }

    property Process pickerListProc: Process {
        id: pickerListProc
        stdout: StdioCollector {
            id: pickerOut
            onStreamFinished: {
                const items = []
                if (root.pickerPath !== "/")
                    items.push({ label: "..", kind: "up" })
                const lines = (pickerOut.text || "").split("\n")
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i].trim()
                    if (!line) continue
                    const parts = line.split("|")
                    if (parts.length < 2) continue
                    items.push({ label: parts[0], kind: parts[1] })
                }
                root.pickerItems = items
            }
        }
    }

    // Picker overlay ---------------------------------------------------------
    Rectangle {
        visible: root.pickerOpen
        anchors.fill: parent
        color: "transparent"
        z: 100

        // Scrim: click outside to cancel.
        MouseArea {
            anchors.fill: parent
            onClicked: root.pickerOpen = false

            Rectangle {
                anchors.fill: parent
                color: "black"
                opacity: 0.5
            }
        }

        Rectangle {
            width: Math.min(620, parent.width - 80)
            height: Math.min(480, parent.height - 120)
            anchors.centerIn: parent
            radius: Style.radiusXLarge + 4
            color: colors ? colors.surfaceContainer : "#111111"
            clip: true

            MouseArea {
                anchors.fill: parent
                onClicked: {}
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: Style.paddingLarge
                spacing: Style.spacingMedium

                RowLayout {
                    Layout.fillWidth: true

                    Text {
                        Layout.fillWidth: true
                        text: qsTr("Install cursor theme")
                        font.family: Style.fontFamilyHeading
                        font.pixelSize: Style.fontTitleLarge
                        font.weight: Font.DemiBold
                        color: colors ? colors.surfaceText : "#e0e0e0"
                    }
                    IconButton {
                        icon: "close"
                        type: IconButton.Text
                        onClicked: root.pickerOpen = false
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: root.pickerPath
                    font.family: Style.fontFamilyMono
                    font.pixelSize: Style.fontCaption
                    color: colors ? colors.surfaceVariantText : "#b0b0b0"
                    elide: Text.ElideMiddle
                }

                ListView {
                    id: pickerList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: root.pickerItems

                    delegate: Rectangle {
                        required property var modelData

                        width: ListView.view.width
                        height: 42
                        radius: Style.radiusMedium
                        color: mouse.containsMouse
                            ? (colors ? colors.surfaceVariant : "#33ffffff")
                            : "transparent"

                        RowLayout {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.leftMargin: Style.paddingMedium
                            anchors.rightMargin: Style.paddingMedium
                            spacing: Style.spacingMedium

                            Text {
                                Layout.fillWidth: true
                                text: modelData.label
                                font.family: Style.fontFamily
                                font.pixelSize: Style.fontBodyLarge
                                color: colors ? colors.surfaceText : "#e0e0e0"
                                elide: Text.ElideRight
                            }

                            Rectangle {
                                visible: modelData.kind === "theme"
                                radius: Style.radiusSmall
                                color: colors ? colors.primaryContainer : "transparent"
                                height: 20
                                width: badgeText.implicitWidth + Style.paddingMedium

                                Text {
                                    id: badgeText
                                    anchors.centerIn: parent
                                    text: qsTr("cursor theme")
                                    font.family: Style.fontFamily
                                    font.pixelSize: Style.fontTiny
                                    color: colors ? colors.primaryContainerText : "#ffffff"
                                }
                            }

                            Text {
                                text: modelData.kind === "up" ? "\u2191" : "\u203A"
                                font.family: Style.fontFamily
                                font.pixelSize: Style.fontBodyLarge
                                color: colors ? colors.surfaceVariantText : "#b0b0b0"
                            }
                        }

                        MouseArea {
                            id: mouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: root.pickerSelect(modelData)
                        }
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: qsTr("Open the theme folder and tap it when it is marked \"cursor theme\".")
                    font.family: Style.fontFamily
                    font.pixelSize: Style.fontCaption
                    color: colors ? colors.surfaceVariantText : "#b0b0b0"
                    wrapMode: Text.WordWrap
                }
            }
        }
    }

    function addBlacklistFromInput() {
        const text = blacklistInput.text
        root.manager.addBlacklist(text)
        blacklistInput.clear()
        blacklistInput.text = ""
    }
}
