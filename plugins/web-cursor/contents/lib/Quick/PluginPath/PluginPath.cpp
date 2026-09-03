#include "PluginPath.hpp"
#include <QCoreApplication>
#include <QDir>
#include <QStandardPaths>

namespace UltralightWebCursorM {

std::filesystem::path PluginPath::dataDir() {
    QString path = QStringLiteral("/usr/share/caelestia/webcursor");
    return std::filesystem::path(path.toStdString());
}

} // namespace UltralightWebCursorM
