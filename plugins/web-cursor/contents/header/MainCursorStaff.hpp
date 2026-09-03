#pragma once

#include "../config/CursorJSON.hpp"
#include "../config/UserConfig.hpp"
#include "../lib/BlackList/BlacklistManager.hpp"
#include "../lib/CrashReport/CrashReport.hpp"
#include "MouseProvider.hpp"
#include "UltralightHtmlEffect.hpp"
#include <QDebug>
#include <QObject>
#include <QPointF>
#include <QRectF>
#include <filesystem>
#include <memory>
#include <string>

namespace UltralightWebCursorM {

class MainCursorStaff {

public:
    MainCursorStaff() = default;
    virtual ~MainCursorStaff() = default;

    bool isWindowBlacklisted(const std::string& windowClass) const { return m_blacklist.contains(windowClass); }

    QRectF getCursorRect(const QPointF& basePos) const {
        if (!m_html)
            return QRectF();
        qreal w = m_html->width();
        qreal h = m_html->height();
        return QRectF(basePos.x() - m_html->hotspotX(), basePos.y() - m_html->hotspotY(), w, h);
    }

protected:
    template <typename MouseProviderType> bool initializeCore() {
        if (m_permanentlyDisabled)
            return false;
        UltralightWebCursorM::CrashHandler::registerHandler();
        try {
            UltralightWebCursorM::UserConfig::instance()->load();
            UltralightWebCursorM::CursorJSON::instance()->load(
                std::filesystem::path(UserConfigimp.html).parent_path().string());
            m_html = std::make_unique<UltralightWebCursorM::UltralightHtmlEffect>();

            m_mouseProvider = std::make_unique<MouseProviderType>();
            m_mouseProvider->initialize();

            if (!m_html || !m_mouseProvider) {
                m_permanentlyDisabled = true;
                return false;
            }

            if (!m_html->initialize(UserConfigimp, CursorJSONImp)) {
                m_html.reset();
                m_mouseProvider.reset();
                m_permanentlyDisabled = true;
                return false;
            }

            m_blacklist.setBlacklist(UltralightWebCursorM::UserConfig::instance()->getBlacklist());
            return true;
        } catch (const std::exception& e) {
            m_permanentlyDisabled = true;
            return false;
        } catch (...) {
            m_permanentlyDisabled = true;
            return false;
        }
    }

    virtual bool checkFullScreen() const { return false; }

    bool m_permanentlyDisabled = false;
    bool m_isIdleHidden = false;
    std::unique_ptr<UltralightWebCursorM::UltralightHtmlEffect> m_html;
    std::unique_ptr<UltralightWebCursorM::IMouseProvider> m_mouseProvider;
    UltralightWebCursorM::BlacklistManager m_blacklist;

    QPointF m_cursorPoint;
};

} // namespace UltralightWebCursorM
