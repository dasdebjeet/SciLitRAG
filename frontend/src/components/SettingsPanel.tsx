
/*
 * SettingsPanel
 *
 * A compact floating settings panel opened from the navigation rail.
 * Theme preference is controlled by App; conversation clearing is delegated
 * to the conversation hook so this component does not own application data.
 */

import { useEffect, useState } from "react";
import {
  Monitor,
  Moon,
  Sun,
  Trash2,
  X,
} from "lucide-react";

export type ThemePreference = "system" | "light" | "dark";

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  theme: ThemePreference;
  onThemeChange: (theme: ThemePreference) => void;
  conversationCount: number;
  onClearConversations: () => void;
}

function SettingsPanel({
  isOpen,
  onClose,
  theme,
  onThemeChange,
  conversationCount,
  onClearConversations,
}: SettingsPanelProps) {
  const [isConfirmingClear, setIsConfirmingClear] = useState(false);

  // Allow keyboard users to dismiss the floating panel with Escape.
  useEffect(() => {
    if (!isOpen) return;

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  function handleClearConversations() {
    onClearConversations();
    setIsConfirmingClear(false);
  }

  if (!isOpen) return null;

  return (
    <section
      id="settings-panel"
      className="settings-panel"
      role="dialog"
      aria-modal="false"
      aria-labelledby="settings-panel-title"
    >
      <header className="settings-panel-header">
        <div>
          <span className="settings-eyebrow">PREFERENCES</span>
          <h2 id="settings-panel-title">Settings</h2>
        </div>

        <button
          type="button"
          className="settings-close-button"
          onClick={onClose}
          aria-label="Close settings"
        >
          <X size={18} strokeWidth={1.8} aria-hidden="true" />
        </button>
      </header>

      <div className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h3>Appearance</h3>
            <p>Choose how SciLitRAG looks.</p>
          </div>
        </div>

        {/* Connected horizontal theme selector: label followed by icon. */}
        <div
          className="settings-theme-toggle"
          role="group"
          aria-label="Color theme"
        >
          <button
            type="button"
            className={`settings-theme-option ${
              theme === "light" ? "active" : ""
            }`}
            onClick={() => onThemeChange("light")}
            aria-pressed={theme === "light"}
          >
            <span>Light</span>
            <Sun size={15} aria-hidden="true" />
          </button>

          <button
            type="button"
            className={`settings-theme-option ${
              theme === "dark" ? "active" : ""
            }`}
            onClick={() => onThemeChange("dark")}
            aria-pressed={theme === "dark"}
          >
            <span>Dark</span>
            <Moon size={15} aria-hidden="true" />
          </button>

          <button
            type="button"
            className={`settings-theme-option ${
              theme === "system" ? "active" : ""
            }`}
            onClick={() => onThemeChange("system")}
            aria-pressed={theme === "system"}
          >
            <span>System</span>
            <Monitor size={15} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h3>Storage</h3>
            <p>Manage your saved conversations.</p>
          </div>
          <span className="settings-count">
            {conversationCount}
          </span>
        </div>

        {!isConfirmingClear ? (
          <button
            type="button"
            className="settings-danger-action"
            onClick={() => setIsConfirmingClear(true)}
            disabled={conversationCount === 0}
          >
            <span>Clear conversation history</span>
            <Trash2 size={16} aria-hidden="true" />
          </button>
        ) : (
          <div className="settings-confirmation" role="alert">
            <p>
              Delete all {conversationCount} saved{" "}
              {conversationCount === 1 ? "conversation" : "conversations"}?
              This cannot be undone.
            </p>

            <div className="settings-confirm-actions">
              <button
                type="button"
                className="settings-secondary-button"
                onClick={() => setIsConfirmingClear(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="settings-confirm-delete"
                onClick={handleClearConversations}
              >
                Clear history
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="settings-section settings-about-section">
        <h3>About</h3>

        <div className="settings-about">
          <img
            src="/favicon.svg"
            alt=""
            className="settings-about-logo"
          />

          <div className="settings-about-details">
            <strong>SciLitRAG</strong>
            <span>Research Assistant</span>
          </div>

          <span className="settings-version">v0.0.0</span>
        </div>
      </div>
    </section>
  );
}

export default SettingsPanel;
