
/*
 * Floating navigation rail.
 *
 * Provides compact, icon-only navigation. Product branding
 * is displayed separately in the main application header.
 */

import {
  Plus,
  MessagesSquare,
  LibraryBig,
  Settings2,
} from "lucide-react";

interface SidebarProps {
  onNewChat: () => void;
  onShowLibrary: () => void;
  onShowConversations: () => void;
  onToggleSettings: () => void;
  activeView: "chat" | "library" | "conversations";
  settingsOpen: boolean;
}

function Sidebar({
  onNewChat,
  onShowLibrary,
  onShowConversations,
  onToggleSettings,
  activeView,
  settingsOpen,
}: SidebarProps) {
  return (
    <aside className="sidebar" aria-label="Application navigation">
      <nav className="sidebar-navigation" aria-label="Main navigation">
        <button
          type="button"
          className={`navigation-item ${
            activeView === "chat" ? "active" : ""
          }`}
          onClick={onNewChat}
          aria-label="New research chat"
          aria-current={activeView === "chat" ? "page" : undefined}
          data-tooltip="New research chat"
        >
          <Plus size={21} strokeWidth={1.8} aria-hidden="true" />
        </button>

        <button
          type="button"
          className={`navigation-item ${
            activeView === "conversations" ? "active" : ""
          }`}
          onClick={onShowConversations}
          aria-label="Conversations"
          aria-current={
            activeView === "conversations" ? "page" : undefined
          }
          data-tooltip="Conversations"
        >
          <MessagesSquare size={20} strokeWidth={1.8} aria-hidden="true" />
        </button>

        <button
          type="button"
          className={`navigation-item ${
            activeView === "library" ? "active" : ""
          }`}
          onClick={onShowLibrary}
          aria-label="Paper library"
          aria-current={activeView === "library" ? "page" : undefined}
          data-tooltip="Paper library"
        >
          <LibraryBig size={20} strokeWidth={1.8} aria-hidden="true" />
        </button>
      </nav>

      <button
        type="button"
        className={`navigation-item ${settingsOpen ? "active" : ""}`}
        onClick={onToggleSettings}
        aria-label="Settings"
        aria-expanded={settingsOpen}
        aria-controls="settings-panel"
        data-tooltip="Settings"
      >
        <Settings2 size={20} strokeWidth={1.8} aria-hidden="true" />
      </button>
    </aside>
  );
}

export default Sidebar;
