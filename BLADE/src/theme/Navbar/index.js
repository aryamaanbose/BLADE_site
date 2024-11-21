import React, { useState, useEffect } from 'react';
import Navbar from '@theme-original/Navbar';
import SearchButton from '@site/src/components/SearchButton'; // Adjust the path as needed
import { useColorMode } from '@docusaurus/theme-common';
import { MdFullscreen, MdFullscreenExit } from 'react-icons/md';

function CustomNavbar(props) {
  const { colorMode, setColorMode } = useColorMode();
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen mode: ${err.message} (${err.name})`);
      });
    } else if (document.exitFullscreen) {
      document.exitFullscreen().catch((err) => {
        console.error(`Error attempting to exit fullscreen mode: ${err.message} (${err.name})`);
      });
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Manually define available locales and their paths
  const locales = [
    { code: 'en', label: 'English', url: '/' },
    { code: 'nl', label: 'Dutch', url: '/nl/' }, // Replace with your actual Dutch locale path
  ];

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      {/* Original Navbar */}
      <Navbar {...props} />

      {/* Right Side: Search Button, Dark Mode Toggle, Fullscreen Toggle, Language Dropdown */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginLeft: 'auto', paddingRight: '1rem' }}>
        {/* Search Button */}
        <SearchButton />

        {/* Dark Mode Toggle */}
        <button
          onClick={() => setColorMode(colorMode === 'dark' ? 'light' : 'dark')}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '1.5rem',
            color: 'var(--ifm-font-color-base)',
          }}
          aria-label={colorMode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {colorMode === 'dark' ? '☀️' : '🌙'}
        </button>

        {/* Fullscreen Toggle */}
        <button
          onClick={toggleFullscreen}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '1.5rem',
            color: 'var(--ifm-font-color-base)',
          }}
          aria-label={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
        >
          {isFullscreen ? <MdFullscreenExit /> : <MdFullscreen />}
        </button>

        {/* Language Dropdown */}
        <div className="language-dropdown" style={{ position: 'relative' }}>
          <select
            onChange={(e) => {
              const selectedLocale = locales.find((locale) => locale.code === e.target.value);
              if (selectedLocale) {
                window.location.href = selectedLocale.url;
              }
            }}
            style={{
              padding: '6px 12px',
              fontSize: '1rem',
              border: '1px solid var(--ifm-color-primary)',
              borderRadius: '5px',
              cursor: 'pointer',
              backgroundColor: 'var(--ifm-navbar-background-color)',
              color: 'var(--ifm-font-color-base)',
            }}
          >
            {locales.map((locale) => (
              <option key={locale.code} value={locale.code}>
                {locale.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

export default CustomNavbar;
