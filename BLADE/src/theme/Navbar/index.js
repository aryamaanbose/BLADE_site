import React, { useState } from 'react';
import Navbar from '@theme-original/Navbar';
import SearchButton from '@site/src/components/SearchButton'; // Path to your SearchButton.js file
import { useColorMode } from '@docusaurus/theme-common';

function CustomNavbar(props) {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const { colorMode, setColorMode } = useColorMode(); // Access the current theme and the toggle function

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      {/* Original Navbar */}
      <Navbar {...props} />

      {/* Right Side: Search Button and Dark Mode Toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginLeft: 'auto', paddingRight: '1rem' }}>
        {/* Search Button */}
        <SearchButton />

        {/* Dark Mode Toggle */}
        <button
          onClick={() => setColorMode(colorMode === 'dark' ? 'light' : 'dark')}
          style={{
            background: 'none',
            border: '1px solid #ccc',
            padding: '5px 10px',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '1rem',
          }}
        >
          {colorMode === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>
    </div>
  );
}

export default CustomNavbar;
