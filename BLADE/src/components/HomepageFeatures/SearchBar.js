// src/components/SearchBar.js
import React, { useState } from 'react';

const SearchBar = () => {
  const [query, setQuery] = useState('');

  const handleSearch = (event) => {
    setQuery(event.target.value.toLowerCase());
    // Add your search logic here (optional)
  };

  return (
    <input
      type="text"
      placeholder="Search..."
      value={query}
      onChange={handleSearch}
      style={{ padding: '5px', margin: '10px', width: '200px' }}
    />
  );
};

export default SearchBar;
