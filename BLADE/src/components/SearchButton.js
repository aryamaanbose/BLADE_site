import React, { useState, useEffect } from 'react';
import './SearchButton.css'; // Optional: Add custom styles

function SearchButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchData, setSearchData] = useState([]);

  // Fetch the search data (search-index.json)
  useEffect(() => {
    const fetchSearchData = async () => {
      try {
        const response = await fetch('/search-index.json');
        const data = await response.json();
        setSearchData(data);
      } catch (error) {
        console.error('Failed to load search data:', error);
      }
    };

    fetchSearchData();
  }, []);

  // Handle search logic
  const handleSearch = (event) => {
    const searchTerm = event.target.value.toLowerCase();
    setQuery(searchTerm);

    const filteredResults = searchData.filter((result) =>
      result.title.toLowerCase().includes(searchTerm) ||
      result.content.toLowerCase().includes(searchTerm)
    );

    setResults(filteredResults);
  };

  return (
    <div style={{ position: 'relative' }}>
      {/* Search Icon */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{
          cursor: 'pointer',
          fontSize: '1.5rem',
          color: '#333',
        }}
      >
        🔍
      </div>

      {/* Search Dropdown */}
      {isOpen && (
        <div
          style={{
            position: 'absolute',
            top: '40px',
            right: '0',
            backgroundColor: '#ffffff',
            border: '1px solid #ddd',
            borderRadius: '4px',
            width: '300px',
            zIndex: '1000',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
            padding: '10px',
          }}
        >
          <input
            type="text"
            placeholder="Search..."
            value={query}
            onChange={handleSearch}
            style={{
              width: '100%',
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              marginBottom: '10px',
            }}
          />
          {query && (
            <div
              style={{
                maxHeight: '200px',
                overflowY: 'auto',
                borderTop: '1px solid #ddd',
                paddingTop: '10px',
              }}
            >
              {results.length > 0 ? (
                results.map((result, index) => (
                  <div
                    key={index}
                    style={{
                      padding: '8px',
                      borderBottom: '1px solid #ddd',
                    }}
                  >
                    <a
                      href={result.url}
                      style={{
                        color: '#7699D4',
                        textDecoration: 'none',
                        fontWeight: 'bold',
                      }}
                    >
                      {result.title}
                    </a>
                    <p style={{ color: '#666', fontSize: '0.85em' }}>
                      {result.content.slice(0, 100)}...
                    </p>
                  </div>
                ))
              ) : (
                <p style={{ color: '#666' }}>No results found.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SearchButton;
