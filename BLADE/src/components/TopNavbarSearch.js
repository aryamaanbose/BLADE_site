import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './TopNavbarSearch.css'; // Custom styles for the search bar

const TopNavbarSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchData, setSearchData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/search-index.json'); // Adjust path if necessary
        const data = await response.json();
        setSearchData(data);
        setResults(data);
      } catch (error) {
        console.error('Failed to load search data', error);
      }
    };
    fetchData();
  }, []);

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
    <div className="top-navbar-search">
      <input
        type="text"
        placeholder="Search..."
        value={query}
        onChange={handleSearch}
        className="top-navbar-search-input"
      />
      {query && (
        <div className="top-navbar-search-results">
          {results.length > 0 ? (
            results.map((result, index) => (
              <div key={index} className="top-navbar-search-result-item">
                <Link to={result.url} className="top-navbar-search-result-link">
                  <strong>{result.title}</strong>
                </Link>
                <p>{result.content.slice(0, 50)}...</p>
              </div>
            ))
          ) : (
            <p>No results found</p>
          )}
        </div>
      )}
    </div>
  );
};

export default TopNavbarSearch;
