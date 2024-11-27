import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './SidebarSearch.css'; // Add this line to link the custom styles

const SidebarSearch = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchData, setSearchData] = useState([]); // Store full search data

  useEffect(() => {
    // Load search data from a JSON or API endpoint (adjust path if necessary)
    const fetchData = async () => {
      try {
        const response = await fetch('/search-index.json'); // Replace with your data source
        const data = await response.json();
        setSearchData(data); // Save full data in searchData
        setResults(data); // Initially set results to full data
      } catch (error) {
        console.error("Failed to load search data", error);
      }
    };
    fetchData();
  }, []);

  const handleSearch = (event) => {
    const searchTerm = event.target.value.toLowerCase();
    setQuery(searchTerm);

    // Filter results from searchData instead of modifying the results state directly
    const filteredResults = searchData.filter((result) =>
      result.title.toLowerCase().includes(searchTerm) ||
      result.content.toLowerCase().includes(searchTerm)
    );

    setResults(filteredResults);
  };

  return (
    <div className="sidebar-search">
      <input
        type="text"
        placeholder="Search..."
        value={query}
        onChange={handleSearch}
        className="sidebar-search-input" // Updated for external styling
      />
      <div className="sidebar-search-results">
        {query && results.length > 0 ? (
          results.map((result, index) => (
            <div key={index} className="sidebar-search-result-item">
              <Link to={result.url} className="sidebar-search-result-link">
                <strong>{result.title}</strong>
              </Link>
              <p>{result.content.slice(0, 100)}...</p>
            </div>
          ))
        ) : (
          query && <p>No results found</p>
        )}
      </div>
    </div>
  );
};

export default SidebarSearch;

