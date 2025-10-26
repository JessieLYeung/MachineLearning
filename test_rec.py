"""
Unit tests for the anime recommendation system.
Run with: pytest test_rec.py
"""
import pytest
import pandas as pd
import numpy as np
from rec import recommendation_system, find_closest_titles, load_and_process_data


@pytest.fixture
def sample_data():
    """Load the actual data for testing."""
    df, features = load_and_process_data()
    return df, features


def test_load_and_process_data():
    """Test that data loads and preprocessing works."""
    df, features = load_and_process_data()
    
    # Check DataFrame is not empty
    assert len(df) > 0, "DataFrame should not be empty"
    
    # Check required columns exist
    required_cols = ['name', 'genre', 'type', 'episodes', 'rating']
    for col in required_cols:
        assert col in df.columns, f"Column {col} should exist"
    
    # Check features matrix shape matches DataFrame
    assert features.shape[0] == len(df), "Feature matrix rows should match DataFrame length"
    
    # Check no NaN in critical columns after processing
    assert df['episodes'].isna().sum() == 0, "Episodes should have no NaN after imputation"
    assert df['rating'].isna().sum() == 0, "Rating should have no NaN after imputation"


def test_recommendation_system_exact_match(sample_data):
    """Test recommendation system with an exact title match."""
    df, features = sample_data
    
    # Use a known anime from the dataset
    test_title = "Steins;Gate"
    top_n = 5
    
    result = recommendation_system(test_title, df, features, top_n=top_n)
    
    # Should return a DataFrame
    assert isinstance(result, pd.DataFrame), "Should return DataFrame for valid title"
    
    # Should return the requested number of recommendations
    assert len(result) == top_n, f"Should return {top_n} recommendations"
    
    # Should have required columns
    assert 'name' in result.columns
    assert 'similarity' in result.columns
    
    # Original title should not be in recommendations
    assert test_title not in result['name'].values, "Should not recommend the input anime itself"
    
    # Similarity scores should be between 0 and 1
    assert all(result['similarity'] >= 0), "Similarity should be >= 0"
    assert all(result['similarity'] <= 1), "Similarity should be <= 1"


def test_recommendation_system_fuzzy_match(sample_data):
    """Test recommendation system with a fuzzy/partial match."""
    df, features = sample_data
    
    # Use a partial/misspelled title
    test_title = "steins gate"  # lowercase, no semicolon
    
    result = recommendation_system(test_title, df, features, top_n=3)
    
    # Should still return recommendations via fuzzy matching
    assert isinstance(result, pd.DataFrame), "Should return DataFrame even with fuzzy match"
    assert len(result) > 0, "Should find recommendations via fuzzy matching"


def test_recommendation_system_not_found(sample_data):
    """Test recommendation system with a title that doesn't exist."""
    df, features = sample_data
    
    # Use a nonsense title unlikely to match
    test_title = "XYZ_NonexistentAnime_12345"
    
    result = recommendation_system(test_title, df, features, top_n=5)
    
    # Should return a string error message
    assert isinstance(result, str), "Should return string message for non-existent title"
    assert "not found" in result.lower(), "Error message should mention 'not found'"


def test_find_closest_titles(sample_data):
    """Test the fuzzy title matching function."""
    df, features = sample_data
    
    names_list = df['name'].tolist()
    
    # Test with a known partial match
    matches = find_closest_titles("cowboy", names_list, n=3, cutoff=0.5)
    
    assert len(matches) <= 3, "Should return at most n matches"
    
    # Should find "Cowboy Bebop" if it exists
    cowboy_matches = [m for m in matches if 'Cowboy' in m]
    assert len(cowboy_matches) > 0, "Should find Cowboy Bebop variants"


def test_find_closest_titles_no_match():
    """Test fuzzy matching with no close matches."""
    names_list = ["Anime A", "Anime B", "Anime C"]
    
    matches = find_closest_titles("XYZ123Nonexistent", names_list, n=5, cutoff=0.6)
    
    assert len(matches) == 0, "Should return empty list when no matches found"


def test_recommendation_diversity(sample_data):
    """Test that recommendations are diverse (not all identical)."""
    df, features = sample_data
    
    test_title = "Fullmetal Alchemist: Brotherhood"
    result = recommendation_system(test_title, df, features, top_n=5)
    
    if isinstance(result, pd.DataFrame) and len(result) > 1:
        # Check that not all recommendations are the same
        unique_names = result['name'].nunique()
        assert unique_names == len(result), "All recommendations should be unique"
        
        # Check that similarity scores vary (not all identical)
        unique_scores = result['similarity'].nunique()
        assert unique_scores > 1, "Similarity scores should vary across recommendations"


def test_recommendation_sorted_by_similarity(sample_data):
    """Test that recommendations are sorted by similarity (descending)."""
    df, features = sample_data
    
    test_title = "Death Note"
    result = recommendation_system(test_title, df, features, top_n=10)
    
    if isinstance(result, pd.DataFrame) and len(result) > 1:
        similarities = result['similarity'].values
        # Check that similarities are in descending order
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), \
            "Recommendations should be sorted by similarity (descending)"


if __name__ == "__main__":
    # Allow running tests directly with: python test_rec.py
    pytest.main([__file__, "-v"])
