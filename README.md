# Character Similarity & Analysis Extension

A SillyTavern extension that analyzes your character library using AI text embeddings. It helps you identify unique characters, generic tropes, and duplicate cards using advanced algorithms (KNN, Harmonic Mean).

## Features

*   **Offline / Multi-Cache Support:** Export your embeddings to JSON. If your API is offline, simply Import the cache file to perform analysis.
*   **Algorithms:**
    *   **Global Mean:** Finds how different a character is from the "average" of your library.
    *   **k-Nearest Neighbors (KNN):** Detects outliers by measuring the density of similar characters around a specific card.
    *   **Harmonic Mean:** A weighted variation that emphasizes closer matches.
*   **Clustering:** Groups duplicates or highly similar variations of characters.
*   **Performance:** Runs math-heavy operations in a background Web Worker to keep SillyTavern smooth.

## Usage

1.  **Setup:** Go to Extensions > Settings > Character Similarity and ensure your KoboldCpp URL is correct.
2.  **Open Panel:** Click the "Network" icon (lines and dots) in the top bar.
3.  **Data Management:**
    *   Click **Fetch from API** to generate embeddings (requires running KoboldCpp).
    *   Click **Export Cache** to save embeddings to your PC.
    *   Click **Import Cache** to load them back later without needing the API.
4.  **Analysis:** Go to the Uniqueness tab, select an Algorithm (KNN is recommended), and click Calculate.