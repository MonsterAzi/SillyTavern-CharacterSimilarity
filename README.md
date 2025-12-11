# Character Similarity

This extension analyzes your character library using text embeddings to help you discover two key insights: which characters are statistically unique or generic, and which characters are highly similar.

## Improvements in V2.0
- Refactored into ES Modules for better maintainability.
- Replaced heavy clustering library with a native Graph-based algorithm.
- Improved code readability and error handling.

## Features

*   **Uniqueness Ranking:** Calculates a 'uniqueness score' for every character by comparing their embedding to the average of your entire library.
*   **Clustering:** Groups characters into clusters of high similarity.
*   **Private & Local:** Uses your local KoboldCpp API.
*   **Background Processing:** Heavy math is offloaded to a Web Worker to keep UI snappy.

## Installation

1. Install via SillyTavern's "Install Extension" feature or clone into `public/scripts/extensions/character_similarity`.
2. Ensure you have a KoboldCpp instance running with an embedding model loaded (e.g., `all-MiniLM-L6-v2`).

## Usage

1. Go to **Extensions Settings** -> **Character Similarity** and set your KoboldCpp URL (e.g., `http://127.0.0.1:5001`).
2. Open the Character Management panel.
3. Click the **Network Icon** button.
4. Click **Load Embeddings**.
5. Switch tabs to calculate Uniqueness or Clusters.