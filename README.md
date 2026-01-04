# Character Similarity Analysis for SillyTavern (v1.4.2)

This extension provides advanced semantic analysis tools for your SillyTavern character library. By leveraging text embeddings, it allows users to quantify character uniqueness, find semantically similar characters, perform conceptual searches, and predict character ratings based on personal preferences using machine learning.

## Features

### Semantic Uniqueness Analysis
The extension analyzes the text data of every character in your library (name, description, personality, scenario, and messages) to determine how unique they are compared to the rest of your collection. It supports multiple statistical outlier detection algorithms:

*   **Global Mean Distance:** Calculates the average distance from the center of your library's semantic space.
*   **kNN (k-Nearest Neighbors):** Measures density based on the distance to the closest neighboring characters.
*   **LOF (Local Outlier Factor):** Compares the local density of a character to the densities of its neighbors.
*   **Isolation Forest:** Randomly partitions data to isolate anomalies; easier isolation implies higher uniqueness.
*   **ECOD & HBOS:** Fast, parameter-free statistical methods for outlier detection.
*   **LUNAR:** A graph neural network approach for detecting anomalies.
*   **LODA:** An ensemble of weak histograms.

### Recommendation Engine & Predictive Ratings
The system includes a built-in recommender system. You can manually rate characters from 0.5 to 5 stars. The extension then trains a Gradient Boosting Regressor (Quantile Regression) on your local machine to learn your preferences based on character writing styles and themes.

*   **Manual Ratings:** Displayed as Gold stars.
*   **Predicted Ratings:** Displayed as Light Blue stars for unrated characters.
*   **Confidence Intervals:** A visual bar beneath predicted stars indicates the model's certainty. A narrower bar implies higher confidence in the prediction.

### Hybrid Semantic Search
The search bar performs a hybrid query. It combines standard fuzzy text matching (finding names similar to your query) with semantic vector search. This allows you to search for concepts (e.g., "medieval warrior" or "sad story") and retrieve relevant characters even if those exact words do not appear in their definitions.

### Similarity Recommendations
When viewing a specific character's details within the extension panel, a "Similar Characters" list is generated. This identifies other cards in your library that share similar themes, personalities, or writing styles.

### Optimization & Caching
*   **Live Caching:** Text embeddings are hashed and cached locally. The extension only requests embeddings for characters that have been added or modified since the last run.
*   **Background Processing:** Heavy mathematical operations (training the regressor or calculating isolation forests) are handled efficiently to minimize UI blocking.

## Prerequisites & Limitations

**KoboldCpp Required**
This extension relies on an embedding endpoint to convert text into vector data. Currently, it is explicitly designed to work with the embedding API provided by **KoboldCpp**.

*   You must have a running instance of KoboldCpp.
*   You must load a model that supports embedding or use a dedicated embedding model (e.g., `nomic-embed-text-v1.5.GGUF`) alongside your main model if KoboldCpp supports it, or simply use your chat model if it is capable of generating decent embeddings.
*   **Limitation:** While other backends (like Ooba or pure OpenAI proxies) might offer embedding endpoints, this extension is currently hardcoded to interface with the specific structure of the KoboldCpp API (`/api/backends/kobold/embed`).

## Installation

1.  Navigate to the `extensions` directory within your SillyTavern installation.
2.  Clone this repository or extract the archive into a folder named `character_similarity`.
3.  Restart SillyTavern.
4.  Open the Extensions menu in SillyTavern and ensure the extension is enabled.

## Usage

### Configuration
1.  Open the Extensions settings panel in SillyTavern.
2.  Locate **Character Similarity**.
3.  **KoboldCpp URL:** Enter the full address of your KoboldCpp instance (e.g., `http://127.0.0.1:5001`).
4.  **Clear Cache:** If you change your embedding model, you must click this button to invalidate old vectors, as vectors from different models are not compatible.

### The Analysis Panel
Access the main interface by clicking the diagram icon (usually found near the character search bar or the top menu).

#### 1. Uniqueness Tab
*   **Method Selection:** Choose an algorithm from the dropdown (e.g., Isolation Forest, kNN).
*   **Parameters:** Adjust `N` (neighbors) or `Trees` depending on the selected algorithm.
*   **Sort:** Toggle the arrow icon to sort characters by most or least unique.

#### 2. Characters Tab
*   **Grid View:** Displays all characters with their ratings.
*   **Filtering:** Use the search bar to filter by name or semantic concept.
*   **Sorting:** Sort the grid by Name, Uniqueness, Token Count, Rating, or Prediction Confidence.
*   **Interaction:** Clicking a character card opens the Details View.

#### 3. Details View
*   Displays the character's full metadata (Description, First Message, etc.).
*   **Rating:** Click the stars to assign a manual rating. Click the "Reset" (rotate left) icon to clear a manual rating and revert to the AI prediction.
*   **Similar Characters:** A horizontal scroll list showing characters most similar to the current one.
*   **Open Chat:** Clicking the large character portrait will close the extension panel and select that character in the main SillyTavern interface.