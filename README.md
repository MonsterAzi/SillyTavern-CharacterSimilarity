# Character Similarity

This extension analyzes your character library using text embeddings to help you discover two key insights: which characters are statistically unique or generic, and which characters are highly similar.

## Features

*   Leverages a local KoboldCpp API for fast and private text embedding generation.
*   **Uniqueness Ranking:** Calculates a 'uniqueness score' for every character by comparing their embedding to the average of your entire library. Sort the list to easily find your most generic or most unique characters.
*   **Clustering:** Groups characters into clusters of high similarity based on the cosine similarity of their embeddings. Perfect for identifying and cleaning up duplicate or very similar character cards.
*   Adjustable similarity threshold for fine-tuning cluster results.
*   Performs heavy calculations in a background worker to keep the UI responsive.

## Installation and Usage

### Installation

Install using SillyTavern's built-in extension installer.

### Initial Setup

1.  Navigate to the main **Extensions settings panel** (the plug icon on the right).
2.  Find the "Character Similarity" section.
3.  Enter the base URL for your running KoboldCpp instance (e.g., `http://127.0.0.1:5001`).

### Workflow

1.  Open the character management screen (the multiple-person icon on the left).
2.  Click the **"Find Similar Characters" button** (it looks like a network diagram) in the top bar to open the main panel.
3.  **Load Embeddings:** Click the `Load Embeddings` button. This process can take several minutes for a large library. Please wait for the "Success" notification before proceeding.
4.  **Choose an Analysis:**
    *   **To find unique/generic characters:**
        1.  Select the **Uniqueness** tab.
        2.  Click `Calculate Uniqueness`.
        3.  Use the arrow button to sort the results. Arrow down shows most unique first; arrow up shows most generic first.
    *   **To find duplicate characters:**
        1.  Select the **Clustering** tab.
        2.  Adjust the `Threshold` slider (higher values require characters to be more similar to be grouped).
        3.  Click `Calculate Clusters`. The results will show groups of similar characters.

## Current Issues

*   It does currently give an annoying error on start-up
*   It currently only works with KoboldCpp, since I didn't bother to make the rest work.

## Prerequisites

*   SillyTavern >= 1.11.7
*   A running instance of KoboldCpp with embedding models loaded.
