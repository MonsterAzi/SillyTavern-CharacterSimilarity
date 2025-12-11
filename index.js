import { EmbeddingManager } from "./src/core/EmbeddingManager.js";
import { AnalysisEngine } from "./src/core/Analysis.js";
import { CharacterSimilarityPanel } from "./src/ui/Panel.js";
import { eventSource, event_types } from "../../../script.js";

// Initialize Managers
const embeddingManager = new EmbeddingManager();
const analysisEngine = new AnalysisEngine(embeddingManager);

// Initialize UI
const uiPanel = new CharacterSimilarityPanel(embeddingManager, analysisEngine);

// Optional: Handle App Ready event if specific initialization is needed
eventSource.on(event_types.APP_READY, () => {
    // If you wanted to auto-populate a list on startup, do it here.
});