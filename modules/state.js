import { extension_settings } from "../../../../extensions.js";
import { saveSettingsDebounced } from "../../../../../script.js";

export const extensionName = "character_similarity";

const defaultSettings = {
    koboldUrl: 'http://127.0.0.1:5001',
    clusterThreshold: 0.95,
};

// Singleton state management
class ExtensionState {
    constructor() {
        this.embeddings = new Map();
        this.uniquenessResults = [];
        this.clusterResults = [];
    }

    initSettings() {
        extension_settings[extensionName] = extension_settings[extensionName] || {};
        // Merge defaults
        Object.assign(defaultSettings, extension_settings[extensionName]);
        Object.assign(extension_settings[extensionName], defaultSettings);
    }

    get setting() {
        return extension_settings[extensionName];
    }

    saveSettings() {
        saveSettingsDebounced();
    }

    clearData() {
        this.embeddings.clear();
        this.uniquenessResults = [];
        this.clusterResults = [];
    }
}

export const state = new ExtensionState();