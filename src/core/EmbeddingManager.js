import { characters, getThumbnailUrl, saveSettingsDebounced, eventSource, event_types, getRequestHeaders } from "../../../../script.js";
import { extension_settings } from "../../../../extensions.js";

const EXTENSION_NAME = "character_similarity";
const FIELDS_TO_EMBED = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example'];

export class EmbeddingManager {
    constructor() {
        this.embeddings = new Map(); // avatar -> vector
        this.charactersMap = new Map(); // avatar -> character object
        this.isLoading = false;
    }

    getKoboldUrl() {
        let url = extension_settings[EXTENSION_NAME]?.koboldUrl?.trim() || 'http://127.0.0.1:5001';
        if (!url.startsWith('http')) url = 'http://' + url;
        return url;
    }

    async fetchEmbeddings() {
        if (this.isLoading) return;
        this.isLoading = true;
        this.embeddings.clear();
        this.charactersMap.clear();

        const koboldUrl = this.getKoboldUrl();
        const textsToEmbed = [];

        // Prepare data
        for (const char of characters) {
            const combinedText = FIELDS_TO_EMBED.map(field => char[field] || '').join('\n').trim();
            if (combinedText) {
                textsToEmbed.push({ avatar: char.avatar, text: combinedText });
                this.charactersMap.set(char.avatar, char);
            }
        }

        try {
            const response = await fetch('/api/backends/kobold/embed', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({
                    items: textsToEmbed.map(item => item.text),
                    server: koboldUrl,
                }),
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const data = await response.json();
            if (!data.embeddings || data.embeddings.length !== textsToEmbed.length) {
                throw new Error('Invalid response format');
            }

            // Map results back to avatars
            for (let i = 0; i < data.embeddings.length; i++) {
                const embedding = data.embeddings[i];
                if (embedding && Array.isArray(embedding) && embedding.length > 0) {
                    this.embeddings.set(textsToEmbed[i].avatar, embedding);
                }
            }
            
            return true;
        } catch (error) {
            console.error("Embedding error:", error);
            toastr.error(`Error loading embeddings: ${error.message}`, 'Error');
            return false;
        } finally {
            this.isLoading = false;
        }
    }

    getEmbeddings() {
        return this.embeddings;
    }

    getCharacter(avatar) {
        return this.charactersMap.get(avatar);
    }
}