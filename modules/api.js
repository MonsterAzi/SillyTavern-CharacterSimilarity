import { getRequestHeaders } from "../../../../../script.js";
import { state } from "./state.js";

const FIELDS_TO_EMBED = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example'];

export async function fetchEmbeddings(characters) {
    let koboldUrl = state.setting.koboldUrl.trim();
    
    // URL Validation
    if (!koboldUrl) {
        throw new Error('Please set the KoboldCpp URL in extension settings.');
    }
    if (!koboldUrl.startsWith('http://') && !koboldUrl.startsWith('https://')) {
        koboldUrl = 'http://' + koboldUrl;
    }

    // Prepare text payloads
    const textsToEmbed = [];
    for (const char of characters) {
        const combinedText = FIELDS_TO_EMBED.map(field => char[field] || '').join('\n').trim();
        if (combinedText) {
            textsToEmbed.push({ avatar: char.avatar, text: combinedText });
        }
    }

    if (textsToEmbed.length === 0) return [];

    const response = await fetch('/api/backends/kobold/embed', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            items: textsToEmbed.map(item => item.text),
            server: koboldUrl,
        }),
    });

    if (!response.ok) {
        throw new Error(`Server failed to get embeddings. Status: ${response.status}`);
    }

    const data = await response.json();
    
    if (!data.embeddings || data.embeddings.length !== textsToEmbed.length) {
        throw new Error('Received incomplete response from embedding server.');
    }

    // Map back to avatars
    const result = [];
    for (let i = 0; i < data.embeddings.length; i++) {
        const embedding = data.embeddings[i];
        if (embedding && Array.isArray(embedding) && embedding.length > 0) {
            result.push({ 
                avatar: textsToEmbed[i].avatar, 
                embedding: embedding 
            });
        }
    }
    
    return result;
}