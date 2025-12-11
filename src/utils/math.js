export function cosineSimilarity(vecA, vecB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    const len = Math.min(vecA.length, vecB.length);
    
    for (let i = 0; i < len; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function normalizeSimilarity(similarity) {
    return (similarity + 1) / 2;
}

export function calculateMeanEmbedding(embeddings) {
    if (!embeddings || embeddings.length === 0) return null;
    
    const embeddingCount = embeddings.length;
    const dimension = embeddings[0].length;
    const meanEmbedding = new Array(dimension).fill(0);

    for (const vector of embeddings) {
        for (let i = 0; i < dimension; i++) {
            meanEmbedding[i] += vector[i];
        }
    }

    for (let i = 0; i < dimension; i++) {
        meanEmbedding[i] /= embeddingCount;
    }
    return meanEmbedding;
}

export function manhattanDistance(vecA, vecB) {
    let distance = 0;
    for (let i = 0; i < vecA.length; i++) {
        distance += Math.abs(vecA[i] - vecB[i]);
    }
    return distance;
}