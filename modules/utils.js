/**
 * Calculates the mean vector of a set of embeddings.
 * @param {Array<Array<number>>} embeddings 
 * @returns {Array<number>|null}
 */
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

/**
 * Calculates L1 Distance (Manhattan) between two vectors.
 * Used for uniqueness scoring against the mean.
 */
export function calculateManhattanDistance(vecA, vecB) {
    let distance = 0;
    for (let i = 0; i < vecA.length; i++) {
        distance += Math.abs(vecA[i] - vecB[i]);
    }
    return distance;
}