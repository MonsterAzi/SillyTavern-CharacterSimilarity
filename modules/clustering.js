/**
 * Creates a Worker to perform clustering.
 * We use a clean Graph-based connected components approach
 * instead of including a massive minified library.
 */
export function createClusteringWorker() {
    const workerCode = `
    self.onmessage = function(event) {
        const { embeddings, threshold } = event.data;
        const items = embeddings; // Array of { avatar, embedding }
        const count = items.length;
        
        // 1. Build Adjacency List (Graph) based on Cosine Similarity
        // Complexity: O(N^2) - heavy but unavoidable for all-pair comparison
        const adj = new Map(); // index -> [indices]
        
        for(let i = 0; i < count; i++) adj.set(i, []);

        for (let i = 0; i < count; i++) {
            for (let j = i + 1; j < count; j++) {
                const sim = cosineSimilarity(items[i].embedding, items[j].embedding);
                if (sim >= threshold) {
                    adj.get(i).push(j);
                    adj.get(j).push(i);
                }
            }
        }

        // 2. Find Connected Components (BFS)
        const visited = new Set();
        const groups = [];

        for (let i = 0; i < count; i++) {
            if (!visited.has(i)) {
                const group = [];
                const queue = [i];
                visited.add(i);

                while(queue.length > 0) {
                    const node = queue.shift();
                    group.push({
                        avatar: items[node].avatar,
                        // We pass back partial info to main thread
                        embedding: items[node].embedding 
                    });

                    const neighbors = adj.get(node);
                    for(const neighbor of neighbors) {
                        if(!visited.has(neighbor)) {
                            visited.add(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
                
                // Only care about groups with actual content
                if (group.length > 0) {
                    groups.push(group);
                }
            }
        }

        self.postMessage(groups);
    };

    function cosineSimilarity(vecA, vecB) { 
        let dotProduct = 0.0; 
        let normA = 0.0; 
        let normB = 0.0; 
        for (let i = 0; i < vecA.length; i++) { 
            dotProduct += vecA[i] * vecB[i]; 
            normA += vecA[i] * vecA[i]; 
            normB += vecB[i] * vecB[i]; 
        } 
        if (normA === 0 || normB === 0) return 0; 
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)); 
    }
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
}