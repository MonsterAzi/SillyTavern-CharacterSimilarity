// Import necessary functions from Silly-Tavern's core scripts.
import { extension_settings } from "../../../extensions.js";
import { characters, getThumbnailUrl, saveSettingsDebounced, eventSource, event_types, getRequestHeaders } from "../../../../script.js";

const extensionName = "character_similarity";

const defaultSettings = {
    koboldUrl: 'http://127.0.0.1:5001',
    clusterThreshold: 0.95,
    uniquenessAlgorithm: 'global_mean', // 'global_mean', 'knn', 'lof', 'iforest'
    uniquenessK: 20,    // For k-NN and LOF
    uniquenessTrees: 100 // For Isolation Forest
};

const characterEmbeddings = new Map();
let uniquenessResults = [];
let clusterResults = [];

const fieldsToEmbed = [
    'name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example',
];

// Helper to keep UI updated
function updateSettings(key, value) {
    extension_settings[extensionName][key] = value;
    saveSettingsDebounced();
}

function populateCharacterList() {
    const sortedCharacters = characters.slice().sort((a, b) => a.name.localeCompare(b.name));
    const characterListHtml = sortedCharacters.map(char => `
        <div class="charSim-character-item" data-avatar="${char.avatar}">
            <img src="${getThumbnailUrl('avatar', char.avatar)}" alt="${char.name}'s avatar">
            <span class="charSim-name">${char.name}</span>
        </div>
    `).join('');
    $('#charSimUniquenessList').html(characterListHtml);
}

function renderUniquenessList() {
    if (uniquenessResults.length === 0) {
        populateCharacterList();
        return;
    }
    const isDescending = $('#charSimSortBtn').hasClass('fa-arrow-down');
    const sortedList = [...uniquenessResults];
    
    // Sort logic depends on algorithm, but generally higher score = more unique/outlier
    sortedList.sort((a, b) => isDescending ? b.score - a.score : a.score - b.score);

    const characterListHtml = sortedList.map(result => `
        <div class="charSim-character-item" data-avatar="${result.avatar}">
            <img src="${getThumbnailUrl('avatar', result.avatar)}" alt="${result.name}'s avatar">
            <span class="charSim-name">${result.name}</span>
            <div class="charSim-score">${result.score.toFixed(4)}</div>
        </div>
    `).join('');
    $('#charSimUniquenessList').html(characterListHtml);
}

function renderClusterList() {
    const container = $('#charSimClusteringList');
    container.html('');
    const similarGroups = clusterResults.filter(group => group.members.length > 1);
    if (similarGroups.length === 0) {
        container.html('<p class="charSim-no-results">No similar character groups found at this threshold.</p>');
        return;
    }
    similarGroups.forEach((groupData, index) => {
        const groupEl = $('<div class="charSim-cluster-group"></div>');
        const headerEl = $(`<div class="charSim-cluster-header"><span>Cluster Variance:</span><span class="charSim-score">${groupData.clusterUniqueness.toFixed(4)}</span></div>`);
        groupEl.append(headerEl);
        const sortedMembers = groupData.members.sort((a, b) => a.intraClusterDistance - b.intraClusterDistance);
        sortedMembers.forEach(member => {
            const charEl = $(`<div class="charSim-character-item" data-avatar="${member.avatar}"><img src="${getThumbnailUrl('avatar', member.avatar)}" alt="${member.name}'s avatar"><span class="charSim-name">${member.name}</span><div class="charSim-score" title="Distance from cluster average">${member.intraClusterDistance.toFixed(4)}</div></div>`);
            groupEl.append(charEl);
        });
        container.append(groupEl);
        if (index < similarGroups.length - 1) container.append('<hr class="charSim-delimiter">');
    });
}

async function onEmbeddingsLoad() {
    let koboldUrl = extension_settings[extensionName].koboldUrl.trim();
    if (!koboldUrl) {
        toastr.warning('Please set the KoboldCpp URL in the extension settings. Use a network IP, not localhost, for multi-device access.');
        return;
    }
    if (!koboldUrl.startsWith('http://') && !koboldUrl.startsWith('https://')) {
        koboldUrl = 'http://' + koboldUrl;
    }

    const buttons = $('#charSimLoadBtn, #charSimCalcUniquenessBtn, #charSimCalcClustersBtn');
    let toastId = null;

    try {
        buttons.prop('disabled', true);
        characterEmbeddings.clear();
        uniquenessResults = [];
        clusterResults = [];
        toastId = toastr.info(`Preparing ${characters.length} characters for embedding...`, 'Loading Embeddings', { timeOut: 0, extendedTimeOut: 0 });

        const textsToEmbed = [];
        for (const char of characters) {
            const combinedText = fieldsToEmbed.map(field => char[field] || '').join('\n').trim();
            if (combinedText) {
                textsToEmbed.push({ avatar: char.avatar, text: combinedText });
            }
        }

        toastr.info(`Sending ${textsToEmbed.length} characters to the API... This may take a while.`, 'Loading Embeddings', { toastId: toastId, timeOut: 0, extendedTimeOut: 0 });

        const response = await fetch('/api/backends/kobold/embed', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                items: textsToEmbed.map(item => item.text),
                server: koboldUrl,
            }),
        });

        if (!response.ok) {
            throw new Error(`The SillyTavern server failed to get embeddings. Status: ${response.status}. Check the server console for details.`);
        }

        const data = await response.json();
        if (!data.embeddings || data.embeddings.length !== textsToEmbed.length) {
            throw new Error('Received an invalid or incomplete response from the server proxy.');
        }

        for (let i = 0; i < data.embeddings.length; i++) {
            const avatar = textsToEmbed[i].avatar;
            const embedding = data.embeddings[i];
            if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
                console.warn(`Received empty embedding for character ${avatar}. Skipping.`);
                continue;
            }
            characterEmbeddings.set(avatar, embedding);
        }

        toastr.remove(toastId);
        toastr.success(`Successfully loaded embeddings for ${characterEmbeddings.size} characters.`);

    } catch (error) {
        if (toastId) toastr.remove(toastId);
        toastr.error(`Error loading embeddings: ${error.message}`, 'Error');
        console.error("Embedding error details:", error);
    } finally {
        buttons.prop('disabled', false);
    }
}

// --- WORKER MANAGEMENT ---

function createWorker() {
    // We bundle the clustering library (simple-clustering/graphs) + our math logic into one string
    // This avoids external file dependencies that might not load correctly in all ST setups.
    const workerCode = `
        // --- 3rd Party Graph/Cluster Lib (Minified approximation for brevity) ---
        const e={52:e=>{function n(e,n){var t=e.nodes.slice(),r=[];if(!t.length)return r;for(var o=[],i=null;t.length;)for(o.length||(i&&r.push(i),i={nodes:(o=[t.pop()]).slice(),edges:{}});o.length;)for(var d=o.pop(),a=t.length-1;a>=0;a--){var s=t[a];if(e.edges[d.id]&&e.edges[d.id][s.id]>=n){o.push(s),i.nodes.push(s),i.edges[d.id]=i.edges[d.id]||{},i.edges[d.id][s.id]=e.edges[d.id][s.id],i.edges[s.id]=i.edges[s.id]||{},i.edges[s.id][d.id]=e.edges[s.id][d.id];var c=t.slice(0,a).concat(t.slice(a+1));t=c}}return i&&r.push(i),r}e.exports={create:function(e,n){var t={},r=1,o=e.map((function(e){return{id:r++,data:e}}));return o.forEach((function(e){t[e.id]=t[e.id]||{},o.forEach((function(r){if(e!==r){var o=n(e.data,r.data);t[e.id][r.id]=o,t[r.id]=t[r.id]||{},t[r.id][e.id]=o}}))})),{nodes:o,edges:t}},data:function(e){return e.data},connected:n,divide:function(e,t,r){for(var o=1/0,i=0,d=function(e){var n=0;return e.nodes.forEach((function(t){e.nodes.forEach((function(r){var o=e.edges[t.id][r.id];o&&o>n&&(n=o)}))})),n}(e)+1,a=null,s=-2;s<r;s++){var c,u=n(e,c=-2==s?i:-1==s?d:(d+i)/2),f=u.length-t;if(f<o&&f>=0&&(o=f,a=u),u.length>t&&(d=c),u.length<t&&(i=c),u.length==t)break;if(i==d)break}return a},findCenter:function(e){var n=function(e){var n={};return e.nodes.forEach((function(e){n[e.id]={},n[e.id][e.id]=0})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){if(t!=r){var o=e.edges[t.id]&&e.edges[t.id][r.id];null==o&&(o=1/0),n[t.id][r.id]=o}}))})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){e.nodes.forEach((function(e){var o=n[r.id][t.id]+n[t.id][e.id];n[r.id][e.id]>o&&(n[r.id][e.id]=o)}))}))})),n}(e),t=1/0,r=null;return e.nodes.forEach((function(o){var i=0;e.nodes.forEach((function(e){var t=n[o.id][e.id];t>i&&(i=t)})),t>i&&(t=i,r=o)})),r},growFromNuclei:function(e,n){for(var t=n.map((function(e){return{nodes:[e],edges:{}}})),r=e.nodes.filter((function(e){return 0==n.filter((function(n){return e==n})).length})),o=0,i=t.length;r.length&&i;){i-=1;var d=t[o];o=(o+1)%t.length;var a=null,s=null,c=-1/0;if(d.nodes.forEach((function(n){r.forEach((function(t){var r=e.edges[n.id]&&e.edges[n.id][t.id];r&&r>c&&(a=n,s=t,c=r)}))})),a){var u=a,f=s;d.edges[u.id]=d.edges[u.id]||{},d.edges[u.id][f.id]=e.edges[u.id][f.id],d.edges[f.id]=d.edges[f.id]||{},d.edges[f.id][u.id]=e.edges[f.id][u.id],d.nodes.push(s),r=r.filter((function(e){return e!=s})),i=t.length}}return{graphs:t,orphans:r}}}},834:(e,n,t)=>{var r=t(52);e.exports=function(e,n){var t,o=r.create(e,(function(e,t){var r=n(e,t);if("number"!=typeof r||r<0)throw new Error("Similarity function did not yield a number in the range [0, +Inf) when comparing "+e+" to "+t+" : "+r);return r}));function i(e){return function(){return e.apply(this,Array.prototype.slice.call(arguments)).map((function(e){return e.nodes.map(r.data)}))}}function d(e,n){var t=n||1e3;return r.divide(o,e,t)}function a(e,n){var t=d(e,n);return t.sort((function(e,n){return n.nodes.length-e.nodes.length})),t.splice(e),t.map(r.findCenter)}return{groups:i(d),representatives:(t=a,function(){return t.apply(this,Array.prototype.slice.call(arguments)).map(r.data)}),similarGroups:i((function(e){return r.connected(o,e)})),evenGroups:function(e,n){for(var t=a(e),i=r.growFromNuclei(o,t),d=i.graphs.map((function(e){return e.nodes.map(r.data)}));i.orphans.length;){var s=r.data(i.orphans.pop());d.sort((function(e,n){return e.length-n.length})),d[0].push(s)}return d}}}}},n={};function t(r){var o=n[r];if(void 0!==o)return o.exports;var i=n[r]={exports:{}};return e[r](i,i.exports,t),i.exports}var cluster = t(834);

        // --- MATH HELPERS ---
        function cosineSimilarity(vecA, vecB) { 
            let dotProduct = 0.0; let normA = 0.0; let normB = 0.0; 
            for (let i = 0; i < vecA.length; i++) { dotProduct += vecA[i] * vecB[i]; normA += vecA[i] * vecA[i]; normB += vecB[i] * vecB[i]; } 
            if (normA === 0 || normB === 0) return 0; 
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)); 
        }

        function cosineDistance(vecA, vecB) {
            return 1 - cosineSimilarity(vecA, vecB);
        }

        // --- ALGORITHMS ---

        function calcGlobalMean(data) {
            // Calculate Mean Vector
            if (data.length === 0) return [];
            const dimension = data[0].embedding.length;
            const meanEmbedding = new Array(dimension).fill(0);
            for (const item of data) {
                for (let i = 0; i < dimension; i++) meanEmbedding[i] += item.embedding[i];
            }
            for (let i = 0; i < dimension; i++) meanEmbedding[i] /= data.length;

            return data.map(item => {
                let dist = 0;
                for (let i = 0; i < dimension; i++) dist += Math.abs(item.embedding[i] - meanEmbedding[i]); // L1 Distance to mean for consistency with prev versions
                // Or use cosine: dist = cosineDistance(item.embedding, meanEmbedding);
                return { avatar: item.avatar, score: dist };
            });
        }

        function calcKNN(data, k) {
            // O(N^2) Distance Matrix
            const n = data.length;
            const scores = [];
            
            for (let i = 0; i < n; i++) {
                const distances = [];
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    distances.push(cosineDistance(data[i].embedding, data[j].embedding));
                }
                distances.sort((a, b) => a - b);
                
                // Average distance to k nearest neighbors
                const actualK = Math.min(k, distances.length);
                let sum = 0;
                for (let x = 0; x < actualK; x++) sum += distances[x];
                const score = actualK > 0 ? sum / actualK : 0;
                scores.push({ avatar: data[i].avatar, score: score });
            }
            return scores;
        }

        function calcLOF(data, k) {
            const n = data.length;
            if (n < 2) return data.map(d => ({ avatar: d.avatar, score: 0 }));

            // 1. Calculate full distance matrix and find k-nearest neighbors for each
            const neighbors = []; // { index, distance }
            
            for (let i = 0; i < n; i++) {
                const dists = [];
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    dists.push({ idx: j, dist: cosineDistance(data[i].embedding, data[j].embedding) });
                }
                dists.sort((a, b) => a.dist - b.dist);
                
                // Get k-neighbors (ensure we don't go out of bounds)
                // Note: LOF definition of k usually allows for ties, we simplify here
                const actualK = Math.min(k, dists.length);
                neighbors.push({
                    kDist: dists[actualK - 1].dist, // k-distance(A)
                    neighbors: dists.slice(0, actualK) // N_k(A)
                });
            }

            // 2. Reachability Distance: reach_dist_k(A, B) = max(k_dist(B), dist(A,B))
            // 3. LRD (Local Reachability Density)
            const lrd = new Array(n).fill(0);
            
            for (let i = 0; i < n; i++) {
                const myNeighbors = neighbors[i].neighbors;
                let sumReachDist = 0;
                for (const neighbor of myNeighbors) {
                    const neighborIdx = neighbor.idx;
                    const distToNeighbor = neighbor.dist;
                    const neighborsKDist = neighbors[neighborIdx].kDist;
                    sumReachDist += Math.max(neighborsKDist, distToNeighbor);
                }
                const avgReachDist = sumReachDist / myNeighbors.length;
                lrd[i] = avgReachDist > 1e-9 ? 1 / avgReachDist : 0; // Avoid division by zero
            }

            // 4. LOF Score
            const lofScores = [];
            for (let i = 0; i < n; i++) {
                const myNeighbors = neighbors[i].neighbors;
                let sumNeighborLrd = 0;
                for (const neighbor of myNeighbors) {
                    sumNeighborLrd += lrd[neighbor.idx];
                }
                const avgNeighborLrd = sumNeighborLrd / myNeighbors.length;
                const score = lrd[i] > 1e-9 ? avgNeighborLrd / lrd[i] : 0;
                lofScores.push({ avatar: data[i].avatar, score: score });
            }

            return lofScores;
        }

        function calcIForest(data, numTrees) {
            // Simplified Isolation Forest for JS
            // Recursive path length calculation
            const n = data.length;
            if (n === 0) return [];
            const dims = data[0].embedding.length;
            const limit = Math.ceil(Math.log2(n));

            // Helper to get c(n) - average path length of unsuccessful search in BST
            function c(size) {
                if (size <= 1) return 0;
                return 2 * (Math.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size);
            }

            // Tree Node structure: { splitAtt, splitVal, left, right, size }
            function buildTree(indices, currentHeight) {
                if (currentHeight >= limit || indices.length <= 1) {
                    return { size: indices.length, type: 'leaf' };
                }
                
                // Randomly select attribute
                const att = Math.floor(Math.random() * dims);
                // Find min/max for this attribute in current subset
                let min = Infinity, max = -Infinity;
                for (const idx of indices) {
                    const val = data[idx].embedding[att];
                    if (val < min) min = val;
                    if (val > max) max = val;
                }

                if (min === max) {
                    return { size: indices.length, type: 'leaf' };
                }

                // Random split value
                const splitVal = Math.random() * (max - min) + min;

                const leftIdx = [];
                const rightIdx = [];
                for (const idx of indices) {
                    if (data[idx].embedding[att] < splitVal) leftIdx.push(idx);
                    else rightIdx.push(idx);
                }

                return {
                    type: 'node',
                    splitAtt: att,
                    splitVal: splitVal,
                    left: buildTree(leftIdx, currentHeight + 1),
                    right: buildTree(rightIdx, currentHeight + 1)
                };
            }

            function pathLength(itemIdx, node, e) {
                if (node.type === 'leaf') {
                    return e + c(node.size);
                }
                const val = data[itemIdx].embedding[node.splitAtt];
                if (val < node.splitVal) return pathLength(itemIdx, node.left, e + 1);
                return pathLength(itemIdx, node.right, e + 1);
            }

            // Build Forest
            const forest = [];
            const allIndices = Array.from({length: n}, (_, i) => i);
            
            // Subsampling is crucial for IForest performance and accuracy, 
            // but for small datasets (char libraries < 10000) we can use full batch or smaller subsample
            const sampleSize = Math.min(256, n); 

            for (let t = 0; t < numTrees; t++) {
                // Fisher-Yates shuffle for subsampling
                const shuffled = allIndices.slice();
                for (let i = shuffled.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
                }
                const sample = shuffled.slice(0, sampleSize);
                forest.push(buildTree(sample, 0));
            }

            // Calculate Scores
            const scores = [];
            const avgPathLenC = c(sampleSize);
            
            for (let i = 0; i < n; i++) {
                let totalPathLen = 0;
                for (let t = 0; t < numTrees; t++) {
                    totalPathLen += pathLength(i, forest[t], 0);
                }
                const avgLen = totalPathLen / numTrees;
                const score = Math.pow(2, -(avgLen / avgPathLenC));
                scores.push({ avatar: data[i].avatar, score: score });
            }

            return scores;
        }

        // --- MAIN WORKER HANDLER ---

        self.onmessage = function(event) {
            const { type, embeddings, threshold, algorithm, params } = event.data;
            const data = Array.from(embeddings.entries()).map(([avatar, embedding]) => ({ avatar, embedding }));
            
            if (type === 'cluster') {
                const clustering = cluster(data, (a, b) => {
                    const similarity = cosineSimilarity(a.embedding, b.embedding);
                    return (similarity + 1) / 2;
                });
                const groups = clustering.similarGroups(threshold);
                self.postMessage({ type: 'cluster_result', groups });
            } 
            else if (type === 'uniqueness') {
                let results = [];
                try {
                    switch (algorithm) {
                        case 'knn':
                            results = calcKNN(data, params.k || 5);
                            break;
                        case 'lof':
                            results = calcLOF(data, params.k || 20);
                            break;
                        case 'iforest':
                            results = calcIForest(data, params.trees || 100);
                            break;
                        case 'global_mean':
                        default:
                            results = calcGlobalMean(data);
                            break;
                    }
                    self.postMessage({ type: 'uniqueness_result', results });
                } catch (e) {
                    self.postMessage({ type: 'error', message: e.message });
                }
            }
        };
    `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
}

function runOnWorker(msgData, callback) {
    const worker = createWorker();
    worker.onmessage = (event) => {
        if (event.data.type === 'error') {
            toastr.error(`Calculation Error: ${event.data.message}`);
        } else {
            callback(event.data);
        }
        worker.terminate();
    };
    worker.onerror = (e) => {
        toastr.error(`Worker Error: ${e.message}`);
        worker.terminate();
    }
    worker.postMessage(msgData);
}

// --- BUTTON HANDLERS ---

function onCalculateUniqueness() {
    if (characterEmbeddings.size === 0) {
        toastr.warning('Please load character embeddings first.');
        return;
    }

    const algo = extension_settings[extensionName].uniquenessAlgorithm;
    const k = parseInt(extension_settings[extensionName].uniquenessK);
    const trees = parseInt(extension_settings[extensionName].uniquenessTrees);

    const buttons = $('#charSimLoadBtn, #charSimCalcUniquenessBtn, #charSimCalcClustersBtn');
    buttons.prop('disabled', true);
    
    toastr.info(`Calculating Uniqueness using ${algo.toUpperCase()}...`, 'Processing');

    runOnWorker({
        type: 'uniqueness',
        embeddings: characterEmbeddings,
        algorithm: algo,
        params: { k, trees }
    }, (data) => {
        uniquenessResults = data.results.map(r => {
             const char = characters.find(c => c.avatar === r.avatar);
             return char ? { ...r, name: char.name } : null;
        }).filter(r => r !== null);
        
        renderUniquenessList();
        toastr.success('Uniqueness calculation complete.');
        buttons.prop('disabled', false);
    });
}

function onCalculateClusters() {
    if (characterEmbeddings.size === 0) {
        toastr.warning('Please load character embeddings first.');
        return;
    }
    const threshold = extension_settings[extensionName].clusterThreshold;
    const buttons = $('#charSimLoadBtn, #charSimCalcUniquenessBtn, #charSimCalcClustersBtn');
    let toastId = toastr.info(`Calculating clusters at ${threshold.toFixed(2)} threshold...`, 'Clustering', { timeOut: 0, extendedTimeOut: 0 });
    buttons.prop('disabled', true);

    // We calculate library mean here just for the post-processing display stats
    // The actual clustering logic happens in worker
    function calculateMeanEmbedding(embeddings) {
        if (!embeddings || embeddings.length === 0) return null;
        const embeddingCount = embeddings.length;
        const dimension = embeddings[0].length;
        const meanEmbedding = new Array(dimension).fill(0);
        for (const vector of embeddings) {
            for (let i = 0; i < dimension; i++) meanEmbedding[i] += vector[i];
        }
        for (let i = 0; i < dimension; i++) meanEmbedding[i] /= embeddingCount;
        return meanEmbedding;
    }

    runOnWorker({
        type: 'cluster',
        embeddings: characterEmbeddings,
        threshold: threshold
    }, (data) => {
        const groups = data.groups;
        const libraryMeanEmbedding = calculateMeanEmbedding(Array.from(characterEmbeddings.values()));
        
        const dimension = libraryMeanEmbedding.length;
        const processedGroups = [];
        
        // Post-process to add stats (done on main thread as it is fast for reduced group count)
        for (const group of groups) {
            const memberEmbeddings = group.map(member => characterEmbeddings.get(member.avatar));
            const clusterMeanEmbedding = calculateMeanEmbedding(memberEmbeddings);
            if (!clusterMeanEmbedding) continue;
            let clusterUniqueness = 0;
            for (let i = 0; i < dimension; i++) clusterUniqueness += Math.abs(clusterMeanEmbedding[i] - libraryMeanEmbedding[i]);
            const members = group.map(member => {
                const char = characters.find(c => c.avatar === member.avatar);
                const embedding = characterEmbeddings.get(member.avatar);
                let intraClusterDistance = 0;
                for (let i = 0; i < dimension; i++) intraClusterDistance += Math.abs(embedding[i] - clusterMeanEmbedding[i]);
                return { avatar: member.avatar, name: char.name, intraClusterDistance };
            });
            processedGroups.push({ clusterUniqueness, members });
        }
        
        clusterResults = processedGroups.sort((a, b) => b.members.length - a.members.length);
        renderClusterList();
        toastr.remove(toastId);
        toastr.success(`Clustering complete. Found ${clusterResults.filter(g => g.members.length > 1).length} groups.`);
        buttons.prop('disabled', false);
    });
}

jQuery(() => {
    // --- SETTINGS ---
    extension_settings[extensionName] = extension_settings[extensionName] || {};
    Object.assign(defaultSettings, extension_settings[extensionName]);
    Object.assign(extension_settings[extensionName], defaultSettings);

    const settingsHtml = `
    <div class="character-similarity-settings">
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Character Similarity</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <div class="character-similarity_block">
                    <label for="kobold_url_input">KoboldCpp URL</label>
                    <input id="kobold_url_input" class="text_pole" type="text" value="${extension_settings[extensionName].koboldUrl}" placeholder="http://192.168.1.100:5001">
                    <small>MUST include http:// and be a network IP for multi-device access.</small>
                </div>
                <hr>
                <div class="character-similarity_block">
                    <label for="algo_select">Uniqueness Algorithm</label>
                    <select id="algo_select" class="text_pole">
                        <option value="global_mean">Global Mean (Centroid)</option>
                        <option value="knn">k-Nearest Neighbors (k-NN)</option>
                        <option value="lof">Local Outlier Factor (LOF)</option>
                        <option value="iforest">Isolation Forest</option>
                    </select>
                </div>
                <div class="character-similarity_block" id="k_param_block">
                    <label for="k_input">Neighbors (k) for k-NN/LOF</label>
                    <input id="k_input" class="text_pole" type="number" min="1" value="${extension_settings[extensionName].uniquenessK}">
                </div>
                <div class="character-similarity_block" id="tree_param_block" style="display:none">
                    <label for="tree_input">Number of Trees (Isolation Forest)</label>
                    <input id="tree_input" class="text_pole" type="number" min="10" step="10" value="${extension_settings[extensionName].uniquenessTrees}">
                </div>
            </div>
        </div>
    </div>`;

    $("#extensions_settings2").append(settingsHtml);

    // Settings Listeners
    $("#kobold_url_input").on("input", (e) => updateSettings('koboldUrl', e.target.value));
    $("#algo_select").val(extension_settings[extensionName].uniquenessAlgorithm).on("change", function() {
        const val = $(this).val();
        updateSettings('uniquenessAlgorithm', val);
        $('#k_param_block').toggle(val === 'knn' || val === 'lof');
        $('#tree_param_block').toggle(val === 'iforest');
    }).trigger('change');

    $("#k_input").on("input", (e) => updateSettings('uniquenessK', parseInt(e.target.value) || 5));
    $("#tree_input").on("input", (e) => updateSettings('uniquenessTrees', parseInt(e.target.value) || 100));

    // --- MAIN PANEL ---
    const panelHtml = `
    <div id="characterSimilarityPanel">
        <div class="charSimPanel-content">
            <div class="charSimPanel-header">
                <div class="fa-solid fa-grip drag-grabber"></div>
                <b>Character Similarity</b>
                <div id="charSimCloseBtn" class="fa-solid fa-circle-xmark floating_panel_close"></div>
            </div>
            <div class="charSimPanel-body">
                <div class="charSim-tabs">
                    <div class="charSim-tab-button active" data-tab="uniqueness">Uniqueness</div>
                    <div class="charSim-tab-button" data-tab="clustering">Clustering</div>
                </div>
                <div id="charSimUniquenessView" class="charSim-tab-pane active">
                    <div class="charSimPanel-controls">
                        <div id="charSimLoadBtn" class="menu_button">Load Embeddings</div>
                        <div id="charSimCalcUniquenessBtn" class="menu_button">Calculate Uniqueness</div>
                        <div class="spacer"></div>
                        <div id="charSimSortBtn" class="menu_button menu_button_icon fa-solid fa-arrow-down" title="Sort Descending"></div>
                    </div>
                    <div id="charSimUniquenessList" class="charSim-list-container"></div>
                </div>
                <div id="charSimClusteringView" class="charSim-tab-pane">
                    <div class="charSimPanel-controls">
                        <div id="charSimCalcClustersBtn" class="menu_button">Calculate Clusters</div>
                        <div class="spacer"></div>
                        <label for="charSimThresholdSlider">Threshold: <span id="charSimThresholdValue">${defaultSettings.clusterThreshold.toFixed(2)}</span></label>
                        <input type="range" id="charSimThresholdSlider" min="0.5" max="1.0" step="0.01" value="${defaultSettings.clusterThreshold}">
                    </div>
                    <div id="charSimClusteringList" class="charSim-list-container">
                        <p class="charSim-no-results">Load embeddings and click "Calculate Clusters" to see results.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>`;
    $('#movingDivs').append(panelHtml);

    // --- EVENT LISTENERS ---
    $('#charSimCloseBtn').on('click', () => $('#characterSimilarityPanel').removeClass('open'));
    $('#charSimLoadBtn').on('click', onEmbeddingsLoad);
    $('#charSimCalcUniquenessBtn').on('click', onCalculateUniqueness);
    $('#charSimCalcClustersBtn').on('click', onCalculateClusters);
    $('#charSimSortBtn').on('click', function() { $(this).toggleClass('fa-arrow-down fa-arrow-up'); $(this).attr('title', $(this).hasClass('fa-arrow-down') ? 'Sort Descending' : 'Sort Ascending'); renderUniquenessList(); });
    $('.charSim-tab-button').on('click', function() { const tab = $(this).data('tab'); $('.charSim-tab-button').removeClass('active'); $(this).addClass('active'); $('.charSim-tab-pane').removeClass('active'); $(`#charSim${tab.charAt(0).toUpperCase() + tab.slice(1)}View`).addClass('active'); });
    $('#charSimThresholdSlider').on('input', function() { const value = parseFloat($(this).val()); $('#charSimThresholdValue').text(value.toFixed(2)); updateSettings('clusterThreshold', value); });

    // --- CHARACTER PANEL BUTTON ---
    const openButton = document.createElement('div');
    openButton.id = 'characterSimilarityOpenBtn';
    openButton.classList.add('menu_button', 'fa-solid', 'fa-project-diagram', 'faSmallFontSquareFix');
    openButton.title = 'Find Similar Characters';
    openButton.addEventListener('click', () => { if (uniquenessResults.length > 0) renderUniquenessList(); else populateCharacterList(); $('#characterSimilarityPanel').addClass('open'); });
    const buttonContainer = document.getElementById('rm_buttons_container');
    if (buttonContainer) buttonContainer.append(openButton);
    else document.getElementById('form_character_search_form').insertBefore(openButton, document.getElementById('character_search_bar'));

    eventSource.on(event_types.APP_READY, populateCharacterList);
});