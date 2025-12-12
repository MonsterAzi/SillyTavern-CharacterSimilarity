// --- Imports ---
import { extension_settings } from "../../../extensions.js";
import { 
    characters, 
    getThumbnailUrl, 
    saveSettingsDebounced, 
    eventSource, 
    event_types, 
    getRequestHeaders 
} from "../../../../script.js";

// --- Constants & Config ---
const EXTENSION_NAME = "character_similarity";
const CACHE_KEY = "character_similarity_cache";
const DEFAULT_SETTINGS = {
    koboldUrl: 'http://127.0.0.1:5001',
    uniquenessMethod: 'mean',
    uniquenessN: 20,
};

// Fields to use for embedding generation
const FIELDS_TO_EMBED = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example'];

/**
 * Helper: SHA-256 Hashing for Text Comparison
 */
async function hashText(message) {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
}

/**
 * Service Layer: Handles communication with embedding providers and Caching.
 */
class SimilarityService {
    constructor(settings) {
        this.settings = settings;
    }

    get koboldUrl() {
        let url = this.settings.koboldUrl.trim();
        if (!url) return null;
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            url = 'http://' + url;
        }
        return url;
    }

    /**
     * Determines the current model name by sending an empty query.
     */
    async getModelName() {
        const url = this.koboldUrl;
        if (!url) return "unknown_model";

        try {
            const response = await fetch('/api/backends/kobold/embed', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({
                    items: [""], // Empty query
                    server: url,
                }),
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.model) return data.model;
            }
        } catch (e) {
            console.warn("Failed to probe model name, using default.", e);
        }
        return "default_model";
    }

    /**
     * Updates the Live Model cache.
     * @param {Array<{avatar:string, text:string, hash:string}>} items 
     */
    async updateLiveCache(items) {
        const url = this.koboldUrl;
        if (!url) throw new Error("KoboldCpp URL is not configured.");

        // 1. Identify Model
        const modelName = await this.getModelName();
        
        // 2. Load Cache
        if (!extension_settings[CACHE_KEY]) extension_settings[CACHE_KEY] = {};
        if (!extension_settings[CACHE_KEY][modelName]) extension_settings[CACHE_KEY][modelName] = {};
        
        const currentCache = extension_settings[CACHE_KEY][modelName];
        const toRequest = [];
        const toRequestIndices = [];
        
        // 3. Check Cache
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            const cached = currentCache[item.avatar];

            // If missing or hash mismatch, we need to fetch
            if (!cached || cached.hash !== item.hash || !cached.vector) {
                toRequest.push(item);
                toRequestIndices.push(i);
            }
        }

        // 4. Batch Request (only if needed)
        if (toRequest.length > 0) {
            toastr.info(`Updating live cache (${modelName}): Embedding ${toRequest.length} items...`);

            const response = await fetch('/api/backends/kobold/embed', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({
                    items: toRequest.map(i => i.text),
                    server: url,
                }),
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}.`);
            }

            const data = await response.json();
            if (!data.embeddings) throw new Error("Invalid response format.");

            // 5. Update Persistent Cache
            data.embeddings.forEach((vec, idx) => {
                if (vec && vec.length > 0) {
                    const originalItem = toRequest[idx];
                    currentCache[originalItem.avatar] = {
                        hash: originalItem.hash,
                        vector: vec
                    };
                }
            });

            // Persist
            extension_settings[CACHE_KEY][modelName] = currentCache;
            saveSettingsDebounced();
            toastr.success(`Updated cache for ${modelName}.`);
        }
    }

    /**
     * Retrieves ALL valid model caches.
     * A model cache is valid ONLY if it contains up-to-date embeddings for ALL provided items.
     * @param {Array<{avatar:string, hash:string}>} items 
     * @returns {Array<{modelName: string, map: Map<string, number[]>}>}
     */
    getValidCaches(items) {
        if (!extension_settings[CACHE_KEY]) return [];

        const validCaches = [];
        const allModels = Object.keys(extension_settings[CACHE_KEY]);

        for (const modelName of allModels) {
            const cache = extension_settings[CACHE_KEY][modelName];
            const map = new Map();
            let isValid = true;

            for (const item of items) {
                const entry = cache[item.avatar];
                
                // If entry missing, or hash mismatch (text changed), this model is incomplete
                if (!entry || entry.hash !== item.hash || !entry.vector) {
                    isValid = false;
                    break; 
                }
                map.set(item.avatar, entry.vector);
            }

            if (isValid) {
                validCaches.push({ modelName, map });
            }
        }

        return validCaches;
    }

    clearCache() {
        extension_settings[CACHE_KEY] = {};
        saveSettingsDebounced();
        toastr.success("Embedding cache cleared.");
    }
}

/**
 * Compute Engine: Handles Math Algorithms
 */
class ComputeEngine {
    
    // --- Basic Math ---

    static calculateMeanVector(vectors) {
        if (!vectors || vectors.length === 0) return null;
        const count = vectors.length;
        const dim = vectors[0].length;
        const mean = new Float32Array(dim).fill(0);
        
        for (const vec of vectors) {
            for (let i = 0; i < dim; i++) mean[i] += vec[i];
        }
        for (let i = 0; i < dim; i++) mean[i] /= count;
        return mean;
    }

    static calculateEuclideanDistance(vecA, vecB) {
        let sum = 0;
        for (let i = 0; i < vecA.length; i++) {
            const diff = vecA[i] - vecB[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    static normalizeScores(results) {
        if (!results || results.length === 0) return [];
        const scores = results.map(r => r.distance);
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const range = max - min;
        
        if (range === 0) return results.map(r => ({ ...r, distance: 0 }));

        return results.map(r => ({
            ...r,
            distance: (r.distance - min) / range
        }));
    }

    // --- Algorithms ---

    static computeGlobalMeanUniqueness(embeddingMap) {
        const vectors = Array.from(embeddingMap.values());
        const mean = this.calculateMeanVector(vectors);
        if (!mean) return [];

        const results = [];
        for (const [avatar, vec] of embeddingMap.entries()) {
            const distance = this.calculateEuclideanDistance(vec, mean);
            results.push({ avatar, distance });
        }
        return results;
    }

    static computeKNNUniqueness(embeddingMap, k) {
        const entries = Array.from(embeddingMap.entries());
        const n = entries.length;
        const neighborCount = Math.max(1, Math.min(k, n - 1));
        const results = [];

        for (let i = 0; i < n; i++) {
            const [currentAvatar, currentVec] = entries[i];
            const distances = [];

            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const d = this.calculateEuclideanDistance(currentVec, entries[j][1]);
                distances.push(d);
            }

            distances.sort((a, b) => a - b);
            const kNearest = distances.slice(0, neighborCount);
            const avgDist = kNearest.reduce((acc, val) => acc + val, 0) / neighborCount;
            results.push({ avatar: currentAvatar, distance: avgDist });
        }

        return results;
    }

    static computeLOF(embeddingMap, k) {
        const entries = Array.from(embeddingMap.entries());
        const n = entries.length;
        const neighborCount = Math.max(1, Math.min(k, n - 1));

        const neighborhoodInfo = new Array(n);

        for (let i = 0; i < n; i++) {
            const vecA = entries[i][1];
            const dists = [];
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                dists.push({ idx: j, dist: this.calculateEuclideanDistance(vecA, entries[j][1]) });
            }
            dists.sort((a, b) => a.dist - b.dist);
            
            const neighbors = dists.slice(0, neighborCount);
            const kDistance = neighbors[neighborCount - 1].dist;
            neighborhoodInfo[i] = { neighbors, kDistance };
        }

        const lrd = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const { neighbors } = neighborhoodInfo[i];
            let sumReachDist = 0;
            for (const neighbor of neighbors) {
                const neighborIdx = neighbor.idx;
                const distToNeighbor = neighbor.dist;
                const neighborKDist = neighborhoodInfo[neighborIdx].kDistance;
                const reachDist = Math.max(neighborKDist, distToNeighbor);
                sumReachDist += reachDist;
            }
            const avgReachDist = sumReachDist / neighborCount;
            lrd[i] = avgReachDist > 0 ? (1 / avgReachDist) : 0;
        }

        const results = [];
        for (let i = 0; i < n; i++) {
            const { neighbors } = neighborhoodInfo[i];
            let sumNeighborLrd = 0;
            for (const neighbor of neighbors) {
                sumNeighborLrd += lrd[neighbor.idx];
            }
            const currentLrd = lrd[i];
            let score = 0;
            if (currentLrd > 0) {
                score = (sumNeighborLrd / neighborCount) / currentLrd;
            }
            results.push({ avatar: entries[i][0], distance: score });
        }
        return results;
    }

    static computeIsolationForest(embeddingMap, nTrees = 100) {
        const entries = Array.from(embeddingMap.entries());
        const data = entries.map(e => e[1]);
        const n = data.length;
        const dim = data[0].length;
        const subsampleSize = Math.min(256, n);
        const heightLimit = Math.ceil(Math.log2(subsampleSize));

        const c = (size) => {
            if (size <= 1) return 0;
            return 2 * (Math.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size);
        };
        const avgPathLengthNormalization = c(subsampleSize);
        const pathLengths = new Float32Array(n).fill(0);

        for (let t = 0; t < nTrees; t++) {
            const indices = [];
            for(let i=0; i<n; i++) indices.push(i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
            
            const buildAndEvaluate = (currentIndices, currentDepth) => {
                if (currentDepth >= heightLimit || currentIndices.length <= 1) {
                    return;
                }
                const feature = Math.floor(Math.random() * dim);
                let min = Infinity, max = -Infinity;
                for(const idx of currentIndices) {
                    const val = data[idx][feature];
                    if(val < min) min = val;
                    if(val > max) max = val;
                }
                if (min === max) return;
                const splitValue = Math.random() * (max - min) + min;
                const left = [];
                const right = [];
                for (const idx of currentIndices) {
                    pathLengths[idx]++;
                    if (data[idx][feature] < splitValue) left.push(idx);
                    else right.push(idx);
                }
                buildAndEvaluate(left, currentDepth + 1);
                buildAndEvaluate(right, currentDepth + 1);
            };
            buildAndEvaluate(indices, 0);
        }

        const results = [];
        for (let i = 0; i < n; i++) {
            const avgPathLen = pathLengths[i] / nTrees;
            const score = Math.pow(2, -(avgPathLen) / avgPathLengthNormalization);
            results.push({ avatar: entries[i][0], distance: score });
        }
        return results;
    }

    static computeECOD(embeddingMap) {
        const entries = Array.from(embeddingMap.entries());
        const data = entries.map(e => e[1]);
        const n = data.length;
        if (n === 0) return [];
        const dim = data[0].length;
        
        const scores = new Float32Array(n).fill(0);

        for (let d = 0; d < dim; d++) {
            const column = new Float32Array(n);
            for(let i=0; i<n; i++) column[i] = data[i][d];
            
            const indices = new Int32Array(n);
            for(let i=0; i<n; i++) indices[i] = i;
            indices.sort((a, b) => column[a] - column[b]);
            
            for(let r=0; r<n; r++) {
                const originalIndex = indices[r];
                const pLeft = (r + 1) / (n + 1);
                const pRight = (n - r) / (n + 1);
                const sLeft = -Math.log(pLeft);
                const sRight = -Math.log(pRight);
                scores[originalIndex] += Math.max(sLeft, sRight);
            }
        }
        
        const results = [];
        for (let i = 0; i < n; i++) {
            results.push({ avatar: entries[i][0], distance: scores[i] });
        }
        return results;
    }

    static computeHBOS(embeddingMap) {
        const entries = Array.from(embeddingMap.entries());
        const data = entries.map(e => e[1]);
        const n = data.length;
        if (n === 0) return [];
        const dim = data[0].length;
        const scores = new Float32Array(n).fill(0);

        for (let d = 0; d < dim; d++) {
            let min = Infinity, max = -Infinity;
            const col = new Float32Array(n);
            let sum = 0;

            for(let i=0; i<n; i++) {
                const val = data[i][d];
                col[i] = val;
                sum += val;
                if(val < min) min = val;
                if(val > max) max = val;
            }

            const mean = sum / n;
            let sqDiff = 0;
            for(let x of col) sqDiff += (x - mean) * (x - mean);
            const std = Math.sqrt(sqDiff / n);

            let binCount = 10; 
            if (std > 0) {
                const binWidth = (3.5 * std) / Math.pow(n, 1/3);
                if (binWidth > 0) {
                    binCount = Math.ceil((max - min) / binWidth);
                }
            }
            binCount = Math.max(5, Math.min(binCount, Math.floor(n), 50));
            
            const hist = new Int32Array(binCount).fill(0);
            const range = max - min;
            const step = range / binCount;
            
            if (step > 0) {
                for(let i=0; i<n; i++) {
                    let binIdx = Math.floor((col[i] - min) / step);
                    if (binIdx >= binCount) binIdx = binCount - 1;
                    hist[binIdx]++;
                }
            } else {
                hist[0] = n;
            }

            for(let i=0; i<n; i++) {
                let binIdx = 0;
                if (step > 0) {
                    binIdx = Math.floor((col[i] - min) / step);
                    if (binIdx >= binCount) binIdx = binCount - 1;
                }
                const count = hist[binIdx];
                if (count > 0) {
                    scores[i] += Math.log(n / count);
                } else {
                    scores[i] += Math.log(n); 
                }
            }
        }
        
        const results = [];
        for (let i = 0; i < n; i++) {
            results.push({ avatar: entries[i][0], distance: scores[i] });
        }
        return results;
    }

    static computeLUNAR(embeddingMap, k) {
        const entries = Array.from(embeddingMap.entries());
        const realData = entries.map(e => e[1]);
        const n = realData.length;
        if (n < 2) return [];

        const dim = realData[0].length;
        const neighborCount = Math.max(1, Math.min(k, n - 1));

        const negativeData = [];
        const noiseScale = 0.1;

        for(let i=0; i<n; i++) {
            const noise = new Float32Array(dim);
            for(let d=0; d<dim; d++) {
                const g = (Math.random() + Math.random() + Math.random() - 1.5) * 2; 
                noise[d] = realData[i][d] + (g * noiseScale);
            }
            negativeData.push(noise);
        }

        const getKNN = (queryPoint, isRealIndex) => {
            const dists = [];
            for(let i=0; i<n; i++) {
                if(isRealIndex === i) continue;
                dists.push(this.calculateEuclideanDistance(queryPoint, realData[i]));
            }
            dists.sort((a,b) => a - b);
            return dists.slice(0, neighborCount);
        };

        const X_real = [];
        for(let i=0; i<n; i++) X_real.push(getKNN(realData[i], i));

        const X_neg = [];
        for(let i=0; i<n; i++) X_neg.push(getKNN(negativeData[i], -1));

        const hiddenSize = 16;
        const learningRate = 0.1;
        const epochs = 50;

        const W1 = new Float32Array(neighborCount * hiddenSize).map(() => Math.random() * 0.2 - 0.1);
        const b1 = new Float32Array(hiddenSize).fill(0);
        const W2 = new Float32Array(hiddenSize).map(() => Math.random() * 0.2 - 0.1);
        let b2 = 0;

        const forward = (input) => {
            const h = new Float32Array(hiddenSize);
            for(let i=0; i<hiddenSize; i++) {
                let sum = b1[i];
                for(let j=0; j<neighborCount; j++) {
                    sum += input[j] * W1[j*hiddenSize + i];
                }
                h[i] = sum > 0 ? sum : 0;
            }
            let z = b2;
            for(let i=0; i<hiddenSize; i++) {
                z += h[i] * W2[i];
            }
            const pred = 1 / (1 + Math.exp(-z));
            return { h, pred };
        };

        for(let epoch=0; epoch<epochs; epoch++) {
            const indices = [];
            for(let i=0; i<n*2; i++) indices.push(i);
            indices.sort(() => Math.random() - 0.5);

            for(const idx of indices) {
                const isReal = idx < n;
                const features = isReal ? X_real[idx] : X_neg[idx - n];
                const target = isReal ? 0 : 1;

                const { h, pred } = forward(features);
                const error = pred - target;
                const gradOut = error * (pred * (1 - pred));

                for(let i=0; i<hiddenSize; i++) {
                    const grad = gradOut * h[i];
                    W2[i] -= learningRate * grad;
                }
                b2 -= learningRate * gradOut;

                for(let i=0; i<hiddenSize; i++) {
                    const gradH = gradOut * W2[i];
                    const gradReLU = h[i] > 0 ? 1 : 0;
                    const gradLayer1 = gradH * gradReLU;
                    b1[i] -= learningRate * gradLayer1;
                    for(let j=0; j<neighborCount; j++) {
                        W1[j*hiddenSize + i] -= learningRate * gradLayer1 * features[j];
                    }
                }
            }
        }

        const results = [];
        for(let i=0; i<n; i++) {
            const { pred } = forward(X_real[i]);
            results.push({ avatar: entries[i][0], distance: pred });
        }
        return results;
    }

    static computeLODAEnsemble(embeddingMap) {
        const n = embeddingMap.size;
        if (n < 2) return [];

        const n_sqrt = Math.sqrt(n);
        const k_025 = Math.max(1, Math.round(Math.pow(n, 0.25)));
        const k_050 = Math.max(1, Math.round(n_sqrt));
        const k_075 = Math.max(1, Math.round(Math.pow(n, 0.75)));
        
        const rawResults = [];
        
        rawResults.push(this.normalizeScores(this.computeKNNUniqueness(embeddingMap, k_025)));
        rawResults.push(this.normalizeScores(this.computeKNNUniqueness(embeddingMap, k_050)));
        rawResults.push(this.normalizeScores(this.computeKNNUniqueness(embeddingMap, k_075)));
        
        rawResults.push(this.normalizeScores(this.computeLOF(embeddingMap, 10)));
        rawResults.push(this.normalizeScores(this.computeLOF(embeddingMap, 30)));
        
        rawResults.push(this.normalizeScores(this.computeIsolationForest(embeddingMap, 100)));
        
        rawResults.push(this.normalizeScores(this.computeHBOS(embeddingMap)));
        
        rawResults.push(this.normalizeScores(this.computeECOD(embeddingMap)));
        
        rawResults.push(this.normalizeScores(this.computeLUNAR(embeddingMap, k_050)));

        const aggregated = new Map();
        
        for (const resultSet of rawResults) {
            for (const item of resultSet) {
                const prev = aggregated.get(item.avatar) || 0;
                aggregated.set(item.avatar, prev + item.distance);
            }
        }

        const finalResults = [];
        const count = rawResults.length;
        for (const [avatar, totalScore] of aggregated.entries()) {
            finalResults.push({ avatar, distance: totalScore / count });
        }

        return finalResults;
    }
}

/**
 * UI Manager: Handles DOM generation and updates.
 */
class UIManager {
    constructor(extension) {
        this.ext = extension;
        this.elements = {};
    }

    /**
     * Injects HTML into the page.
     */
    initialize() {
        // Settings Injection
        const settingsTemplate = `
        <div class="charSim-settings-block">
            <label>KoboldCpp URL</label>
            <input id="charSim_url_input" class="text_pole" type="text" placeholder="http://127.0.0.1:5001" />
            <div style="display:flex; justify-content: space-between; align-items: center;">
                <small>Must include http:// and be accessible.</small>
                <div id="charSimBtn_clearCache" class="menu_button" style="width: auto; padding: 2px 8px; font-size: 0.8em;">Clear Cache</div>
            </div>
        </div>`;
        $("#extensions_settings2").append(settingsTemplate);

        // Update settings Input
        const urlInput = $("#charSim_url_input");
        urlInput.val(this.ext.settings.koboldUrl);
        urlInput.on("input", (e) => this.ext.updateSetting('koboldUrl', e.target.value));
        $('#charSimBtn_clearCache').on('click', () => this.ext.clearCache());

        // Main Panel Injection
        const panelTemplate = `
        <div id="characterSimilarityPanel">
            <div class="charSimPanel-content">
                <div class="charSimPanel-header">
                    <div class="fa-solid fa-grip drag-grabber"></div>
                    <h3>Character Library & Analysis</h3>
                    <div id="charSimCloseBtn" class="fa-solid fa-circle-xmark floating_panel_close" title="Close"></div>
                </div>
                <div class="charSim-tabs">
                    <div class="charSim-tab-button active" data-tab="uniqueness">Uniqueness</div>
                    <div class="charSim-tab-button" data-tab="characters">Characters</div>
                </div>
                <div class="charSimPanel-body">
                    <div id="charSimView_uniqueness" class="charSim-view active">
                        <div class="charSim-controls">
                            <select id="charSimSelect_method" class="text_pole charSim-select">
                                <option value="mean">Global Mean Distance</option>
                                <option value="loda">LODA (Ensemble)</option>
                                <option value="lunar">LUNAR (GNN)</option>
                                <option value="hbos">HBOS (Birgé-Rozenblac)</option>
                                <option value="ecod">ECOD (Parameter-free)</option>
                                <option value="isolation">Isolation Forest</option>
                                <option value="lof">Local Outlier Factor</option>
                                <option value="knn">kNN Distance</option>
                            </select>

                            <div id="charSim_param_container" class="charSim-param-group">
                                <label for="charSimInput_n">N:</label>
                                <input id="charSimInput_n" type="number" class="text_pole" min="1" value="20" />
                            </div>
                            
                            <div class="charSim-spacer"></div>
                            <div id="charSimBtn_sort" class="menu_button menu_button_icon fa-solid fa-arrow-down" title="Toggle Sort"></div>
                        </div>
                        <div id="charSimList_uniqueness" class="charSim-list">
                            <div class="charSim-empty-state"><p>Load embeddings to begin.</p></div>
                        </div>
                    </div>

                    <div id="charSimView_characters" class="charSim-view">
                        <div class="charSim-search-bar">
                             <input type="text" id="charSimInput_filter" class="text_pole" placeholder="Filter characters..." />
                        </div>
                        <div id="charSimList_characters" class="charSim-list charSim-grid-container">
                            <!-- Grid Items Injected Here -->
                        </div>
                    </div>
                    
                    <div id="charSimView_details" class="charSim-view">
                        <div class="charSim-details-toolbar">
                             <div class="menu_button" id="charSimBtn_back"><i class="fa-solid fa-arrow-left"></i> Back</div>
                        </div>
                        <div class="charSim-details-content">
                            <!-- Injected content -->
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        
        $('#movingDivs').append(panelTemplate);
        
        // Open Button Injection
        const openBtn = $(`<div id="charSimOpenBtn" class="menu_button fa-solid fa-project-diagram faSmallFontSquareFix" title="Character Analysis"></div>`);
        const btnContainer = $('#rm_buttons_container');
        if (btnContainer.length) btnContainer.append(openBtn);
        else $('#form_character_search_form').before(openBtn);

        // Set initial state from settings
        $('#charSimSelect_method').val(this.ext.settings.uniquenessMethod);
        $('#charSimInput_n').val(this.ext.settings.uniquenessN);
        this.toggleParamInput(this.ext.settings.uniquenessMethod);

        this.bindEvents();
    }

    bindEvents() {
        // Global Open/Close
        $('#charSimOpenBtn').on('click', () => this.ext.onOpenPanel());
        $('#charSimCloseBtn').on('click', () => $('#characterSimilarityPanel').removeClass('open'));

        // Tabs
        $('.charSim-tab-button').on('click', (e) => {
            const tab = $(e.currentTarget).data('tab');
            $('.charSim-tab-button').removeClass('active');
            $(e.currentTarget).addClass('active');
            $('.charSim-view').removeClass('active');
            $(`#charSimView_${tab}`).addClass('active');
        });

        // Sort
        $('#charSimBtn_sort').on('click', (e) => {
            $(e.currentTarget).toggleClass('fa-arrow-down fa-arrow-up');
            this.ext.toggleSort();
        });

        // Dropdown & Params (Trigger Recalc)
        $('#charSimSelect_method').on('change', (e) => {
            const val = e.target.value;
            this.toggleParamInput(val);
            this.ext.updateSetting('uniquenessMethod', val);
            this.ext.runUniqueness();
        });

        $('#charSimInput_n').on('change', (e) => {
            this.ext.updateSetting('uniquenessN', parseInt(e.target.value) || 20);
            this.ext.runUniqueness();
        });

        // Filter Characters
        $('#charSimInput_filter').on('input', (e) => {
            const term = e.target.value.toLowerCase();
            $('.charSim-grid-card').each(function() {
                const name = $(this).find('.charSim-card-name').text().toLowerCase();
                $(this).toggle(name.includes(term));
            });
        });

        // Details View Navigation
        $('#charSimList_characters').on('click', '.charSim-grid-card', (e) => {
             const avatar = $(e.currentTarget).attr('title'); // Using title as lookup or better attach data attribute
             // Ideally re-find from character array by name or index
             // Let's refactor renderCharacterGrid to include data-avatar
             const actualAvatar = $(e.currentTarget).data('avatar');
             this.showCharacterDetails(actualAvatar);
        });

        $('#charSimBtn_back').on('click', () => {
             $('#charSimView_details').removeClass('active');
             $('#charSimView_characters').addClass('active');
        });

        // Details Toggle
        $('#characterSimilarityPanel').on('click', '.charSim-details-toggle', function() {
             const content = $('.charSim-details-advanced');
             const icon = $(this).find('i');
             
             if (content.is(':visible')) {
                 content.slideUp();
                 icon.removeClass('fa-eye').addClass('fa-eye-slash');
             } else {
                 content.slideDown();
                 icon.removeClass('fa-eye-slash').addClass('fa-eye');
             }
        });
    }

    toggleParamInput(method) {
        const show = ['isolation', 'lof', 'knn', 'lunar'].includes(method);
        $('#charSim_param_container').css('display', show ? 'flex' : 'none');
        
        let label = "N:";
        if (method === 'isolation') label = "Trees:";
        if (method === 'lof' || method === 'knn' || method === 'lunar') label = "k:";
        $('label[for="charSimInput_n"]').text(label);
    }

    renderUniquenessList(items, isDescending) {
        const container = $('#charSimList_uniqueness');
        if (!items || items.length === 0) {
            container.html(this.createEmptyState("Processing..."));
            return;
        }

        const sorted = [...items].sort((a, b) => isDescending ? b.distance - a.distance : a.distance - b.distance);

        const html = sorted.map(item => `
            <div class="charSim-item" data-avatar="${item.avatar}">
                <img src="${getThumbnailUrl('avatar', item.avatar)}" />
                <span class="charSim-item-name">${item.name}</span>
                <span class="charSim-item-score" title="Score">${item.distance.toFixed(4)}</span>
            </div>
        `).join('');
        container.html(html);
    }

    renderCharacterGrid(charList) {
        const container = $('#charSimList_characters');
        if (!charList || charList.length === 0) {
            container.html(this.createEmptyState("No characters found."));
            return;
        }

        const html = charList.map(c => `
            <div class="charSim-grid-card" title="${c.name}" data-avatar="${c.avatar}">
                <div class="charSim-card-img-wrapper">
                    <img src="${getThumbnailUrl('avatar', c.avatar)}" loading="lazy" />
                </div>
                <div class="charSim-card-name">${c.name}</div>
            </div>
        `).join('');
        container.html(html);
    }

    showCharacterDetails(avatar) {
        const char = characters.find(c => c.avatar === avatar);
        if (!char) return;

        const container = $('.charSim-details-content');
        
        // Safe access to fields
        const creator = char.creator || char.create_date || "Unknown";
        const desc = char.description || "";
        const personality = char.personality || "";
        const scenario = char.scenario || "";
        const firstMes = char.first_mes || "";
        const mesEx = char.mes_example || "";

        const html = `
            <div class="charSim-details-header">
                <img src="${getThumbnailUrl('avatar', avatar)}" class="charSim-details-img">
                <div class="charSim-details-info">
                    <h1>${char.name}</h1>
                    <div class="charSim-creator-notes">Creator: ${creator}</div>
                </div>
            </div>
            
            <div class="charSim-details-toggle">
                <i class="fa-solid fa-eye-slash"></i> Show Character Data
            </div>
            
            <div class="charSim-details-advanced" style="display:none;">
                <div class="charSim-field">
                    <label>Description</label>
                    <div class="charSim-field-text">${desc}</div>
                </div>
                <div class="charSim-field">
                    <label>Personality</label>
                    <div class="charSim-field-text">${personality}</div>
                </div>
                <div class="charSim-field">
                    <label>Scenario</label>
                    <div class="charSim-field-text">${scenario}</div>
                </div>
                <div class="charSim-field">
                    <label>First Message</label>
                    <div class="charSim-field-text">${firstMes}</div>
                </div>
                <div class="charSim-field">
                    <label>Message Examples</label>
                    <div class="charSim-field-text">${mesEx}</div>
                </div>
            </div>
        `;

        container.html(html);
        
        // Switch Views
        $('.charSim-view').removeClass('active');
        $('#charSimView_details').addClass('active');
    }

    createEmptyState(msg) {
        return `<div class="charSim-empty-state"><p>${msg}</p></div>`;
    }
}

/**
 * Main Extension Controller
 */
class CharacterSimilarityExtension {
    constructor() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, extension_settings[EXTENSION_NAME] || {});
        extension_settings[EXTENSION_NAME] = this.settings;

        this.dataItems = []; 
        this.validCaches = []; 
        this.uniquenessData = [];
        this.isProcessing = false;
        
        this.service = new SimilarityService(this.settings);
        this.ui = new UIManager(this);
    }

    async init() {
        this.ui.initialize();
    }

    updateSetting(key, value) {
        this.settings[key] = value;
        saveSettingsDebounced();
    }

    populateLists() {
        if (this.uniquenessData.length === 0) {
            const simpleList = characters.map(c => ({ 
                avatar: c.avatar, 
                name: c.name, 
                distance: 0 
            }));
            this.ui.renderUniquenessList(simpleList, true);
        } else {
            this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
        }
        this.ui.renderCharacterGrid(characters);
    }

    async onOpenPanel() {
        $('#characterSimilarityPanel').addClass('open');
        this.populateLists();
        
        if (!this.isProcessing) {
            await this.refreshAnalysis();
        }
    }

    async refreshAnalysis() {
        this.isProcessing = true;
        try {
            // 1. Prepare Data & Hashes
            const texts = [];
            for(const char of characters) {
                const text = FIELDS_TO_EMBED.map(f => char[f] || '').join('\n').trim();
                if(text) {
                    const hash = await hashText(text);
                    texts.push({ avatar: char.avatar, text, hash });
                }
            }
            this.dataItems = texts;

            if (this.dataItems.length === 0) throw new Error("No valid characters found.");

            // 2. Update Live Cache (Fetch missing)
            await this.service.updateLiveCache(this.dataItems);

            // 3. Collect ALL valid caches (Multi-Model)
            this.validCaches = this.service.getValidCaches(this.dataItems);

            if(this.validCaches.length === 0) {
                throw new Error("No valid embedding caches available.");
            }
            
            // 4. Run Calculation
            this.runUniqueness();

        } catch (err) {
            toastr.error(err.message, "Analysis Error");
            console.error(err);
        } finally {
            this.isProcessing = false;
        }
    }

    clearCache() {
        this.service.clearCache();
    }

    runUniqueness() {
        if (this.validCaches.length === 0) return;
        
        setTimeout(() => {
            try {
                const method = this.settings.uniquenessMethod;
                const n = this.settings.uniquenessN || 20;
                
                const compositeScores = new Map(); 
                
                // Run algorithm on EACH model cache independently
                for(const cacheObj of this.validCaches) {
                    const embeddingMap = cacheObj.map;
                    let rawResults = [];

                    switch (method) {
                        case 'loda':
                            rawResults = ComputeEngine.computeLODAEnsemble(embeddingMap);
                            break;
                        case 'lunar':
                            rawResults = ComputeEngine.computeLUNAR(embeddingMap, n);
                            break;
                        case 'hbos':
                            rawResults = ComputeEngine.computeHBOS(embeddingMap);
                            break;
                        case 'ecod':
                            rawResults = ComputeEngine.computeECOD(embeddingMap);
                            break;
                        case 'isolation':
                            rawResults = ComputeEngine.computeIsolationForest(embeddingMap, n);
                            break;
                        case 'lof':
                            rawResults = ComputeEngine.computeLOF(embeddingMap, n);
                            break;
                        case 'knn':
                            rawResults = ComputeEngine.computeKNNUniqueness(embeddingMap, n);
                            break;
                        case 'mean':
                        default:
                            rawResults = ComputeEngine.computeGlobalMeanUniqueness(embeddingMap);
                            break;
                    }

                    const normalized = ComputeEngine.normalizeScores(rawResults);
                    
                    for(const item of normalized) {
                        const prev = compositeScores.get(item.avatar) || 0;
                        compositeScores.set(item.avatar, prev + item.distance);
                    }
                }

                const finalResults = [];
                const modelCount = this.validCaches.length;
                
                for(const [avatar, total] of compositeScores.entries()) {
                    const char = characters.find(c => c.avatar === avatar);
                    if(char) {
                        finalResults.push({
                            avatar,
                            name: char.name,
                            distance: total / modelCount
                        });
                    }
                }

                this.uniquenessData = finalResults;
                this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
                
                toastr.success(`Calculated using ${modelCount} model(s).`);

            } catch (e) {
                toastr.error("Error calculating uniqueness: " + e.message);
                console.error(e);
            }
        }, 10);
    }

    toggleSort() {
        this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
    }
}

// --- Init ---
jQuery(() => {
    const extension = new CharacterSimilarityExtension();
    eventSource.on(event_types.APP_READY, () => extension.init());
});