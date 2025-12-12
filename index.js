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
const DEFAULT_SETTINGS = {
    koboldUrl: 'http://127.0.0.1:5001',
    uniquenessMethod: 'mean',
    uniquenessN: 20,
};

// Fields to use for embedding generation
const FIELDS_TO_EMBED = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example'];

/**
 * Service Layer: Handles communication with embedding providers.
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
     * Fetches embeddings for a list of strings.
     * @param {Array<{avatar:string, text:string}>} items 
     * @returns {Promise<Map<string, number[]>>} Map of avatar -> embedding vector
     */
    async fetchEmbeddings(items) {
        const url = this.koboldUrl;
        if (!url) throw new Error("KoboldCpp URL is not configured.");

        const response = await fetch('/api/backends/kobold/embed', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                items: items.map(i => i.text),
                server: url,
            }),
        });

        if (!response.ok) {
            throw new Error(`Server returned status ${response.status}. Check console.`);
        }

        const data = await response.json();
        if (!data.embeddings || data.embeddings.length !== items.length) {
            throw new Error("Invalid response format from embedding server.");
        }

        const resultMap = new Map();
        data.embeddings.forEach((vec, idx) => {
            if (vec && vec.length > 0) {
                resultMap.set(items[idx].avatar, vec);
            }
        });

        return resultMap;
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

    // Standard Euclidean Distance (L2)
    static calculateEuclideanDistance(vecA, vecB) {
        let sum = 0;
        for (let i = 0; i < vecA.length; i++) {
            const diff = vecA[i] - vecB[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // --- Algorithms ---

    /**
     * Algorithm 1: Global Mean Distance
     * Simple distance from the centroid of the entire library.
     */
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

    /**
     * Algorithm 2: k-Nearest Neighbors (kNN) Outlier Score
     * Score is the average distance to the k nearest neighbors.
     * Higher score = more isolated (unique).
     */
    static computeKNNUniqueness(embeddingMap, k) {
        const entries = Array.from(embeddingMap.entries()); // [[avatar, vec], ...]
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

    /**
     * Algorithm 3: Local Outlier Factor (LOF)
     * Compares local density of a point to local density of its neighbors.
     * Score > 1 implies outlier.
     */
    static computeLOF(embeddingMap, k) {
        const entries = Array.from(embeddingMap.entries());
        const n = entries.length;
        const neighborCount = Math.max(1, Math.min(k, n - 1));

        // 1. Find neighbors and k-distance
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

        // 2. Reachability Distance and LRD
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

        // 3. LOF Score
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

    /**
     * Algorithm 4: Isolation Forest
     * Randomly partitions data. Outliers are isolated in fewer steps.
     */
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
            // Subsample
            const indices = [];
            for(let i=0; i<n; i++) indices.push(i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }
            
            // Build & Evaluate Tree recursively
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

    /**
     * Algorithm 5: ECOD (Empirical Cumulative Distribution Functions for Outlier Detection)
     * Parameter-free. Estimates tail probabilities for each dimension.
     */
    static computeECOD(embeddingMap) {
        const entries = Array.from(embeddingMap.entries()); // [[avatar, vec], ...]
        const data = entries.map(e => e[1]);
        const n = data.length;
        if (n === 0) return [];
        const dim = data[0].length;
        
        // Initialize scores
        const scores = new Float32Array(n).fill(0);

        // Process per dimension
        for (let d = 0; d < dim; d++) {
            // Extract column
            const column = new Float32Array(n);
            for(let i=0; i<n; i++) column[i] = data[i][d];
            
            // Create rank indices
            const indices = new Int32Array(n);
            for(let i=0; i<n; i++) indices[i] = i;
            
            indices.sort((a, b) => column[a] - column[b]);
            
            // Calculate ECDF probabilities
            // Left tail: P(X <= x)
            // Right tail: P(X >= x)
            // Using (r + 1) / (n + 1) for smoothing
            
            for(let r=0; r<n; r++) {
                const originalIndex = indices[r];
                
                const pLeft = (r + 1) / (n + 1);
                const pRight = (n - r) / (n + 1);
                
                // Score is the log of the inverse probability (skewness independent)
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
            <small>Must include http:// and be accessible.</small>
        </div>`;
        $("#extensions_settings2").append(settingsTemplate);

        // Update settings Input
        const urlInput = $("#charSim_url_input");
        urlInput.val(this.ext.settings.koboldUrl);
        urlInput.on("input", (e) => this.ext.updateSetting('koboldUrl', e.target.value));

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
                            <div id="charSimBtn_load" class="menu_button">Load Embeddings</div>
                            
                            <select id="charSimSelect_method" class="text_pole charSim-select">
                                <option value="mean">Global Mean Distance</option>
                                <option value="ecod">ECOD (Parameter-free)</option>
                                <option value="isolation">Isolation Forest</option>
                                <option value="lof">Local Outlier Factor</option>
                                <option value="knn">kNN Distance</option>
                            </select>

                            <div id="charSim_param_container" class="charSim-param-group">
                                <label for="charSimInput_n">N:</label>
                                <input id="charSimInput_n" type="number" class="text_pole" min="1" value="20" />
                            </div>

                            <div id="charSimBtn_calcUnique" class="menu_button">Calculate Uniqueness</div>
                            
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
        $('#charSimOpenBtn').on('click', () => {
            $('#characterSimilarityPanel').addClass('open');
            this.ext.populateLists();
        });
        $('#charSimCloseBtn').on('click', () => $('#characterSimilarityPanel').removeClass('open'));

        // Tabs
        $('.charSim-tab-button').on('click', (e) => {
            const tab = $(e.currentTarget).data('tab');
            $('.charSim-tab-button').removeClass('active');
            $(e.currentTarget).addClass('active');
            $('.charSim-view').removeClass('active');
            $(`#charSimView_${tab}`).addClass('active');
        });

        // Actions
        $('#charSimBtn_load').on('click', () => this.ext.loadEmbeddings());
        $('#charSimBtn_calcUnique').on('click', () => this.ext.runUniqueness());
        $('#charSimBtn_sort').on('click', (e) => {
            $(e.currentTarget).toggleClass('fa-arrow-down fa-arrow-up');
            this.ext.toggleSort();
        });

        // Dropdown & Params
        $('#charSimSelect_method').on('change', (e) => {
            const val = e.target.value;
            this.toggleParamInput(val);
            this.ext.updateSetting('uniquenessMethod', val);
        });

        $('#charSimInput_n').on('change', (e) => {
            this.ext.updateSetting('uniquenessN', parseInt(e.target.value) || 20);
        });

        // Filter Characters
        $('#charSimInput_filter').on('input', (e) => {
            const term = e.target.value.toLowerCase();
            $('.charSim-grid-card').each(function() {
                const name = $(this).find('.charSim-card-name').text().toLowerCase();
                $(this).toggle(name.includes(term));
            });
        });
    }

    toggleParamInput(method) {
        const show = ['isolation', 'lof', 'knn'].includes(method);
        $('#charSim_param_container').css('display', show ? 'flex' : 'none');
        
        // Update Label contextually
        let label = "N:";
        if (method === 'isolation') label = "Trees:";
        if (method === 'lof' || method === 'knn') label = "k:";
        $('label[for="charSimInput_n"]').text(label);
    }

    renderUniquenessList(items, isDescending) {
        const container = $('#charSimList_uniqueness');
        if (!items || items.length === 0) {
            container.html(this.createEmptyState("No data available. Load embeddings first."));
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
            <div class="charSim-grid-card" title="${c.name}">
                <div class="charSim-card-img-wrapper">
                    <img src="${getThumbnailUrl('avatar', c.avatar)}" loading="lazy" />
                </div>
                <div class="charSim-card-name">${c.name}</div>
            </div>
        `).join('');
        container.html(html);
    }

    createEmptyState(msg) {
        return `<div class="charSim-empty-state"><p>${msg}</p></div>`;
    }

    setLoading(isLoading, msg = "Processing...") {
        const buttons = $('#charSimBtn_load, #charSimBtn_calcUnique');
        buttons.prop('disabled', isLoading);
        if(isLoading) toastr.info(msg);
    }
}

/**
 * Main Extension Controller
 */
class CharacterSimilarityExtension {
    constructor() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, extension_settings[EXTENSION_NAME] || {});
        extension_settings[EXTENSION_NAME] = this.settings;

        this.embeddings = new Map();
        this.uniquenessData = [];
        
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
        // Populate Uniqueness List (Uses cached data or simple list if no scores)
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

        // Populate Character Grid
        this.ui.renderCharacterGrid(characters);
    }

    async loadEmbeddings() {
        this.ui.setLoading(true, "Preparing text data...");
        try {
            const texts = characters
                .map(char => {
                    const text = FIELDS_TO_EMBED.map(f => char[f] || '').join('\n').trim();
                    return text ? { avatar: char.avatar, text } : null;
                })
                .filter(item => item !== null);

            if (texts.length === 0) throw new Error("No characters found with text content.");

            toastr.info(`Sending ${texts.length} items to embedding server...`);
            this.embeddings = await this.service.fetchEmbeddings(texts);
            
            toastr.success(`Loaded ${this.embeddings.size} embeddings.`);
        } catch (err) {
            toastr.error(err.message, "Embedding Error");
            console.error(err);
        } finally {
            this.ui.setLoading(false);
        }
    }

    runUniqueness() {
        if (this.embeddings.size === 0) return toastr.warning("Please load embeddings first.");
        
        this.ui.setLoading(true, "Calculating...");
        
        // Small timeout to allow UI to show loading toast
        setTimeout(() => {
            try {
                const method = this.settings.uniquenessMethod;
                const n = this.settings.uniquenessN || 20;
                let results = [];

                switch (method) {
                    case 'ecod':
                        results = ComputeEngine.computeECOD(this.embeddings);
                        break;
                    case 'isolation':
                        results = ComputeEngine.computeIsolationForest(this.embeddings, n);
                        break;
                    case 'lof':
                        results = ComputeEngine.computeLOF(this.embeddings, n);
                        break;
                    case 'knn':
                        results = ComputeEngine.computeKNNUniqueness(this.embeddings, n);
                        break;
                    case 'mean':
                    default:
                        results = ComputeEngine.computeGlobalMeanUniqueness(this.embeddings);
                        break;
                }
                
                // Map names back
                this.uniquenessData = results.map(r => {
                    const char = characters.find(c => c.avatar === r.avatar);
                    return char ? { ...r, name: char.name } : null;
                }).filter(r => r);

                this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
                toastr.success("Uniqueness calculated using " + method.toUpperCase());
            } catch (e) {
                toastr.error("Error calculating uniqueness: " + e.message);
                console.error(e);
            } finally {
                this.ui.setLoading(false);
            }
        }, 50);
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