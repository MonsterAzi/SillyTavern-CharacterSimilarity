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
    clusterThreshold: 0.95,
};

// Fields to use for embedding generation
const FIELDS_TO_EMBED = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example'];

// --- Helper: Minified Graph Clustering Library ---
// Storing this as a constant string to inject into the Worker. 
// This keeps the worker logic clean.
const CLUSTER_LIB_SOURCE = `
const e={52:e=>{function n(e,n){var t=e.nodes.slice(),r=[];if(!t.length)return r;for(var o=[],i=null;t.length;)for(o.length||(i&&r.push(i),i={nodes:(o=[t.pop()]).slice(),edges:{}});o.length;)for(var d=o.pop(),a=t.length-1;a>=0;a--){var s=t[a];if(e.edges[d.id]&&e.edges[d.id][s.id]>=n){o.push(s),i.nodes.push(s),i.edges[d.id]=i.edges[d.id]||{},i.edges[d.id][s.id]=e.edges[d.id][s.id],i.edges[s.id]=i.edges[s.id]||{},i.edges[s.id][d.id]=e.edges[s.id][d.id];var c=t.slice(0,a).concat(t.slice(a+1));t=c}}return i&&r.push(i),r}e.exports={create:function(e,n){var t={},r=1,o=e.map((function(e){return{id:r++,data:e}}));return o.forEach((function(e){t[e.id]=t[e.id]||{},o.forEach((function(r){if(e!==r){var o=n(e.data,r.data);t[e.id][r.id]=o,t[r.id]=t[r.id]||{},t[r.id][e.id]=o}}))})),{nodes:o,edges:t}},data:function(e){return e.data},connected:n,divide:function(e,t,r){for(var o=1/0,i=0,d=function(e){var n=0;return e.nodes.forEach((function(t){e.nodes.forEach((function(r){var o=e.edges[t.id][r.id];o&&o>n&&(n=o)}))})),n}(e)+1,a=null,s=-2;s<r;s++){var c,u=n(e,c=-2==s?i:-1==s?d:(d+i)/2),f=u.length-t;if(f<o&&f>=0&&(o=f,a=u),u.length>t&&(d=c),u.length<t&&(i=c),u.length==t)break;if(i==d)break}return a},findCenter:function(e){var n=function(e){var n={};return e.nodes.forEach((function(e){n[e.id]={},n[e.id][e.id]=0})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){if(t!=r){var o=e.edges[t.id]&&e.edges[t.id][r.id];null==o&&(o=1/0),n[t.id][r.id]=o}}))})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){e.nodes.forEach((function(e){var o=n[r.id][t.id]+n[t.id][e.id];n[r.id][e.id]>o&&(n[r.id][e.id]=o)}))}))})),n}(e),t=1/0,r=null;return e.nodes.forEach((function(o){var i=0;e.nodes.forEach((function(e){var t=n[o.id][e.id];t>i&&(i=t)})),t>i&&(t=i,r=o)})),r},growFromNuclei:function(e,n){for(var t=n.map((function(e){return{nodes:[e],edges:{}}})),r=e.nodes.filter((function(e){return 0==n.filter((function(n){return e==n})).length})),o=0,i=t.length;r.length&&i;){i-=1;var d=t[o];o=(o+1)%t.length;var a=null,s=null,c=-1/0;if(d.nodes.forEach((function(n){r.forEach((function(t){var r=e.edges[n.id]&&e.edges[n.id][t.id];r&&r>c&&(a=n,s=t,c=r)}))})),a){var u=a,f=s;d.edges[u.id]=d.edges[u.id]||{},d.edges[u.id][f.id]=e.edges[u.id][f.id],d.edges[f.id]=d.edges[f.id]||{},d.edges[f.id][u.id]=e.edges[f.id][u.id],d.nodes.push(s),r=r.filter((function(e){return e!=s})),i=t.length}}return{graphs:t,orphans:r}}}},834:(e,n,t)=>{var r=t(52);e.exports=function(e,n){var t,o=r.create(e,(function(e,t){var r=n(e,t);if("number"!=typeof r||r<0)throw new Error("Similarity function did not yield a number in the range [0, +Inf) when comparing "+e+" to "+t+" : "+r);return r}));function i(e){return function(){return e.apply(this,Array.prototype.slice.call(arguments)).map((function(e){return e.nodes.map(r.data)}))}}function d(e,n){var t=n||1e3;return r.divide(o,e,t)}function a(e,n){var t=d(e,n);return t.sort((function(e,n){return n.nodes.length-e.nodes.length})),t.splice(e),t.map(r.findCenter)}return{groups:i(d),representatives:(t=a,function(){return t.apply(this,Array.prototype.slice.call(arguments)).map(r.data)}),similarGroups:i((function(e){return r.connected(o,e)})),evenGroups:function(e,n){for(var t=a(e),i=r.growFromNuclei(o,t),d=i.graphs.map((function(e){return e.nodes.map(r.data)}));i.orphans.length;){var s=r.data(i.orphans.pop());d.sort((function(e,n){return e.length-n.length})),d[0].push(s)}return d}}}}},n={};function t(r){var o=n[r];if(void 0!==o)return o.exports;var i=n[r]={exports:{}};return e[r](i,i.exports,t),i.exports}var cluster = t(834);
`;

/**
 * Service Layer: Handles communication with embedding providers.
 * Modularized so you can add OpenAI or Transformers.js later easily.
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
 * Compute Engine: Handles Math and Worker Tasks
 */
class ComputeEngine {
    
    // Standard Math Utils
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
            sum += Math.abs(vecA[i] - vecB[i]); // Manhattan distance used in original code, kept for consistency
        }
        return sum;
    }

    /**
     * Calculates uniqueness scores relative to library average.
     */
    static computeUniqueness(embeddingMap) {
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
     * Spawns a worker to perform clustering.
     */
    static async computeClusters(embeddingMap, threshold) {
        return new Promise((resolve, reject) => {
            // Worker Logic Definition
            const workerScript = `
                ${CLUSTER_LIB_SOURCE}
                
                function cosineSimilarity(vecA, vecB) { 
                    let dotProduct = 0.0, normA = 0.0, normB = 0.0; 
                    for (let i = 0; i < vecA.length; i++) { 
                        dotProduct += vecA[i] * vecB[i]; 
                        normA += vecA[i] * vecA[i]; 
                        normB += vecB[i] * vecB[i]; 
                    } 
                    if (normA === 0 || normB === 0) return 0; 
                    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)); 
                }

                self.onmessage = function(event) {
                    try {
                        const { embeddings, threshold } = event.data;
                        const data = Array.from(embeddings.entries()).map(([avatar, embedding]) => ({ avatar, embedding }));
                        
                        const clustering = cluster(data, (a, b) => {
                            const similarity = cosineSimilarity(a.embedding, b.embedding);
                            // Normalize [-1, 1] to [0, 1]
                            return (similarity + 1) / 2;
                        });
                        
                        const groups = clustering.similarGroups(threshold);
                        self.postMessage({ success: true, groups });
                    } catch (e) {
                        self.postMessage({ success: false, error: e.message });
                    }
                };
            `;

            const blob = new Blob([workerScript], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));

            worker.onmessage = (e) => {
                worker.terminate();
                if (e.data.success) resolve(e.data.groups);
                else reject(new Error(e.data.error));
            };

            worker.onerror = (e) => {
                worker.terminate();
                reject(e);
            };

            worker.postMessage({ embeddings: embeddingMap, threshold });
        });
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
                    <h3>Character Similarity</h3>
                    <div id="charSimCloseBtn" class="fa-solid fa-circle-xmark floating_panel_close" title="Close"></div>
                </div>
                <div class="charSim-tabs">
                    <div class="charSim-tab-button active" data-tab="uniqueness">Uniqueness</div>
                    <div class="charSim-tab-button" data-tab="clustering">Clustering</div>
                </div>
                <div class="charSimPanel-body">
                    <div id="charSimView_uniqueness" class="charSim-view active">
                        <div class="charSim-controls">
                            <div id="charSimBtn_load" class="menu_button">Load Embeddings</div>
                            <div id="charSimBtn_calcUnique" class="menu_button">Calculate Uniqueness</div>
                            <div class="charSim-spacer"></div>
                            <div id="charSimBtn_sort" class="menu_button menu_button_icon fa-solid fa-arrow-down" title="Toggle Sort"></div>
                        </div>
                        <div id="charSimList_uniqueness" class="charSim-list">
                            <div class="charSim-empty-state"><p>Load embeddings to begin.</p></div>
                        </div>
                    </div>

                    <div id="charSimView_clustering" class="charSim-view">
                        <div class="charSim-controls">
                            <div id="charSimBtn_calcCluster" class="menu_button">Calculate Clusters</div>
                            <div class="charSim-spacer"></div>
                            <label>Threshold: <b id="charSimLabel_threshold">0.95</b></label>
                            <input type="range" id="charSimInput_threshold" min="0.5" max="1.0" step="0.01" value="0.95">
                        </div>
                        <div id="charSimList_clustering" class="charSim-list">
                            <div class="charSim-empty-state"><p>Load embeddings to begin.</p></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        
        $('#movingDivs').append(panelTemplate);
        
        // Open Button Injection
        const openBtn = $(`<div id="charSimOpenBtn" class="menu_button fa-solid fa-project-diagram faSmallFontSquareFix" title="Character Similarity"></div>`);
        const btnContainer = $('#rm_buttons_container');
        if (btnContainer.length) btnContainer.append(openBtn);
        else $('#form_character_search_form').before(openBtn);

        this.bindEvents();
    }

    bindEvents() {
        // Global Open/Close
        $('#charSimOpenBtn').on('click', () => {
            $('#characterSimilarityPanel').addClass('open');
            this.ext.populateInitialList();
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
        $('#charSimBtn_calcCluster').on('click', () => this.ext.runClustering());
        $('#charSimBtn_sort').on('click', (e) => {
            $(e.currentTarget).toggleClass('fa-arrow-down fa-arrow-up');
            this.ext.toggleSort();
        });

        // Inputs
        $('#charSimInput_threshold').on('input', (e) => {
            const val = parseFloat(e.target.value);
            $('#charSimLabel_threshold').text(val.toFixed(2));
            this.ext.updateSetting('clusterThreshold', val);
        });
    }

    renderUniquenessList(items, isDescending) {
        const container = $('#charSimList_uniqueness');
        if (!items || items.length === 0) {
            container.html(this.createEmptyState("No data available."));
            return;
        }

        const sorted = [...items].sort((a, b) => isDescending ? b.distance - a.distance : a.distance - b.distance);

        const html = sorted.map(item => `
            <div class="charSim-item" data-avatar="${item.avatar}">
                <img src="${getThumbnailUrl('avatar', item.avatar)}" />
                <span class="charSim-item-name">${item.name}</span>
                <span class="charSim-item-score" title="Distance from average">${item.distance.toFixed(4)}</span>
            </div>
        `).join('');
        container.html(html);
        this.addClickListeners();
    }

    renderClusterList(groups) {
        const container = $('#charSimList_clustering');
        const meaningfulGroups = groups.filter(g => g.members.length > 1);

        if (meaningfulGroups.length === 0) {
            container.html(this.createEmptyState("No similar groups found at this threshold."));
            return;
        }

        const html = meaningfulGroups.map(group => `
            <div class="charSim-cluster">
                <div class="charSim-cluster-header">
                    <span>Cluster Score (Uniqueness)</span>
                    <span>${group.clusterUniqueness.toFixed(4)}</span>
                </div>
                ${group.members.map(m => `
                    <div class="charSim-item" data-avatar="${m.avatar}">
                        <img src="${getThumbnailUrl('avatar', m.avatar)}" />
                        <span class="charSim-item-name">${m.name}</span>
                        <span class="charSim-item-score" title="Distance from cluster center">${m.intraClusterDistance.toFixed(4)}</span>
                    </div>
                `).join('')}
            </div>
        `).join('');
        container.html(html);
        this.addClickListeners();
    }

    createEmptyState(msg) {
        return `<div class="charSim-empty-state"><p>${msg}</p></div>`;
    }

    addClickListeners() {
        // Optional: Add click to open character or select them
        // Currently just visual
    }

    setLoading(isLoading, msg = "Processing...") {
        const buttons = $('#charSimBtn_load, #charSimBtn_calcUnique, #charSimBtn_calcCluster');
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
        this.clusterData = [];
        
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

    populateInitialList() {
        // Just show characters without scores if no data exists
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
        
        try {
            const results = ComputeEngine.computeUniqueness(this.embeddings);
            
            // Map names back
            this.uniquenessData = results.map(r => {
                const char = characters.find(c => c.avatar === r.avatar);
                return char ? { ...r, name: char.name } : null;
            }).filter(r => r);

            this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
            toastr.success("Uniqueness calculated.");
        } catch (e) {
            toastr.error("Error calculating uniqueness: " + e.message);
        }
    }

    toggleSort() {
        this.ui.renderUniquenessList(this.uniquenessData, $('#charSimBtn_sort').hasClass('fa-arrow-down'));
    }

    async runClustering() {
        if (this.embeddings.size === 0) return toastr.warning("Please load embeddings first.");
        
        this.ui.setLoading(true, "Clustering...");
        try {
            const groups = await ComputeEngine.computeClusters(this.embeddings, this.settings.clusterThreshold);
            
            // Process raw groups into display data
            const libMean = ComputeEngine.calculateMeanVector(Array.from(this.embeddings.values()));
            
            this.clusterData = groups.map(group => {
                // Get embeddings for this group
                const groupEmbeddings = group.map(m => this.embeddings.get(m.avatar));
                const groupMean = ComputeEngine.calculateMeanVector(groupEmbeddings);
                
                // Calculate how unique this group is compared to the library
                const clusterUniqueness = ComputeEngine.calculateEuclideanDistance(groupMean, libMean);

                const members = group.map(m => {
                    const char = characters.find(c => c.avatar === m.avatar);
                    const vec = this.embeddings.get(m.avatar);
                    const intraDist = ComputeEngine.calculateEuclideanDistance(vec, groupMean);
                    return char ? { avatar: m.avatar, name: char.name, intraClusterDistance: intraDist } : null;
                }).filter(m => m);
                
                // Sort members by how close they are to the group center
                members.sort((a,b) => a.intraClusterDistance - b.intraClusterDistance);

                return { clusterUniqueness, members };
            }).sort((a,b) => b.members.length - a.members.length); // Largest groups first

            this.ui.renderClusterList(this.clusterData);
            toastr.success(`Found ${this.clusterData.filter(g => g.members.length > 1).length} groups.`);
        } catch (e) {
            toastr.error(e.message, "Clustering Error");
        } finally {
            this.ui.setLoading(false);
        }
    }
}

// --- Init ---
jQuery(() => {
    const extension = new CharacterSimilarityExtension();
    eventSource.on(event_types.APP_READY, () => extension.init());
});