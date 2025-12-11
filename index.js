import { extension_settings } from "../../../extensions.js";
import { characters, getThumbnailUrl, saveSettingsDebounced, eventSource, event_types, getRequestHeaders, saveFile, loadFile } from "../../../../script.js";

const extensionName = "character_similarity";

const defaultSettings = {
    koboldUrl: 'http://127.0.0.1:5001',
    clusterThreshold: 0.95,
    algorithm: 'knn', // 'mean', 'knn', 'harmonic'
    algoParams: {
        k: 5,
    }
};

// State
const characterEmbeddings = new Map(); // Map<avatar, float[]>
let uniquenessResults = [];
let clusterResults = [];
let calculationWorker = null;

const fieldsToEmbed = [
    'name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example',
];

// --- HTML GENERATION ---

function populateCharacterList() {
    // Initial populate before calculation
    const sortedCharacters = characters.slice().sort((a, b) => a.name.localeCompare(b.name));
    const characterListHtml = sortedCharacters.map(char => `
        <div class="charSim-character-item" data-avatar="${char.avatar}">
            <img src="${getThumbnailUrl('avatar', char.avatar)}" alt="${char.name}">
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
    
    // Sort based on score
    sortedList.sort((a, b) => isDescending ? b.score - a.score : a.score - b.score);

    const characterListHtml = sortedList.map(result => `
        <div class="charSim-character-item" data-avatar="${result.avatar}">
            <img src="${getThumbnailUrl('avatar', result.avatar)}" alt="${result.name}">
            <div class="charSim-name-block">
                <span class="charSim-name">${result.name}</span>
                <span class="charSim-sub">${result.filename}</span>
            </div>
            <div class="charSim-score" title="Uniqueness Score">${result.score.toFixed(4)}</div>
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
        const headerEl = $(`<div class="charSim-cluster-header"><span>Group Size: ${groupData.members.length}</span><span class="charSim-score">Avg Sim: ${(1-groupData.clusterUniqueness).toFixed(4)}</span></div>`);
        groupEl.append(headerEl);
        
        // Sort members by distance from cluster center
        const sortedMembers = groupData.members.sort((a, b) => a.intraClusterDistance - b.intraClusterDistance);
        
        sortedMembers.forEach(member => {
            const charEl = $(`<div class="charSim-character-item" data-avatar="${member.avatar}"><img src="${getThumbnailUrl('avatar', member.avatar)}" alt="${member.name}"><span class="charSim-name">${member.name}</span><div class="charSim-score" title="Distance from center">${member.intraClusterDistance.toFixed(4)}</div></div>`);
            groupEl.append(charEl);
        });
        container.append(groupEl);
        if (index < similarGroups.length - 1) container.append('<hr class="charSim-delimiter">');
    });
}

// --- API & DATA HANDLING ---

async function onEmbeddingsLoad() {
    let koboldUrl = extension_settings[extensionName].koboldUrl.trim();
    if (!koboldUrl) {
        toastr.warning('Please set the KoboldCpp URL in settings.');
        return;
    }
    if (!koboldUrl.startsWith('http')) koboldUrl = 'http://' + koboldUrl;

    const buttons = $('.charSim-btn');
    let toastId = null;

    try {
        buttons.prop('disabled', true);
        
        // Identify missing embeddings
        const textsToEmbed = [];
        for (const char of characters) {
            if (!characterEmbeddings.has(char.avatar)) {
                const combinedText = fieldsToEmbed.map(field => char[field] || '').join('\n').trim();
                if (combinedText) {
                    textsToEmbed.push({ avatar: char.avatar, text: combinedText });
                }
            }
        }

        if (textsToEmbed.length === 0) {
            toastr.info('All characters are already embedded.');
            buttons.prop('disabled', false);
            return;
        }

        toastId = toastr.info(`Generating embeddings for ${textsToEmbed.length} new characters...`, 'Embedding', { timeOut: 0, extendedTimeOut: 0 });

        // Batch processing to avoid timeouts
        const batchSize = 10;
        for (let i = 0; i < textsToEmbed.length; i += batchSize) {
            const batch = textsToEmbed.slice(i, i + batchSize);
            const response = await fetch('/api/backends/kobold/embed', {
                method: 'POST',
                headers: getRequestHeaders(),
                body: JSON.stringify({
                    items: batch.map(item => item.text),
                    server: koboldUrl,
                }),
            });

            if (!response.ok) throw new Error(`API Error: ${response.status}`);
            const data = await response.json();
            
            if (!data.embeddings) throw new Error('Invalid API response');

            for (let j = 0; j < data.embeddings.length; j++) {
                const avatar = batch[j].avatar;
                const embedding = data.embeddings[j];
                if (embedding && Array.isArray(embedding) && embedding.length > 0) {
                    characterEmbeddings.set(avatar, embedding);
                }
            }
            
            // Update progress in toast if possible, or console
            console.log(`Embedded ${Math.min(i + batchSize, textsToEmbed.length)} / ${textsToEmbed.length}`);
        }

        toastr.remove(toastId);
        toastr.success(`Successfully loaded embeddings for ${characterEmbeddings.size} characters.`);

    } catch (error) {
        if (toastId) toastr.remove(toastId);
        toastr.error(`Embedding Error: ${error.message}`);
        console.error(error);
    } finally {
        buttons.prop('disabled', false);
    }
}

// --- CACHING (OFFLINE MODE) ---

function onExportCache() {
    if (characterEmbeddings.size === 0) {
        toastr.warning("No embeddings to save.");
        return;
    }
    const cacheObj = {};
    for (const [avatar, embedding] of characterEmbeddings.entries()) {
        cacheObj[avatar] = embedding;
    }
    const jsonStr = JSON.stringify(cacheObj);
    const blob = new Blob([jsonStr], { type: "application/json" });
    saveFile(blob, "character_embeddings_cache.json");
}

function onImportCache() {
    $('#charSimImportInput').click();
}

function handleImportFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const cacheObj = JSON.parse(e.target.result);
            let count = 0;
            for (const key in cacheObj) {
                if (Array.isArray(cacheObj[key])) {
                    characterEmbeddings.set(key, cacheObj[key]);
                    count++;
                }
            }
            toastr.success(`Loaded ${count} embeddings from cache.`);
            // Clear file input
            event.target.value = '';
        } catch (err) {
            toastr.error("Failed to parse cache file.");
            console.error(err);
        }
    };
    reader.readAsText(file);
}

// --- WORKER MANAGEMENT ---

function initWorker() {
    if (calculationWorker) return;

    const workerCode = `
    // --- WORKER SCRIPT ---
    
    // Similarity Functions
    function cosineSimilarity(vecA, vecB) {
        let dot = 0.0, normA = 0.0, normB = 0.0;
        for (let i = 0; i < vecA.length; i++) {
            dot += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // Distance = 1 - Cosine Similarity
    function cosineDistance(vecA, vecB) {
        return 1.0 - cosineSimilarity(vecA, vecB);
    }

    function calculateMeanVector(vectors) {
        if (!vectors.length) return null;
        const dim = vectors[0].length;
        const mean = new Float32Array(dim).fill(0);
        for (const vec of vectors) {
            for (let i = 0; i < dim; i++) mean[i] += vec[i];
        }
        for (let i = 0; i < dim; i++) mean[i] /= vectors.length;
        return mean;
    }

    // Algorithms

    function runGlobalMean(items) {
        const vectors = items.map(i => i.embedding);
        const meanVec = calculateMeanVector(vectors);
        const results = [];
        
        for (let i = 0; i < items.length; i++) {
            const dist = cosineDistance(items[i].embedding, meanVec);
            results.push({ index: i, score: dist });
        }
        return results;
    }

    function runKNN(items, k) {
        const n = items.length;
        const results = [];
        const safeK = Math.min(k, n - 1);
        
        if (safeK <= 0) {
             return items.map((_, i) => ({ index: i, score: 0 }));
        }

        // Calculate distance matrix (or row by row to save memory)
        for (let i = 0; i < n; i++) {
            const distances = [];
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                distances.push(cosineDistance(items[i].embedding, items[j].embedding));
            }
            // Sort distances asc
            distances.sort((a, b) => a - b);
            // Take top k
            let sum = 0;
            for(let x = 0; x < safeK; x++) sum += distances[x];
            results.push({ index: i, score: sum / safeK });
        }
        return results;
    }

    function runHarmonic(items) {
        const n = items.length;
        const results = [];
        // Weights: 1, 1/2, 1/3 ...
        const weights = [];
        for(let i=1; i < n; i++) weights.push(1/i);

        for (let i = 0; i < n; i++) {
            const distances = [];
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                distances.push(cosineDistance(items[i].embedding, items[j].embedding));
            }
            distances.sort((a, b) => a - b);
            
            let score = 0;
            for(let x=0; x < distances.length; x++) {
                score += distances[x] * weights[x];
            }
            results.push({ index: i, score: score });
        }
        return results;
    }

    // Clustering (Simple Agglomerative or Threshold based grouping)
    // Re-using the logic from original extension but cleaned up
    function runClustering(items, threshold) {
        // Precompute full distance matrix for speed
        const n = items.length;
        const adj = new Array(n).fill(0).map(() => new Array(n).fill(false));
        
        for(let i=0; i<n; i++){
            for(let j=i+1; j<n; j++){
                const sim = cosineSimilarity(items[i].embedding, items[j].embedding);
                if(sim >= threshold) {
                    adj[i][j] = true;
                    adj[j][i] = true;
                }
            }
        }

        const visited = new Array(n).fill(false);
        const groups = [];

        for(let i=0; i<n; i++) {
            if(!visited[i]) {
                const group = [];
                const queue = [i];
                visited[i] = true;
                while(queue.length > 0) {
                    const curr = queue.shift();
                    group.push(curr);
                    for(let neighbor=0; neighbor<n; neighbor++) {
                        if(adj[curr][neighbor] && !visited[neighbor]) {
                            visited[neighbor] = true;
                            queue.push(neighbor);
                        }
                    }
                }
                groups.push(group);
            }
        }
        
        // Return array of arrays of indices
        return groups;
    }

    // Message Handler
    self.onmessage = function(e) {
        const { type, items, params } = e.data;
        
        try {
            if (type === 'uniqueness') {
                let results;
                if (params.algorithm === 'knn') {
                    results = runKNN(items, params.k);
                } else if (params.algorithm === 'harmonic') {
                    results = runHarmonic(items);
                } else {
                    results = runGlobalMean(items);
                }
                self.postMessage({ type: 'uniqueness_result', results });
            } 
            else if (type === 'clustering') {
                const groups = runClustering(items, params.threshold);
                self.postMessage({ type: 'clustering_result', groups });
            }
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
    };
    `;
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    calculationWorker = new Worker(URL.createObjectURL(blob));
    
    calculationWorker.onmessage = function(e) {
        const data = e.data;
        const buttons = $('.charSim-btn');
        buttons.prop('disabled', false);

        if (data.type === 'error') {
            toastr.error(data.message);
            return;
        }

        if (data.type === 'uniqueness_result') {
            // Map indices back to avatars
            const items = lastWorkerPayload; // Access closed over var
            uniquenessResults = data.results.map(r => {
                const charInfo = items[r.index];
                return {
                    avatar: charInfo.avatar,
                    name: charInfo.name,
                    filename: charInfo.avatar, // Usually avatar filename is key
                    score: r.score
                };
            });
            renderUniquenessList();
            toastr.success('Analysis complete.');
        }

        if (data.type === 'clustering_result') {
            const items = lastWorkerPayload;
            const groups = data.groups;
            
            // Process groups into renderable format
            clusterResults = [];
            
            // We need a helper to calc mean of a group for display logic
            // (Re-implement simple mean logic here or pass back from worker)
            // For simplicity, we just calc stats here in main thread for display
            
            groups.forEach(groupIndices => {
                const groupMembers = groupIndices.map(idx => items[idx]);
                // Calculate centroid
                const dim = groupMembers[0].embedding.length;
                const centroid = new Array(dim).fill(0);
                for(let m of groupMembers) {
                    for(let i=0; i<dim; i++) centroid[i] += m.embedding[i];
                }
                for(let i=0; i<dim; i++) centroid[i] /= groupMembers.length;

                // Calculate distances
                const processedMembers = groupMembers.map(m => {
                    let dot=0, nA=0, nB=0;
                    for(let i=0; i<dim; i++) {
                        dot += m.embedding[i]*centroid[i];
                        nA += m.embedding[i]**2;
                        nB += centroid[i]**2;
                    }
                    const sim = dot / (Math.sqrt(nA)*Math.sqrt(nB));
                    return {
                        avatar: m.avatar,
                        name: m.name,
                        intraClusterDistance: 1 - sim
                    };
                });

                // Cluster uniqueness (distance of centroid from global library mean) is harder to calc cheaply here without global context
                // We'll just store 0 for now or calculate later if needed.
                clusterResults.push({
                    clusterUniqueness: 0, // Placeholder
                    members: processedMembers
                });
            });

            clusterResults.sort((a, b) => b.members.length - a.members.length);
            renderClusterList();
            toastr.success(`Found ${clusterResults.filter(g => g.members.length > 1).length} groups.`);
        }
    };
}

let lastWorkerPayload = [];

function prepareWorkerData() {
    const items = [];
    for (const [avatar, embedding] of characterEmbeddings.entries()) {
        const char = characters.find(c => c.avatar === avatar);
        if (char && embedding) {
            items.push({ avatar: avatar, name: char.name, embedding: embedding });
        }
    }
    lastWorkerPayload = items;
    return items;
}

function onCalculateUniqueness() {
    if (characterEmbeddings.size === 0) {
        toastr.warning('Please load character embeddings first.');
        return;
    }
    
    const algo = $('#charSimAlgorithm').val();
    const k = parseInt($('#charSimKParam').val()) || 5;
    
    initWorker();
    const items = prepareWorkerData();
    
    $('.charSim-btn').prop('disabled', true);
    toastr.info('Calculating Uniqueness...');
    
    calculationWorker.postMessage({
        type: 'uniqueness',
        items: items,
        params: { algorithm: algo, k: k }
    });
}

function onCalculateClusters() {
    if (characterEmbeddings.size === 0) {
        toastr.warning('Please load character embeddings first.');
        return;
    }
    const threshold = parseFloat($('#charSimThresholdSlider').val());
    
    initWorker();
    const items = prepareWorkerData();
    
    $('.charSim-btn').prop('disabled', true);
    toastr.info('Clustering...');
    
    calculationWorker.postMessage({
        type: 'clustering',
        items: items,
        params: { threshold: threshold }
    });
}

// --- SETUP & EVENT LISTENERS ---

jQuery(() => {
    // Settings Injection
    extension_settings[extensionName] = extension_settings[extensionName] || {};
    Object.assign(defaultSettings, extension_settings[extensionName]);
    Object.assign(extension_settings[extensionName], defaultSettings); // Ensure defaults exist

    const settingsHtml = `
    <div class="character-similarity-settings">
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>Character Similarity</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content">
                <div class="character-similarity_block">
                    <label>KoboldCpp URL</label>
                    <input id="kobold_url_input" class="text_pole" type="text" value="${extension_settings[extensionName].koboldUrl}">
                </div>
            </div>
        </div>
    </div>`;
    $("#extensions_settings2").append(settingsHtml);
    
    $("#kobold_url_input").on("input", (e) => { 
        extension_settings[extensionName].koboldUrl = e.target.value; 
        saveSettingsDebounced(); 
    });

    // Panel HTML
    const panelHtml = `
    <div id="characterSimilarityPanel">
        <div class="charSimPanel-content">
            <div class="charSimPanel-header">
                <div class="fa-solid fa-grip drag-grabber"></div>
                <b>Character Similarity & Analysis</b>
                <div id="charSimCloseBtn" class="fa-solid fa-circle-xmark floating_panel_close"></div>
            </div>
            <div class="charSimPanel-body">
                <div class="charSim-tabs">
                    <div class="charSim-tab-button active" data-tab="uniqueness">Uniqueness</div>
                    <div class="charSim-tab-button" data-tab="clustering">Clustering</div>
                    <div class="charSim-tab-button" data-tab="data">Data Management</div>
                </div>
                
                <!-- UNIQUENESS VIEW -->
                <div id="charSimUniquenessView" class="charSim-tab-pane active">
                    <div class="charSimPanel-controls">
                        <label>Algo:</label>
                        <select id="charSimAlgorithm" class="text_pole" style="width: auto;">
                            <option value="mean">Global Mean</option>
                            <option value="knn" selected>k-Nearest Neighbors</option>
                            <option value="harmonic">Harmonic Mean</option>
                        </select>
                        <input id="charSimKParam" class="text_pole" type="number" min="1" max="50" value="${defaultSettings.algoParams.k}" style="width: 50px;" title="k (for KNN)">
                        
                        <div id="charSimCalcUniquenessBtn" class="menu_button charSim-btn">Calculate</div>
                        <div class="spacer"></div>
                        <div id="charSimSortBtn" class="menu_button menu_button_icon fa-solid fa-arrow-down" title="Sort by Score"></div>
                    </div>
                    <div id="charSimUniquenessList" class="charSim-list-container"></div>
                </div>

                <!-- CLUSTERING VIEW -->
                <div id="charSimClusteringView" class="charSim-tab-pane">
                    <div class="charSimPanel-controls">
                        <div id="charSimCalcClustersBtn" class="menu_button charSim-btn">Calculate Clusters</div>
                        <div class="spacer"></div>
                        <label style="font-size: 0.9em;">Sim Threshold: <span id="charSimThresholdValue">0.95</span></label>
                        <input type="range" id="charSimThresholdSlider" min="0.5" max="1.0" step="0.01" value="0.95">
                    </div>
                    <div id="charSimClusteringList" class="charSim-list-container">
                        <p class="charSim-no-results">Select parameters and click Calculate.</p>
                    </div>
                </div>

                <!-- DATA VIEW (Cache/Load) -->
                <div id="charSimDataView" class="charSim-tab-pane">
                    <div class="charSimPanel-controls column">
                        <p>Manage embeddings cache to use offline or save progress.</p>
                        <div class="flex-row gap-10">
                            <div id="charSimLoadBtn" class="menu_button charSim-btn">Fetch from API</div>
                            <div id="charSimExportBtn" class="menu_button charSim-btn">Export Cache to JSON</div>
                            <div id="charSimImportBtn" class="menu_button charSim-btn">Import Cache from JSON</div>
                            <input type="file" id="charSimImportInput" style="display:none" accept=".json">
                        </div>
                        <hr>
                        <small>Fetching requires running KoboldCpp with an embedding model.</small>
                    </div>
                </div>

            </div>
        </div>
    </div>`;
    $('#movingDivs').append(panelHtml);

    // Listeners
    $('#charSimCloseBtn').on('click', () => $('#characterSimilarityPanel').removeClass('open'));
    $('#charSimLoadBtn').on('click', onEmbeddingsLoad);
    $('#charSimExportBtn').on('click', onExportCache);
    $('#charSimImportBtn').on('click', onImportCache);
    $('#charSimImportInput').on('change', handleImportFile);

    $('#charSimCalcUniquenessBtn').on('click', onCalculateUniqueness);
    $('#charSimCalcClustersBtn').on('click', onCalculateClusters);
    
    $('#charSimSortBtn').on('click', function() {
        $(this).toggleClass('fa-arrow-down fa-arrow-up');
        renderUniquenessList();
    });

    $('.charSim-tab-button').on('click', function() {
        const tab = $(this).data('tab');
        $('.charSim-tab-button').removeClass('active');
        $(this).addClass('active');
        $('.charSim-tab-pane').removeClass('active');
        $(`#charSim${tab.charAt(0).toUpperCase() + tab.slice(1)}View`).addClass('active');
    });

    $('#charSimThresholdSlider').on('input', function() {
        const val = parseFloat($(this).val());
        $('#charSimThresholdValue').text(val.toFixed(2));
    });

    // Sidebar Button
    const openButton = document.createElement('div');
    openButton.id = 'characterSimilarityOpenBtn';
    openButton.classList.add('menu_button', 'fa-solid', 'fa-network-wired', 'faSmallFontSquareFix');
    openButton.title = 'Character Analysis';
    openButton.addEventListener('click', () => {
        populateCharacterList();
        $('#characterSimilarityPanel').addClass('open');
    });

    // Add to UI
    const buttonContainer = document.getElementById('rm_buttons_container');
    if (buttonContainer) buttonContainer.append(openButton);
    
    eventSource.on(event_types.APP_READY, populateCharacterList);
});