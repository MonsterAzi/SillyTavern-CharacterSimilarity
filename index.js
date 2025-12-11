// File: /index.js

import { characters, eventSource, event_types } from "../../../script.js";

// Import local modules
import { state } from "./modules/state.js";
import { UIManager } from "./modules/ui.js";
import { fetchEmbeddings } from "./modules/api.js";
import { calculateMeanEmbedding, calculateManhattanDistance } from "./modules/utils.js";
import { createClusteringWorker } from "./modules/clustering.js";

// --- Logic Controllers ---

async function handleLoadEmbeddings() {
    const buttons = $('#charSimLoadBtn, #charSimCalcUniquenessBtn, #charSimCalcClustersBtn');
    let loadingToast = null;

    try {
        buttons.prop('disabled', true);
        state.clearData();

        // Use a spinner icon and capture the specific toast object
        loadingToast = toastr.info(
            `<i class="fa-solid fa-spinner fa-spin"></i> Preparing ${characters.length} characters...`, 
            'Loading Embeddings', 
            { timeOut: 0, extendedTimeOut: 0, preventDuplicates: true }
        );
        
        const results = await fetchEmbeddings(characters);
        
        for (const res of results) {
            state.embeddings.set(res.avatar, res.embedding);
        }

        // Immediately clear the loading toast before showing success
        if (loadingToast) toastr.clear(loadingToast);
        toastr.success(`Successfully loaded embeddings for ${state.embeddings.size} characters.`);
    } catch (error) {
        if (loadingToast) toastr.clear(loadingToast);
        toastr.error(error.message, 'Error');
        console.error("Character Similarity Error:", error);
    } finally {
        buttons.prop('disabled', false);
    }
}

function handleCalculateUniqueness() {
    if (state.embeddings.size === 0) return toastr.warning('Please load embeddings first.');

    // Add spinner here too for consistency, even if it is usually fast
    const calcToast = toastr.info(
        '<i class="fa-solid fa-spinner fa-spin"></i> Calculating...', 
        '', 
        { timeOut: 0, extendedTimeOut: 0 }
    );
    
    // Allow UI to render the toast before freezing main thread with math
    setTimeout(() => {
        try {
            const embeddingsList = Array.from(state.embeddings.values());
            const libraryMean = calculateMeanEmbedding(embeddingsList);
            
            if (!libraryMean) {
                toastr.clear(calcToast);
                return toastr.error('Calculation failed.');
            }

            const results = [];
            for (const [avatar, embedding] of state.embeddings.entries()) {
                const distance = calculateManhattanDistance(embedding, libraryMean);
                const char = characters.find(c => c.avatar === avatar);
                if (char) results.push({ avatar, name: char.name, distance });
            }

            state.uniquenessResults = results;
            UIManager.renderCharacterList('#charSimUniquenessList', state.uniquenessResults);
            
            toastr.clear(calcToast);
            toastr.success('Calculation complete.');
        } catch (e) {
            toastr.clear(calcToast);
            toastr.error(e.message);
        }
    }, 50);
}

function handleCalculateClusters() {
    if (state.embeddings.size === 0) return toastr.warning('Please load embeddings first.');

    const buttons = $('#charSimLoadBtn, #charSimCalcUniquenessBtn, #charSimCalcClustersBtn');
    buttons.prop('disabled', true);

    const threshold = state.setting.clusterThreshold;
    
    // Persistent toast with spinner
    const loadingToast = toastr.info(
        `<i class="fa-solid fa-spinner fa-spin"></i> Clustering at ${threshold.toFixed(2)}...`, 
        'Clustering', 
        { timeOut: 0, extendedTimeOut: 0 }
    );

    const worker = createClusteringWorker();
    
    // Convert Map to Array for transfer
    const dataForWorker = Array.from(state.embeddings.entries()).map(([avatar, embedding]) => ({ avatar, embedding }));

    worker.postMessage({ embeddings: dataForWorker, threshold });

    worker.onmessage = (e) => {
        const groups = e.data;
        processClusterResults(groups);
        
        // Clear loading, show success
        toastr.clear(loadingToast);
        toastr.success(`Found ${state.clusterResults.filter(g => g.members.length > 1).length} groups.`);
        
        buttons.prop('disabled', false);
        worker.terminate();
    };

    worker.onerror = (err) => {
        toastr.clear(loadingToast);
        toastr.error('Worker error: ' + err.message);
        buttons.prop('disabled', false);
        worker.terminate();
    };
}

function processClusterResults(rawGroups) {
    const embeddingsList = Array.from(state.embeddings.values());
    const libraryMean = calculateMeanEmbedding(embeddingsList);
    
    const processedGroups = [];

    for (const groupMembers of rawGroups) {
        // Calculate the center of this specific cluster
        const memberEmbeddings = groupMembers.map(m => m.embedding); // Worker returns embedding
        const clusterMean = calculateMeanEmbedding(memberEmbeddings);
        
        // How unique is this cluster compared to the whole library?
        const clusterUniqueness = calculateManhattanDistance(clusterMean, libraryMean);

        // Map members with their distance from the Cluster Center
        const members = groupMembers.map(m => {
            const char = characters.find(c => c.avatar === m.avatar);
            const intraClusterDistance = calculateManhattanDistance(m.embedding, clusterMean);
            return { 
                avatar: m.avatar, 
                name: char ? char.name : 'Unknown', 
                intraClusterDistance 
            };
        });

        processedGroups.push({ clusterUniqueness, members });
    }

    state.clusterResults = processedGroups.sort((a, b) => b.members.length - a.members.length);
    UIManager.renderClusters('#charSimClusteringList', state.clusterResults);
}

// --- Initialization ---

jQuery(() => {
    state.initSettings();
    UIManager.injectSettings();
    UIManager.injectMainPanel();
    
    const openBtn = UIManager.injectTriggerButton();
    $(openBtn).on('click', () => {
        if (state.uniquenessResults.length > 0) {
            UIManager.renderCharacterList('#charSimUniquenessList', state.uniquenessResults);
        } else {
            UIManager.renderCharacterList('#charSimUniquenessList', []);
        }
        $('#characterSimilarityPanel').addClass('open');
    });

    // Event Bindings
    $('#charSimCloseBtn').on('click', () => $('#characterSimilarityPanel').removeClass('open'));
    $('#charSimLoadBtn').on('click', handleLoadEmbeddings);
    $('#charSimCalcUniquenessBtn').on('click', handleCalculateUniqueness);
    $('#charSimCalcClustersBtn').on('click', handleCalculateClusters);
    
    $('#charSimSortBtn').on('click', function() { 
        $(this).toggleClass('fa-arrow-down fa-arrow-up'); 
        $(this).attr('title', $(this).hasClass('fa-arrow-down') ? 'Sort Descending' : 'Sort Ascending'); 
        UIManager.renderCharacterList('#charSimUniquenessList', state.uniquenessResults); 
    });

    $('.charSim-tab-button').on('click', function() { 
        const tab = $(this).data('tab'); 
        $('.charSim-tab-button').removeClass('active'); 
        $(this).addClass('active'); 
        $('.charSim-tab-pane').removeClass('active'); 
        $(`#charSim${tab.charAt(0).toUpperCase() + tab.slice(1)}View`).addClass('active'); 
    });

    $('#charSimThresholdSlider').on('input', function() { 
        const value = parseFloat($(this).val()); 
        $('#charSimThresholdValue').text(value.toFixed(2)); 
        state.setting.clusterThreshold = value; 
        state.saveSettings(); 
    });

    // Handle character reload event
    eventSource.on(event_types.APP_READY, () => {
    });
});