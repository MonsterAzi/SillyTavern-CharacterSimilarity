import { state, extensionName } from "./state.js";
import { characters, getThumbnailUrl, saveSettingsDebounced } from "../../../../../script.js";

// --- HTML Generators ---

function generateCharacterRow(name, avatar, score = null, scoreTooltip = "") {
    const scoreHtml = score !== null 
        ? `<div class="charSim-score" title="${scoreTooltip}">${score.toFixed(4)}</div>` 
        : '';

    return `
        <div class="charSim-character-item" data-avatar="${avatar}">
            <img src="${getThumbnailUrl('avatar', avatar)}" alt="${name}'s avatar">
            <span class="charSim-name">${name}</span>
            ${scoreHtml}
        </div>
    `;
}

function generateClusterGroup(groupData) {
    const memberRows = groupData.members
        .sort((a, b) => a.intraClusterDistance - b.intraClusterDistance)
        .map(m => generateCharacterRow(m.name, m.avatar, m.intraClusterDistance, "Distance from cluster average"))
        .join('');

    return `
        <div class="charSim-cluster-group">
            <div class="charSim-cluster-header">
                <span>Cluster Uniqueness:</span>
                <span class="charSim-score">${groupData.clusterUniqueness.toFixed(4)}</span>
            </div>
            ${memberRows}
        </div>
    `;
}

export const UIManager = {
    injectSettings() {
        const html = `
        <div class="character-similarity-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>Character Similarity</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <div class="character-similarity_block">
                        <label for="kobold_url_input">KoboldCpp URL</label>
                        <input id="kobold_url_input" class="text_pole" type="text" 
                               value="${state.setting.koboldUrl}" placeholder="http://192.168.1.100:5001">
                        <small><b>MUST include http:// and be a network IP for multi-device access.</b></small>
                    </div>
                </div>
            </div>
        </div>`;
        $("#extensions_settings2").append(html);
        $("#kobold_url_input").on("input", (e) => {
            state.setting.koboldUrl = e.target.value;
            state.saveSettings();
        });
    },

    injectMainPanel() {
        const html = `
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
                    
                    <!-- Uniqueness Tab -->
                    <div id="charSimUniquenessView" class="charSim-tab-pane active">
                        <div class="charSimPanel-controls">
                            <div id="charSimLoadBtn" class="menu_button">Load Embeddings</div>
                            <div id="charSimCalcUniquenessBtn" class="menu_button">Calculate Uniqueness</div>
                            <div class="spacer"></div>
                            <div id="charSimSortBtn" class="menu_button menu_button_icon fa-solid fa-arrow-down" title="Sort Descending"></div>
                        </div>
                        <div id="charSimUniquenessList" class="charSim-list-container"></div>
                    </div>
                    
                    <!-- Clustering Tab -->
                    <div id="charSimClusteringView" class="charSim-tab-pane">
                        <div class="charSimPanel-controls">
                            <div id="charSimCalcClustersBtn" class="menu_button">Calculate Clusters</div>
                            <div class="charSim-slider-group">
                                <label for="charSimThresholdSlider">Threshold: <span id="charSimThresholdValue">${state.setting.clusterThreshold.toFixed(2)}</span></label>
                                <input type="range" id="charSimThresholdSlider" min="0.5" max="1.0" step="0.01" value="${state.setting.clusterThreshold}">
                            </div>
                        </div>
                        <div id="charSimClusteringList" class="charSim-list-container">
                            <p class="charSim-no-results">Load embeddings and click "Calculate Clusters" to see results.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        $('#movingDivs').append(html);
    },

    injectTriggerButton() {
        const openButton = document.createElement('div');
        openButton.id = 'characterSimilarityOpenBtn';
        openButton.classList.add('menu_button', 'fa-solid', 'fa-project-diagram', 'faSmallFontSquareFix');
        openButton.title = 'Find Similar Characters';
        
        const buttonContainer = document.getElementById('rm_buttons_container');
        if (buttonContainer) {
            buttonContainer.append(openButton);
        } else {
            // Fallback for older layouts
            const searchBar = document.getElementById('character_search_bar');
            if(searchBar) {
                document.getElementById('form_character_search_form').insertBefore(openButton, searchBar);
            }
        }
        return openButton;
    },

    renderCharacterList(containerId, listData) {
        const container = $(containerId);
        if (!listData || listData.length === 0) {
            // Fallback: Show all characters without scores
            const sortedChars = characters.slice().sort((a, b) => a.name.localeCompare(b.name));
            container.html(sortedChars.map(c => generateCharacterRow(c.name, c.avatar)).join(''));
            return;
        }

        // Determine sort order based on UI state
        const isDescending = $('#charSimSortBtn').hasClass('fa-arrow-down');
        const sorted = [...listData].sort((a, b) => isDescending ? b.distance - a.distance : a.distance - b.distance);
        
        container.html(sorted.map(item => generateCharacterRow(item.name, item.avatar, item.distance)).join(''));
    },

    renderClusters(containerId, clusterGroups) {
        const container = $(containerId);
        container.empty();

        const significantGroups = clusterGroups.filter(g => g.members.length > 1);
        
        if (significantGroups.length === 0) {
            container.html('<p class="charSim-no-results">No similar character groups found at this threshold.</p>');
            return;
        }

        significantGroups.forEach(group => {
            container.append(generateClusterGroup(group));
        });
    }
};