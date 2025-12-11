import { extension_settings } from "../../../../extensions.js";
import { createCharacterItemHTML, createClusterGroupHTML } from "../utils/dom.js";

export class CharacterSimilarityPanel {
    constructor(embeddingManager, analysisEngine) {
        this.manager = embeddingManager;
        this.analysis = analysisEngine;
        this.isOpen = false;
        this.currentTab = 'uniqueness';
        
        this.init();
    }

    init() {
        this.renderSettings();
        this.renderPanel();
        this.bindEvents();
        this.addOpenButton();
    }

    renderSettings() {
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
                                value="${extension_settings['character_similarity']?.koboldUrl || 'http://127.0.0.1:5001'}">
                            <small><b>MUST include http:// for network access.</b></small>
                        </div>
                    </div>
                </div>
            </div>`;
        $("#extensions_settings2").append(html);

        $("#kobold_url_input").on("input", (e) => {
            extension_settings['character_similarity'].koboldUrl = e.target.value;
            saveSettingsDebounced();
        });
    }

    renderPanel() {
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
                                <div class="spacer"></div>
                                <label for="charSimThresholdSlider">Threshold: <span id="charSimThresholdValue">0.95</span></label>
                                <input type="range" id="charSimThresholdSlider" min="0.5" max="1.0" step="0.01" value="0.95">
                            </div>
                            <div id="charSimClusteringList" class="charSim-list-container">
                                <p class="charSim-no-results">Load embeddings to start.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
        $('#movingDivs').append(html);
    }

    bindEvents() {
        // Panel Controls
        $('#charSimCloseBtn').on('click', () => this.togglePanel(false));
        
        $('.charSim-tab-button').on('click', (e) => {
            const tab = $(e.currentTarget).data('tab');
            this.switchTab(tab);
        });

        // Buttons
        $('#charSimLoadBtn').on('click', () => this.handleLoadEmbeddings());
        $('#charSimCalcUniquenessBtn').on('click', () => this.handleUniqueness());
        $('#charSimCalcClustersBtn').on('click', () => this.handleClustering());
        
        // Sort & Slider
        $('#charSimSortBtn').on('click', (e) => {
            const btn = $(e.currentTarget);
            btn.toggleClass('fa-arrow-down fa-arrow-up');
            this.renderUniquenessList(btn.hasClass('fa-arrow-down'));
        });

        $('#charSimThresholdSlider').on('input', (e) => {
            const val = parseFloat($(e.target).val());
            $('#charSimThresholdValue').text(val.toFixed(2));
            extension_settings['character_similarity'].clusterThreshold = val;
            saveSettingsDebounced();
        });
    }

    addOpenButton() {
        const btn = document.createElement('div');
        btn.id = 'characterSimilarityOpenBtn';
        btn.className = 'menu_button fa-solid fa-project-diagram faSmallFontSquareFix';
        btn.title = 'Find Similar Characters';
        btn.onclick = () => {
            // If data exists, refresh list, else just show
            if (this.manager.getEmbeddings().size > 0) {
                if (this.currentTab === 'uniqueness') this.handleUniqueness();
                else this.renderClusteringList([]); // Clear or refresh
            }
            this.togglePanel(true);
        };
        
        const container = document.getElementById('rm_buttons_container');
        if (container) container.append(btn);
        else document.getElementById('form_character_search_form').insertBefore(btn, document.getElementById('character_search_bar'));
    }

    // --- Actions ---

    togglePanel(show) {
        this.isOpen = show;
        const panel = $('#characterSimilarityPanel');
        if (show) panel.addClass('open');
        else panel.removeClass('open');
    }

    switchTab(tabName) {
        this.currentTab = tabName;
        $('.charSim-tab-button').removeClass('active');
        $(`.charSim-tab-button[data-tab="${tabName}"]`).addClass('active');
        $('.charSim-tab-pane').removeClass('active');
        $(`#charSim${tabName.charAt(0).toUpperCase() + tabName.slice(1)}View`).addClass('active');
    }

    async handleLoadEmbeddings() {
        const btn = $('#charSimLoadBtn');
        btn.prop('disabled', true);
        
        const success = await this.manager.fetchEmbeddings();
        
        btn.prop('disabled', false);
        if (success) {
            toastr.success(`Loaded ${this.manager.getEmbeddings().size} embeddings.`);
            // Reset views
            $('#charSimUniquenessList').empty();
            $('#charSimClusteringList').empty();
        }
    }

    handleUniqueness() {
        if (this.manager.getEmbeddings().size === 0) {
            toastr.warning('Load embeddings first.');
            return;
        }
        
        const results = this.analysis.calculateUniqueness();
        this.renderUniquenessList(true, results);
        toastr.success('Uniqueness calculated.');
    }

    renderUniquenessList(descending = true, data = null) {
        const container = $('#charSimUniquenessList');
        let results = data;
        
        if (!results) {
            // Try to get from internal state if not passed
            toastr.warning('No data available. Calculate first.');
            return;
        }

        results.sort((a, b) => descending ? b.distance - a.distance : a.distance - b.distance);

        const html = results.map(item => createCharacterItemHTML(item.name, item.avatar, item.distance)).join('');
        container.html(html);
    }

    async handleClustering() {
        if (this.manager.getEmbeddings().size === 0) {
            toastr.warning('Load embeddings first.');
            return;
        }

        const threshold = extension_settings['character_similarity']?.clusterThreshold || 0.95;
        const btn = $('#charSimCalcClustersBtn');
        btn.prop('disabled', true);
        
        toastr.info('Calculating clusters...', 'Working');

        try {
            const groups = await this.analysis.calculateClusters(threshold);
            this.renderClusteringList(groups);
            toastr.success(`Found ${groups.length} groups.`);
        } catch (e) {
            toastr.error(e);
        } finally {
            btn.prop('disabled', false);
        }
    }

    renderClusteringList(groups) {
        const container = $('#charSimClusteringList');
        if (!groups || groups.length === 0) {
            container.html('<p class="charSim-no-results">No clusters found at this threshold.</p>');
            return;
        }

        const html = groups.map(group => createClusterGroupHTML(group)).join('<hr class="charSim-delimiter">');
        container.html(html);
    }
}