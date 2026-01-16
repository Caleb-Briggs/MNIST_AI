import { createForceGraph } from './forceGraph.js';
import { createCorrelationMatrix } from './correlation.js';
import { createImageGrid } from './imageGrid.js';
import { createDetailsPanel } from './details.js';
import { createTrainingCurves } from './trainingCurves.js';

// Global state
const state = {
    data: null,
    selectedImage: null,
    highlightedImages: new Set(),
    filters: {
        digit: 'all',
        sort: 'difficulty-asc',
        minDifficulty: 0
    }
};

// Event bus for cross-component communication
const events = {
    listeners: {},
    on(event, callback) {
        if (!this.listeners[event]) this.listeners[event] = [];
        this.listeners[event].push(callback);
    },
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(cb => cb(data));
        }
    }
};

// Tooltip
const tooltip = {
    el: document.getElementById('tooltip'),
    show(html, x, y) {
        this.el.innerHTML = html;
        this.el.classList.remove('hidden');

        const rect = this.el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        let left = x + 10;
        let top = y + 10;

        if (left + rect.width > vw) left = x - rect.width - 10;
        if (top + rect.height > vh) top = y - rect.height - 10;

        this.el.style.left = left + 'px';
        this.el.style.top = top + 'px';
    },
    hide() {
        this.el.classList.add('hidden');
    }
};

// Load data and initialize
async function init() {
    try {
        const response = await fetch('data/dashboard_data.json');
        if (!response.ok) throw new Error('Failed to load data');
        state.data = await response.json();

        // Update stats
        document.getElementById('stats').textContent =
            `${state.data.config.num_models} models | ${state.data.images.length} images | ${state.data.config.samples_per_digit} samples/digit`;

        // Initialize components
        createForceGraph('#treemap', state, events, tooltip);
        createCorrelationMatrix('#correlation', state, events, tooltip);
        createImageGrid('#image-grid', state, events, tooltip);
        createDetailsPanel('#details', state, events);
        createTrainingCurves('#training-curves', state, events);

        // Setup filters
        setupFilters();

        // Setup modals
        setupModals();

    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('stats').textContent = 'Error loading data. Make sure dashboard_data.json exists in the data/ folder.';
    }
}

function setupFilters() {
    const digitFilter = document.getElementById('digit-filter');
    const sortBy = document.getElementById('sort-by');
    const difficultyRange = document.getElementById('difficulty-range');
    const difficultyLabel = document.getElementById('difficulty-label');
    const clearBtn = document.getElementById('clear-filters');
    const activeFilter = document.getElementById('active-filter');

    function updateActiveFilterDisplay() {
        const parts = [];
        if (state.filters.modelId !== undefined) {
            parts.push(`Model ${state.filters.modelId}`);
        }
        if (state.filters.digit !== 'all') {
            parts.push(`Digit ${state.filters.digit}`);
        }
        if (state.filters.minDifficulty > 0) {
            parts.push(`Diff ≥${(state.filters.minDifficulty * 100).toFixed(0)}%`);
        }
        activeFilter.textContent = parts.length > 0 ? `Filtered: ${parts.join(', ')}` : '';
    }

    digitFilter.addEventListener('change', (e) => {
        state.filters.digit = e.target.value;
        updateActiveFilterDisplay();
        events.emit('filter-change', state.filters);
    });

    sortBy.addEventListener('change', (e) => {
        state.filters.sort = e.target.value;
        events.emit('filter-change', state.filters);
    });

    difficultyRange.addEventListener('input', (e) => {
        state.filters.minDifficulty = parseInt(e.target.value) / 100;
        difficultyLabel.textContent = `Min difficulty: ${e.target.value}%`;
        updateActiveFilterDisplay();
        events.emit('filter-change', state.filters);
    });

    clearBtn.addEventListener('click', () => {
        state.filters.modelId = undefined;
        state.filters.digit = 'all';
        state.filters.minDifficulty = 0;
        digitFilter.value = 'all';
        difficultyRange.value = 0;
        difficultyLabel.textContent = 'Min difficulty: 0%';
        updateActiveFilterDisplay();
        events.emit('filter-change', state.filters);
        events.emit('highlight-images', []);
    });

    events.on('model-filter', () => {
        updateActiveFilterDisplay();
    });
}

// Modal functionality
function setupModals() {
    // Model Overview Modal
    window.openModelOverviewModal = function() {
        const modal = document.getElementById('model-overview-modal');
        const modalBody = document.getElementById('model-overview-expanded');

        // Render expanded model overview
        const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);

        modalBody.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(10, 1fr); gap: 1rem; max-width: 1200px; margin: 0 auto;">
                ${state.data.models.map((model, i) => `
                    <div style="text-align: center; cursor: pointer; padding: 0.5rem; background: var(--bg-tertiary); border-radius: 8px; transition: transform 0.2s;"
                         onmouseover="this.style.transform='scale(1.05)'"
                         onmouseout="this.style.transform='scale(1)'"
                         onclick="window.highlightModelImages(${i}); document.getElementById('model-overview-modal').classList.remove('active');">
                        <svg width="80" height="80" style="margin-bottom: 0.5rem;">
                            <circle cx="40" cy="40" r="35" fill="${colorScale(model.test_accuracy)}" stroke="#fff" stroke-width="2"/>
                            <text x="40" y="45" text-anchor="middle" font-size="20" fill="#fff" font-weight="bold">${i}</text>
                        </svg>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">Model ${i}</div>
                        <div style="font-size: 0.85rem; font-weight: bold;">${(model.test_accuracy * 100).toFixed(1)}%</div>
                        <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.25rem;">${model.training_indices.length} samples</div>
                    </div>
                `).join('')}
            </div>
        `;

        modal.classList.add('active');
    };

    // Training Curves Modal
    window.openTrainingCurvesModal = function() {
        const modal = document.getElementById('training-curves-modal');
        const modalBody = document.getElementById('training-curves-expanded');

        // Render large training curves grid
        modalBody.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1.5rem;">
                ${state.data.models.map((model, i) => {
                    const checkpoints = model.checkpoints;
                    const maxEpoch = Math.max(...checkpoints.map(c => c.epoch));
                    const width = 350;
                    const height = 200;
                    const margin = {top: 20, right: 20, bottom: 30, left: 40};
                    const chartWidth = width - margin.left - margin.right;
                    const chartHeight = height - margin.top - margin.bottom;

                    // Create SVG paths for train and eval accuracy
                    const xScale = (epoch) => margin.left + (epoch / maxEpoch) * chartWidth;
                    const yScale = (acc) => margin.top + (1 - acc) * chartHeight;

                    const trainPath = checkpoints.map((c, i) =>
                        `${i === 0 ? 'M' : 'L'} ${xScale(c.epoch)} ${yScale(c.train_acc)}`
                    ).join(' ');

                    const evalPath = checkpoints.map((c, i) =>
                        `${i === 0 ? 'M' : 'L'} ${xScale(c.epoch)} ${yScale(c.eval_acc)}`
                    ).join(' ');

                    return `
                        <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 8px; cursor: pointer;"
                             onclick="window.highlightModelImages(${i}); document.getElementById('training-curves-modal').classList.remove('active');">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-weight: bold;">Model ${i}</span>
                                <span style="font-size: 0.85rem; color: var(--success);">${(model.test_accuracy * 100).toFixed(1)}%</span>
                            </div>
                            <svg width="${width}" height="${height}">
                                <!-- Grid lines -->
                                ${[0, 0.25, 0.5, 0.75, 1].map(y =>
                                    `<line x1="${margin.left}" y1="${yScale(y)}" x2="${width - margin.right}" y2="${yScale(y)}"
                                           stroke="#333" stroke-dasharray="2,2"/>`
                                ).join('')}
                                <!-- Axes -->
                                <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"
                                      stroke="#666" stroke-width="1"/>
                                <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}"
                                      stroke="#666" stroke-width="1"/>
                                <!-- Train accuracy line -->
                                <path d="${trainPath}" stroke="#4ade80" stroke-width="2" fill="none"/>
                                <!-- Eval accuracy line -->
                                <path d="${evalPath}" stroke="#60a5fa" stroke-width="2" fill="none"/>
                                <!-- Labels -->
                                <text x="${margin.left - 5}" y="${margin.top - 5}" text-anchor="end" fill="#aaa" font-size="10">100%</text>
                                <text x="${margin.left - 5}" y="${height - margin.bottom + 3}" text-anchor="end" fill="#aaa" font-size="10">0%</text>
                                <text x="${width - margin.right}" y="${height - margin.bottom + 15}" text-anchor="end" fill="#aaa" font-size="10">Epoch ${maxEpoch}</text>
                            </svg>
                            <div style="display: flex; gap: 1rem; font-size: 0.75rem; margin-top: 0.5rem;">
                                <span><span style="display: inline-block; width: 12px; height: 3px; background: #4ade80; margin-right: 4px;"></span>Train</span>
                                <span><span style="display: inline-block; width: 12px; height: 3px; background: #60a5fa; margin-right: 4px;"></span>Eval</span>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;

        modal.classList.add('active');
    };

    // Close modals
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.onclick = function() {
            this.closest('.modal').classList.remove('active');
        };
    });

    // Close on background click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.onclick = function(e) {
            if (e.target === this) {
                this.classList.remove('active');
            }
        };
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.active').forEach(modal => {
                modal.classList.remove('active');
            });
        }
    });
}

// Export for use by components
export { state, events, tooltip };

// Start app
init();
