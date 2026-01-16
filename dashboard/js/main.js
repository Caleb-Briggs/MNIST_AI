import { createTreemap } from './treemap.js';
import { createCorrelationMatrix } from './correlation.js';
import { createImageGrid } from './imageGrid.js';
import { createDetailsPanel } from './details.js';

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
            `${state.data.config.num_models} models | ${state.data.test_images.length} test images | ${state.data.config.samples_per_digit} samples/digit`;

        // Initialize components
        createTreemap('#treemap', state, events, tooltip);
        createCorrelationMatrix('#correlation', state, events, tooltip);
        createImageGrid('#image-grid', state, events, tooltip);
        createDetailsPanel('#details', state, events);

        // Setup filters
        setupFilters();

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

    digitFilter.addEventListener('change', (e) => {
        state.filters.digit = e.target.value;
        events.emit('filter-change', state.filters);
    });

    sortBy.addEventListener('change', (e) => {
        state.filters.sort = e.target.value;
        events.emit('filter-change', state.filters);
    });

    difficultyRange.addEventListener('input', (e) => {
        state.filters.minDifficulty = parseInt(e.target.value) / 100;
        difficultyLabel.textContent = `Min difficulty: ${e.target.value}%`;
        events.emit('filter-change', state.filters);
    });
}

// Export for use by components
export { state, events, tooltip };

// Start app
init();
