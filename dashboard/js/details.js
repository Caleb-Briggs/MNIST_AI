export function createDetailsPanel(container, state, events) {
    const el = document.querySelector(container);
    const modelStatsEl = document.getElementById('model-stats');

    function init() {
        events.on('image-select', (imgIdx) => {
            renderImageDetails(imgIdx);
        });

        renderModelStats();
    }

    function renderImageDetails(imgIdx) {
        if (imgIdx === null || imgIdx === undefined) {
            el.innerHTML = '<p class="placeholder">Click an image to see details</p>';
            return;
        }

        const img = state.data.test_images[imgIdx];
        if (!img) {
            el.innerHTML = '<p class="placeholder">Image not found</p>';
            return;
        }

        const correctCount = img.correct_by.length;
        const totalModels = state.data.config.num_models;
        const wrongBy = [];

        // Find which models got it wrong
        for (let i = 0; i < totalModels; i++) {
            if (!img.correct_by.includes(i)) {
                wrongBy.push(i);
            }
        }

        // Find similar images (same digit, similar difficulty)
        const similar = state.data.test_images
            .filter(other => other.id !== img.id && other.digit === img.digit)
            .sort((a, b) => Math.abs(a.difficulty - img.difficulty) - Math.abs(b.difficulty - img.difficulty))
            .slice(0, 8);

        const accuracyColor = img.difficulty > 0.8 ? 'var(--success)' :
                             img.difficulty > 0.5 ? 'var(--warning)' : 'var(--error)';

        el.innerHTML = `
            <img class="detail-image" src="${img.image}" alt="Digit ${img.digit}">

            <dl class="detail-stats">
                <dt>Digit</dt>
                <dd>${img.digit}</dd>

                <dt>Index</dt>
                <dd>${img.id}</dd>

                <dt>Accuracy (${correctCount}/${totalModels} models)</dt>
                <dd>
                    <div class="accuracy-bar">
                        <div class="accuracy-bar-fill" style="width: ${img.difficulty * 100}%; background: ${accuracyColor}"></div>
                    </div>
                    ${(img.difficulty * 100).toFixed(1)}%
                </dd>

                <dt>Correct by (${correctCount})</dt>
                <dd>
                    <div class="model-list">
                        ${img.correct_by.slice(0, 20).map(m => `<span class="model-chip correct">${m}</span>`).join('')}
                        ${img.correct_by.length > 20 ? `<span class="model-chip">+${img.correct_by.length - 20} more</span>` : ''}
                    </div>
                </dd>

                <dt>Wrong by (${wrongBy.length})</dt>
                <dd>
                    <div class="model-list">
                        ${wrongBy.slice(0, 20).map(m => `<span class="model-chip wrong">${m}</span>`).join('')}
                        ${wrongBy.length > 20 ? `<span class="model-chip">+${wrongBy.length - 20} more</span>` : ''}
                    </div>
                </dd>

                <dt>Similar images</dt>
                <dd>
                    <div class="similar-images" style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;">
                        ${similar.map(s => `
                            <img src="${s.image}"
                                 style="width: 32px; height: 32px; cursor: pointer; border-radius: 2px;"
                                 title="Index: ${s.id}, Acc: ${(s.difficulty * 100).toFixed(0)}%"
                                 onclick="window.selectImage(${s.id})">
                        `).join('')}
                    </div>
                </dd>
            </dl>
        `;

        // Highlight related images in grid
        events.emit('highlight-images', [imgIdx, ...similar.map(s => s.id)]);
    }

    // Global function for clicking similar images
    window.selectImage = function(imgIdx) {
        state.selectedImage = imgIdx;
        events.emit('image-select', imgIdx);
    };

    function renderModelStats() {
        const models = state.data.models;

        // Sort by accuracy
        const sortedModels = [...models].sort((a, b) => b.test_accuracy - a.test_accuracy);

        modelStatsEl.innerHTML = `
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                Sorted by test accuracy
            </div>
            ${sortedModels.slice(0, 20).map(m => `
                <div class="model-stat-row" onclick="window.highlightModelImages(${m.id})" style="cursor: pointer;">
                    <span>Model ${m.id}</span>
                    <span style="color: ${m.test_accuracy > 0.5 ? 'var(--success)' : m.test_accuracy > 0.3 ? 'var(--warning)' : 'var(--error)'}">
                        ${(m.test_accuracy * 100).toFixed(1)}%
                    </span>
                </div>
            `).join('')}
            ${models.length > 20 ? `<div style="text-align: center; color: #888; font-size: 0.75rem; padding: 0.5rem;">+${models.length - 20} more models</div>` : ''}
        `;
    }

    // Global function to highlight a model's training images
    window.highlightModelImages = function(modelId) {
        const model = state.data.models[modelId];
        if (model) {
            events.emit('highlight-images', model.training_indices);
        }
    };

    init();

    return { renderImageDetails, renderModelStats };
}
