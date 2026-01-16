export function createImageGrid(container, state, events, tooltip) {
    const el = document.querySelector(container);

    let filteredImages = [];
    let digitFilter = null;

    function init() {
        render();

        // Listen for filter changes
        events.on('filter-change', (filters) => {
            render();
        });

        events.on('digit-filter', (digits) => {
            digitFilter = digits;
            render();
        });

        events.on('image-select', (imgIdx) => {
            updateSelection(imgIdx);
        });
    }

    function getFilteredAndSortedImages() {
        let images = [...state.data.test_images];

        // Apply digit filter from correlation matrix
        if (digitFilter) {
            images = images.filter(img => digitFilter.includes(img.digit));
        }

        // Apply dropdown digit filter
        if (state.filters.digit !== 'all') {
            const digit = parseInt(state.filters.digit);
            images = images.filter(img => img.digit === digit);
        }

        // Apply difficulty filter
        if (state.filters.minDifficulty > 0) {
            images = images.filter(img => img.difficulty >= state.filters.minDifficulty);
        }

        // Sort
        switch (state.filters.sort) {
            case 'difficulty-asc':
                images.sort((a, b) => a.difficulty - b.difficulty);
                break;
            case 'difficulty-desc':
                images.sort((a, b) => b.difficulty - a.difficulty);
                break;
            case 'digit':
                images.sort((a, b) => a.digit - b.digit || a.id - b.id);
                break;
            case 'id':
                images.sort((a, b) => a.id - b.id);
                break;
        }

        return images;
    }

    function render() {
        filteredImages = getFilteredAndSortedImages();

        el.innerHTML = '';

        // Limit to avoid performance issues
        const displayImages = filteredImages.slice(0, 2000);

        displayImages.forEach(img => {
            const imgEl = document.createElement('img');
            imgEl.className = 'grid-image';
            imgEl.src = img.image;
            imgEl.alt = `Digit ${img.digit}`;
            imgEl.dataset.id = img.id;

            if (state.selectedImage === img.id) {
                imgEl.classList.add('selected');
            }

            if (state.highlightedImages.has(img.id)) {
                imgEl.classList.add('highlighted');
            }

            imgEl.addEventListener('click', () => handleClick(img));
            imgEl.addEventListener('mouseover', (e) => handleMouseOver(e, img));
            imgEl.addEventListener('mouseout', () => tooltip.hide());

            el.appendChild(imgEl);
        });

        if (filteredImages.length > 2000) {
            const note = document.createElement('div');
            note.style.cssText = 'grid-column: 1/-1; text-align: center; color: #888; padding: 1rem;';
            note.textContent = `Showing 2000 of ${filteredImages.length} images. Use filters to narrow down.`;
            el.appendChild(note);
        }
    }

    function handleClick(img) {
        state.selectedImage = img.id;
        updateSelection(img.id);
        events.emit('image-select', img.id);
    }

    function updateSelection(imgIdx) {
        el.querySelectorAll('.grid-image').forEach(imgEl => {
            const id = parseInt(imgEl.dataset.id);
            imgEl.classList.toggle('selected', id === imgIdx);
        });
    }

    function handleMouseOver(event, img) {
        const correctCount = img.correct_by.length;
        const totalModels = state.data.config.num_models;

        const html = `
            <img src="${img.image}" alt="digit">
            <div><strong>Digit ${img.digit}</strong></div>
            <div>Index: ${img.id}</div>
            <div>Accuracy: ${(img.difficulty * 100).toFixed(1)}%</div>
            <div>${correctCount}/${totalModels} models correct</div>
        `;
        tooltip.show(html, event.pageX, event.pageY);
    }

    // Highlight specific images
    function highlightImages(imageIds) {
        state.highlightedImages = new Set(imageIds);
        el.querySelectorAll('.grid-image').forEach(imgEl => {
            const id = parseInt(imgEl.dataset.id);
            imgEl.classList.toggle('highlighted', state.highlightedImages.has(id));
            imgEl.classList.toggle('dimmed', imageIds.length > 0 && !state.highlightedImages.has(id));
        });
    }

    events.on('highlight-images', highlightImages);

    init();

    return { render, highlightImages };
}
