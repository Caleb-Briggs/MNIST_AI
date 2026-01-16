export function createTreemap(container, state, events, tooltip) {
    const el = document.querySelector(container);
    const breadcrumb = document.getElementById('treemap-breadcrumb');

    let svg, width, height;
    let currentRoot = null;
    let currentView = null;

    function init() {
        const rect = el.getBoundingClientRect();
        width = rect.width;
        height = rect.height;

        svg = d3.select(container)
            .append('svg')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .style('font', '10px sans-serif');

        buildHierarchy();
        render(currentRoot);

        // Handle resize
        window.addEventListener('resize', debounce(() => {
            const rect = el.getBoundingClientRect();
            width = rect.width;
            height = rect.height;
            svg.attr('viewBox', `0 0 ${width} ${height}`);
            render(currentView || currentRoot);
        }, 250));
    }

    function buildHierarchy() {
        // Build hierarchy: Root -> Models -> Digits -> Images
        const data = state.data;

        const children = data.models.map((model, modelIdx) => {
            // Group training images by digit
            const digitGroups = {};
            model.training_indices.forEach(imgIdx => {
                const img = data.test_images[imgIdx];
                if (!img) return;
                const digit = img.digit;
                if (!digitGroups[digit]) digitGroups[digit] = [];
                digitGroups[digit].push({
                    name: `img_${imgIdx}`,
                    imageIdx: imgIdx,
                    image: img.image,
                    difficulty: img.difficulty,
                    digit: digit,
                    value: 1
                });
            });

            return {
                name: `Model ${modelIdx}`,
                modelIdx: modelIdx,
                accuracy: model.test_accuracy,
                children: Object.entries(digitGroups).map(([digit, images]) => ({
                    name: `Digit ${digit}`,
                    digit: parseInt(digit),
                    modelIdx: modelIdx,
                    children: images
                }))
            };
        });

        const hierarchyData = {
            name: 'All Models',
            children: children
        };

        currentRoot = d3.hierarchy(hierarchyData)
            .sum(d => d.value || 0)
            .sort((a, b) => b.value - a.value);

        currentView = currentRoot;
    }

    function render(root) {
        currentView = root;

        const treemap = d3.treemap()
            .size([width, height])
            .paddingOuter(3)
            .paddingTop(19)
            .paddingInner(1)
            .round(true);

        treemap(root);

        svg.selectAll('*').remove();

        const nodes = root.descendants();

        // Color scale based on accuracy/difficulty
        const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
            .domain([0, 1]);

        // Render cells
        const cell = svg.selectAll('g')
            .data(nodes.filter(d => d.depth > 0))
            .join('g')
            .attr('transform', d => `translate(${d.x0},${d.y0})`);

        cell.append('rect')
            .attr('class', 'treemap-cell')
            .attr('width', d => Math.max(0, d.x1 - d.x0))
            .attr('height', d => Math.max(0, d.y1 - d.y0))
            .attr('fill', d => {
                if (d.data.imageIdx !== undefined) {
                    return colorScale(d.data.difficulty);
                } else if (d.data.accuracy !== undefined) {
                    return colorScale(d.data.accuracy);
                } else if (d.data.digit !== undefined) {
                    return d3.schemeTableau10[d.data.digit];
                }
                return '#666';
            })
            .on('click', (event, d) => handleClick(event, d))
            .on('mouseover', (event, d) => handleMouseOver(event, d))
            .on('mouseout', () => tooltip.hide());

        // Add images for leaf nodes
        cell.filter(d => d.data.imageIdx !== undefined && d.data.image)
            .append('image')
            .attr('class', 'treemap-image')
            .attr('xlink:href', d => d.data.image)
            .attr('width', d => Math.max(0, d.x1 - d.x0))
            .attr('height', d => Math.max(0, d.y1 - d.y0))
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .on('click', (event, d) => handleClick(event, d))
            .on('mouseover', (event, d) => handleMouseOver(event, d))
            .on('mouseout', () => tooltip.hide());

        // Add labels for non-leaf nodes
        cell.filter(d => d.depth === 1 && (d.x1 - d.x0) > 40)
            .append('text')
            .attr('class', 'treemap-label')
            .attr('x', d => (d.x1 - d.x0) / 2)
            .attr('y', 12)
            .text(d => d.data.name);

        updateBreadcrumb();
    }

    function handleClick(event, d) {
        event.stopPropagation();

        if (d.data.imageIdx !== undefined) {
            // Clicked on an image - select it
            state.selectedImage = d.data.imageIdx;
            events.emit('image-select', d.data.imageIdx);
        } else if (d.children) {
            // Zoom into this node
            render(d);
        }
    }

    function handleMouseOver(event, d) {
        let html = '';

        if (d.data.imageIdx !== undefined) {
            const img = state.data.test_images[d.data.imageIdx];
            html = `
                <img src="${d.data.image}" alt="digit">
                <div>Digit: ${d.data.digit}</div>
                <div>Difficulty: ${(d.data.difficulty * 100).toFixed(1)}%</div>
                <div>Index: ${d.data.imageIdx}</div>
            `;
        } else if (d.data.modelIdx !== undefined && d.data.digit === undefined) {
            html = `
                <div><strong>${d.data.name}</strong></div>
                <div>Test accuracy: ${(d.data.accuracy * 100).toFixed(1)}%</div>
                <div>Training samples: ${d.value}</div>
            `;
        } else if (d.data.digit !== undefined) {
            html = `
                <div><strong>Digit ${d.data.digit}</strong></div>
                <div>Samples: ${d.value}</div>
            `;
        }

        if (html) {
            tooltip.show(html, event.pageX, event.pageY);
        }
    }

    function updateBreadcrumb() {
        const path = [];
        let node = currentView;
        while (node) {
            path.unshift(node);
            node = node.parent;
        }

        breadcrumb.innerHTML = path.map((n, i) => {
            const name = n.data.name || 'Root';
            if (i === path.length - 1) {
                return `<span>${name}</span>`;
            }
            return `<span onclick="window.treemapZoomTo(${i})">${name}</span> > `;
        }).join('');
    }

    // Global function for breadcrumb navigation
    window.treemapZoomTo = function(index) {
        let node = currentView;
        const path = [];
        while (node) {
            path.unshift(node);
            node = node.parent;
        }
        if (path[index]) {
            render(path[index]);
        }
    };

    // Listen for events
    events.on('image-select', (imgIdx) => {
        // Could highlight the image in treemap
    });

    events.on('digit-filter', (digits) => {
        // Could filter treemap to show only certain digits
    });

    init();

    return { render, zoomTo: (node) => render(node) };
}

function debounce(fn, ms) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn.apply(this, args), ms);
    };
}
