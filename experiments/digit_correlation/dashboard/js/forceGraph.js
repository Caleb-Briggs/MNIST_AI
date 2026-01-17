export function createForceGraph(container, state, events, tooltip) {
    const el = document.querySelector(container);

    let svg, width, height;
    let simulation;
    let nodes = [];
    let links = [];
    let selectedModel = null;

    function init() {
        const rect = el.getBoundingClientRect();
        width = rect.width;
        height = rect.height;

        // Add expand button
        const expandButton = document.createElement('button');
        expandButton.className = 'expand-button';
        expandButton.textContent = 'Expand ⤢';
        expandButton.onclick = () => window.openModelOverviewModal();
        expandButton.style.position = 'absolute';
        expandButton.style.top = '0.5rem';
        expandButton.style.right = '0.5rem';
        expandButton.style.zIndex = '100';
        el.style.position = 'relative';
        el.appendChild(expandButton);

        svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Add zoom behavior
        const g = svg.append('g');
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        svg.call(zoom);

        // Create simulation
        simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(30))
            .force('charge', d3.forceManyBody().strength(-50))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(15));

        renderAllModels();

        // Handle resize
        window.addEventListener('resize', debounce(() => {
            const rect = el.getBoundingClientRect();
            width = rect.width;
            height = rect.height;
            svg.attr('width', width).attr('height', height);
            simulation.force('center', d3.forceCenter(width / 2, height / 2));
            simulation.alpha(0.3).restart();
        }, 250));
    }

    function renderAllModels() {
        // Show all 50 models as clustered nodes
        nodes = state.data.models.map((model, idx) => ({
            id: `model-${idx}`,
            modelIdx: idx,
            type: 'model',
            accuracy: model.test_accuracy,
            trainingIndices: model.training_indices
        }));

        links = [];

        render();
    }

    function renderModelDetail(modelIdx) {
        selectedModel = modelIdx;
        const model = state.data.models[modelIdx];

        // Create nodes for each training image
        nodes = model.training_indices.map(imgIdx => {
            const img = state.data.images[imgIdx];
            if (!img) {
                console.warn('Image not found:', imgIdx);
                return null;
            }
            return {
                id: `img-${imgIdx}`,
                imageIdx: imgIdx,
                type: 'image',
                digit: img.digit,
                difficulty: img.difficulty,
                image: img.image
            };
        }).filter(n => n !== null);

        // Group by digit for better layout
        nodes.forEach(node => {
            node.fx = undefined;
            node.fy = undefined;
        });

        // Create links between images of the same digit
        links = [];
        const digitGroups = {};
        nodes.forEach(node => {
            if (!digitGroups[node.digit]) digitGroups[node.digit] = [];
            digitGroups[node.digit].push(node);
        });

        Object.values(digitGroups).forEach(group => {
            for (let i = 0; i < group.length - 1; i++) {
                links.push({
                    source: group[i].id,
                    target: group[i + 1].id
                });
            }
        });

        render();
    }

    function render() {
        const g = svg.select('g');
        g.selectAll('*').remove();

        if (selectedModel === null) {
            // Model overview - simple grid layout
            const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
                .domain([0, 1]);

            nodes.forEach((d, i) => {
                const row = Math.floor(i / 10);
                const col = i % 10;
                d.x = col * 60 + 30;
                d.y = row * 60 + 30;
            });

            const node = g.selectAll('g')
                .data(nodes)
                .join('g')
                .attr('transform', d => `translate(${d.x},${d.y})`);

            node.append('circle')
                .attr('r', 20)
                .attr('fill', d => colorScale(d.accuracy))
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .style('cursor', 'pointer');

            node.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', 5)
                .attr('font-size', '12px')
                .attr('fill', '#fff')
                .attr('font-weight', 'bold')
                .style('pointer-events', 'none')
                .text(d => d.modelIdx);

            // Click handlers
            node.on('click', (event, d) => {
                console.log('Clicked model:', d.modelIdx);
                handleNodeClick(event, d);
            })
            .on('mouseover', (event, d) => handleNodeHover(event, d))
            .on('mouseout', () => tooltip.hide());

        } else {
            // Image detail - simple grid
            const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
                .domain([0, 1]);

            nodes.forEach((d, i) => {
                const row = Math.floor(i / 4);
                const col = i % 4;
                d.x = col * 80 + 40;
                d.y = row * 80 + 40;
            });

            const node = g.selectAll('g')
                .data(nodes)
                .join('g')
                .attr('transform', d => `translate(${d.x},${d.y})`);

            node.append('circle')
                .attr('r', 30)
                .attr('fill', d => colorScale(d.difficulty))
                .attr('stroke', d => d3.schemeTableau10[d.digit])
                .attr('stroke-width', 3);

            node.append('image')
                .attr('xlink:href', d => d.image)
                .attr('x', -25)
                .attr('y', -25)
                .attr('width', 50)
                .attr('height', 50)
                .style('pointer-events', 'none');

            node.on('click', (event, d) => {
                console.log('Clicked image:', d.imageIdx);
                handleNodeClick(event, d);
            })
            .on('mouseover', (event, d) => handleNodeHover(event, d))
            .on('mouseout', () => tooltip.hide());
        }
    }

    function handleNodeClick(event, d) {
        event.stopPropagation();

        if (d.type === 'model') {
            // Zoom into model details
            renderModelDetail(d.modelIdx);
            updateBreadcrumb(`Model ${d.modelIdx}`);
        } else if (d.type === 'image') {
            // Select image
            state.selectedImage = d.imageIdx;
            events.emit('image-select', d.imageIdx);
        }
    }

    function handleNodeHover(event, d) {
        let html = '';

        if (d.type === 'model') {
            html = `
                <div><strong>Model ${d.modelIdx}</strong></div>
                <div>Test accuracy: ${(d.accuracy * 100).toFixed(1)}%</div>
                <div>Training samples: ${d.trainingIndices.length}</div>
                <div style="font-size: 0.75em; margin-top: 4px;">Click to explore</div>
            `;
        } else if (d.type === 'image') {
            html = `
                <img src="${d.image}" style="width: 60px;">
                <div>Digit: ${d.digit}</div>
                <div>Difficulty: ${(d.difficulty * 100).toFixed(1)}%</div>
                <div>Index: ${d.imageIdx}</div>
            `;
        }

        tooltip.show(html, event.pageX, event.pageY);
    }

    function updateBreadcrumb(path) {
        const breadcrumb = document.getElementById('treemap-breadcrumb');
        if (selectedModel === null) {
            breadcrumb.innerHTML = '<span onclick="window.forceGraphZoomOut()">All Models</span>';
        } else {
            breadcrumb.innerHTML = `
                <span onclick="window.forceGraphZoomOut()">All Models</span> >
                <span>${path}</span>
            `;
        }
    }

    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }

    // Global function for breadcrumb navigation
    window.forceGraphZoomOut = function() {
        selectedModel = null;
        renderAllModels();
        updateBreadcrumb('All Models');
    };

    init();
    updateBreadcrumb('All Models');

    return { renderModelDetail, renderAllModels };
}

function debounce(fn, ms) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn.apply(this, args), ms);
    };
}
