export function createTrainingCurves(container, state, events) {
    const el = document.querySelector(container);
    const modelSelector = document.getElementById('model-selector');
    const checkpointSlider = document.getElementById('checkpoint-slider');
    const checkpointInfo = document.getElementById('checkpoint-info');

    let svg, width, height;
    let selectedModelIdx = null;

    function init() {
        const rect = el.getBoundingClientRect();
        width = rect.width;
        height = rect.height;

        // Add expand button to section header
        const sectionHeader = document.querySelector('#training-curves-section .section-header');
        const expandButton = document.createElement('button');
        expandButton.className = 'expand-button';
        expandButton.textContent = 'Expand ⤢';
        expandButton.onclick = () => window.openTrainingCurvesModal();
        expandButton.style.marginLeft = '0.5rem';
        sectionHeader.appendChild(expandButton);

        svg = d3.select(container)
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Populate model selector
        state.data.models.forEach((model, idx) => {
            const option = document.createElement('option');
            option.value = idx;
            option.textContent = `Model ${idx} (${(model.test_accuracy * 100).toFixed(1)}%)`;
            modelSelector.appendChild(option);
        });

        modelSelector.addEventListener('change', (e) => {
            if (e.target.value === '') {
                selectedModelIdx = null;
                render();
                checkpointSlider.disabled = true;
                checkpointSlider.value = 0;
                checkpointInfo.textContent = 'Select a model to explore checkpoints';
            } else {
                selectedModelIdx = parseInt(e.target.value);
                render();
                setupCheckpointSlider();
            }
        });

        checkpointSlider.addEventListener('input', (e) => {
            if (selectedModelIdx !== null) {
                const ckptIdx = parseInt(e.target.value);
                updateCheckpointView(ckptIdx);
            }
        });

        // Handle resize
        window.addEventListener('resize', debounce(() => {
            const rect = el.getBoundingClientRect();
            width = rect.width;
            height = rect.height;
            svg.attr('width', width).attr('height', height);
            if (selectedModelIdx !== null) render();
        }, 250));
    }

    function setupCheckpointSlider() {
        if (selectedModelIdx === null) return;

        const model = state.data.models[selectedModelIdx];
        const checkpoints = model.checkpoints;

        checkpointSlider.disabled = false;
        checkpointSlider.max = checkpoints.length - 1;
        checkpointSlider.value = checkpoints.length - 1;

        updateCheckpointView(checkpoints.length - 1);
    }

    function updateCheckpointView(ckptIdx) {
        const model = state.data.models[selectedModelIdx];
        const ckpt = model.checkpoints[ckptIdx];

        checkpointInfo.innerHTML = `
            <div>Epoch ${ckpt.epoch}: Train ${(ckpt.train_acc * 100).toFixed(1)}%, Test ${(ckpt.test_acc * 100).toFixed(1)}%</div>
            <div style="margin-top: 4px;">Loss: ${ckpt.loss.toFixed(4)}</div>
        `;

        // Emit event to update image highlighting based on checkpoint
        events.emit('checkpoint-change', {
            modelIdx: selectedModelIdx,
            checkpointIdx: ckptIdx,
            sampleCorrect: ckpt.sample_correct
        });
    }

    function render() {
        svg.selectAll('*').remove();

        if (selectedModelIdx === null) {
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', height / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#666')
                .attr('font-size', '12px')
                .text('Select a model to view training curves');
            return;
        }

        const model = state.data.models[selectedModelIdx];
        const checkpoints = model.checkpoints;

        const margin = { top: 20, right: 60, bottom: 30, left: 40 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, d3.max(checkpoints, d => d.epoch)])
            .range([0, plotWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([plotHeight, 0]);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${plotHeight})`)
            .call(d3.axisBottom(xScale).ticks(5))
            .attr('color', '#888');

        g.append('g')
            .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${(d * 100).toFixed(0)}%`))
            .attr('color', '#888');

        // Lines
        const trainLine = d3.line()
            .x(d => xScale(d.epoch))
            .y(d => yScale(d.train_acc));

        const testLine = d3.line()
            .x(d => xScale(d.epoch))
            .y(d => yScale(d.test_acc));

        g.append('path')
            .datum(checkpoints)
            .attr('fill', 'none')
            .attr('stroke', '#4ade80')
            .attr('stroke-width', 2)
            .attr('d', trainLine);

        g.append('path')
            .datum(checkpoints)
            .attr('fill', 'none')
            .attr('stroke', '#fbbf24')
            .attr('stroke-width', 2)
            .attr('d', testLine);

        // Points
        g.selectAll('.train-point')
            .data(checkpoints)
            .join('circle')
            .attr('class', 'train-point')
            .attr('cx', d => xScale(d.epoch))
            .attr('cy', d => yScale(d.train_acc))
            .attr('r', 3)
            .attr('fill', '#4ade80')
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => {
                d3.select(event.target).attr('r', 5);
            })
            .on('mouseout', (event, d) => {
                d3.select(event.target).attr('r', 3);
            });

        g.selectAll('.test-point')
            .data(checkpoints)
            .join('circle')
            .attr('class', 'test-point')
            .attr('cx', d => xScale(d.epoch))
            .attr('cy', d => yScale(d.test_acc))
            .attr('r', 3)
            .attr('fill', '#fbbf24')
            .style('cursor', 'pointer')
            .on('mouseover', (event, d) => {
                d3.select(event.target).attr('r', 5);
            })
            .on('mouseout', (event, d) => {
                d3.select(event.target).attr('r', 3);
            });

        // Legend
        const legend = g.append('g')
            .attr('transform', `translate(${plotWidth - 80}, 10)`);

        legend.append('line')
            .attr('x1', 0).attr('x2', 20)
            .attr('y1', 0).attr('y2', 0)
            .attr('stroke', '#4ade80')
            .attr('stroke-width', 2);
        legend.append('text')
            .attr('x', 25).attr('y', 4)
            .attr('font-size', '10px')
            .attr('fill', '#aaa')
            .text('Train');

        legend.append('line')
            .attr('x1', 0).attr('x2', 20)
            .attr('y1', 15).attr('y2', 15)
            .attr('stroke', '#fbbf24')
            .attr('stroke-width', 2);
        legend.append('text')
            .attr('x', 25).attr('y', 19)
            .attr('font-size', '10px')
            .attr('fill', '#aaa')
            .text('Test');

        // Labels
        g.append('text')
            .attr('x', plotWidth / 2)
            .attr('y', plotHeight + margin.bottom - 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('fill', '#888')
            .text('Epoch');

        g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -plotHeight / 2)
            .attr('y', -margin.left + 12)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('fill', '#888')
            .text('Accuracy');
    }

    init();

    return { render };
}

function debounce(fn, ms) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn.apply(this, args), ms);
    };
}
