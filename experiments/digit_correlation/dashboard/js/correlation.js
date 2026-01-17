export function createCorrelationMatrix(container, state, events, tooltip) {
    const el = document.querySelector(container);

    const margin = { top: 30, right: 10, bottom: 10, left: 30 };
    const size = 220;
    const cellSize = (size - margin.left - margin.right) / 10;

    let svg;
    let selectedCell = null;

    function init() {
        // Add expand button to section
        const section = el.closest('section');
        const sectionHeader = section.querySelector('h2');
        if (sectionHeader && !document.getElementById('correlation-expand-btn')) {
            const expandButton = document.createElement('button');
            expandButton.id = 'correlation-expand-btn';
            expandButton.className = 'expand-button';
            expandButton.textContent = 'Expand ⤢';
            expandButton.onclick = () => window.openImageCorrelationModal();
            expandButton.style.marginLeft = '0.5rem';
            expandButton.style.fontSize = '0.7rem';
            sectionHeader.appendChild(expandButton);
        }

        svg = d3.select(container)
            .append('svg')
            .attr('width', size)
            .attr('height', size);

        render();
    }

    function render() {
        const data = state.data.digit_correlation;

        svg.selectAll('*').remove();

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Color scale: diverging from negative (blue) to positive (red)
        const colorScale = d3.scaleSequential(d3.interpolateRdBu)
            .domain([1, -1]); // Reversed so red = positive correlation

        // Create cells
        for (let i = 0; i < 10; i++) {
            for (let j = 0; j < 10; j++) {
                const value = data[i][j];

                g.append('rect')
                    .attr('class', 'correlation-cell')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize - 1)
                    .attr('height', cellSize - 1)
                    .attr('fill', colorScale(value))
                    .attr('data-row', i)
                    .attr('data-col', j)
                    .on('click', (event) => handleClick(event, i, j, value))
                    .on('mouseover', (event) => handleMouseOver(event, i, j, value))
                    .on('mouseout', () => tooltip.hide());

                // Add value text for larger cells
                if (cellSize > 15) {
                    g.append('text')
                        .attr('class', 'correlation-value')
                        .attr('x', j * cellSize + cellSize / 2)
                        .attr('y', i * cellSize + cellSize / 2 + 3)
                        .text(value.toFixed(2));
                }
            }
        }

        // Row labels (digits 0-9)
        for (let i = 0; i < 10; i++) {
            g.append('text')
                .attr('class', 'correlation-label')
                .attr('x', -8)
                .attr('y', i * cellSize + cellSize / 2 + 3)
                .text(i);
        }

        // Column labels
        for (let j = 0; j < 10; j++) {
            g.append('text')
                .attr('class', 'correlation-label')
                .attr('x', j * cellSize + cellSize / 2)
                .attr('y', -8)
                .text(j);
        }

        // Title
        svg.append('text')
            .attr('x', size / 2)
            .attr('y', 15)
            .attr('text-anchor', 'middle')
            .attr('fill', '#aaa')
            .attr('font-size', '11px')
            .text('Digit Accuracy Correlation');
    }

    function handleClick(event, row, col, value) {
        event.stopPropagation();

        // Toggle selection
        if (selectedCell && selectedCell.row === row && selectedCell.col === col) {
            selectedCell = null;
            events.emit('digit-filter', null);
        } else {
            selectedCell = { row, col, value };
            events.emit('digit-filter', [row, col]);
        }

        updateSelection();
    }

    function updateSelection() {
        svg.selectAll('.correlation-cell')
            .attr('stroke', function() {
                if (!selectedCell) return null;
                const row = +this.getAttribute('data-row');
                const col = +this.getAttribute('data-col');
                if (row === selectedCell.row && col === selectedCell.col) {
                    return 'white';
                }
                return null;
            })
            .attr('stroke-width', function() {
                if (!selectedCell) return null;
                const row = +this.getAttribute('data-row');
                const col = +this.getAttribute('data-col');
                if (row === selectedCell.row && col === selectedCell.col) {
                    return 2;
                }
                return null;
            });
    }

    function handleMouseOver(event, row, col, value) {
        const html = `
            <div><strong>Digits ${row} vs ${col}</strong></div>
            <div>Correlation: ${value.toFixed(3)}</div>
            <div style="font-size: 0.75em; color: #aaa; margin-top: 4px;">
                ${value > 0.5 ? 'Strong positive' :
                  value > 0.2 ? 'Moderate positive' :
                  value > -0.2 ? 'Weak/no correlation' :
                  value > -0.5 ? 'Moderate negative' : 'Strong negative'}
            </div>
            <div style="font-size: 0.75em; color: #888; margin-top: 2px;">
                Click to filter images
            </div>
        `;
        tooltip.show(html, event.pageX, event.pageY);
    }

    // Listen for external events
    events.on('image-select', (imgIdx) => {
        // Could highlight relevant correlations
    });

    init();

    return { render };
}
