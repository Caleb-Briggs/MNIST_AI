#!/usr/bin/env node

import { JSDOM } from 'jsdom';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test results
const results = {
  passed: 0,
  failed: 0,
  errors: []
};

function assert(condition, message) {
  if (condition) {
    results.passed++;
    console.log(`✓ ${message}`);
  } else {
    results.failed++;
    results.errors.push(message);
    console.error(`✗ ${message}`);
  }
}

// Generate mock data
function generateMockData() {
  const NUM_MODELS = 5;
  const NUM_IMAGES = 50;
  const mockImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  const images = [];
  for (let i = 0; i < NUM_IMAGES; i++) {
    images.push({
      id: i,
      digit: i % 10,
      difficulty: Math.random(),
      correct_by: Array.from({length: Math.floor(Math.random() * NUM_MODELS)}, () => Math.floor(Math.random() * NUM_MODELS)),
      image: mockImage
    });
  }

  const models = [];
  for (let i = 0; i < NUM_MODELS; i++) {
    const checkpoints = [{
      epoch: 10,
      train_acc: 0.95,
      eval_acc: 0.45,
      digit_accs: Object.fromEntries(Array.from({length: 10}, (_, d) => [d, Math.random()])),
      sample_correct: Array(NUM_IMAGES).fill(false).map(() => Math.random() > 0.5),
      loss: 0.5
    }];

    models.push({
      id: i,
      training_indices: Array.from({length: 20}, () => Math.floor(Math.random() * NUM_IMAGES)),
      test_accuracy: 0.4 + Math.random() * 0.2,
      checkpoints: checkpoints
    });
  }

  return {
    config: {
      num_models: NUM_MODELS,
      samples_per_digit: 2,
      target_train_acc: 0.99,
      seed: 42,
      checkpoint_freq: 10
    },
    models: models,
    images: images,
    digit_correlation: Array(10).fill(0).map(() => Array(10).fill(0).map(() => Math.random() * 2 - 1)),
    model_correlation: []
  };
}

// Mock D3
function mockD3() {
  const selection = {
    append: () => selection,
    attr: () => selection,
    style: () => selection,
    text: () => selection,
    html: () => selection,
    on: () => selection,
    call: () => selection,
    selectAll: () => selection,
    data: () => selection,
    enter: () => selection,
    exit: () => selection,
    remove: () => selection,
    merge: () => selection,
    transition: () => selection,
    duration: () => selection,
    select: () => selection,
    classed: () => selection,
    each: () => selection,
    node: () => ({ getBBox: () => ({width: 100, height: 100}) }),
    nodes: () => []
  };

  return {
    select: () => selection,
    selectAll: () => selection,
    scaleLinear: () => ({
      domain: () => ({ range: () => ({}) }),
      range: () => ({})
    }),
    scaleSequential: () => ({
      domain: () => ({})
    }),
    scaleBand: () => ({
      domain: () => ({ range: () => ({ padding: () => ({}) }) }),
      range: () => ({ padding: () => ({}) }),
      padding: () => ({}),
      bandwidth: () => 10
    }),
    axisBottom: () => {},
    axisLeft: () => {},
    max: (arr) => Math.max(...arr.map(d => typeof d === 'function' ? 0 : d)),
    extent: (arr) => [0, 1],
    line: () => {
      const l = () => '';
      l.x = () => l;
      l.y = () => l;
      return l;
    },
    interpolateRdBu: () => '#fff'
  };
}

async function runTests() {
  console.log('\n=== Dashboard Automated Tests ===\n');

  try {
    // Load HTML
    const html = readFileSync(join(__dirname, 'index.html'), 'utf-8');
    const dom = new JSDOM(html, {
      url: 'http://localhost',
      runScripts: 'outside-only',
      resources: 'usable'
    });

    const { window } = dom;
    const { document } = window;

    // Inject D3 mock
    window.d3 = mockD3();
    global.window = window;
    global.document = document;

    console.log('--- DOM Structure Tests ---\n');

    // Test 1: Required elements exist
    const requiredElements = [
      'stats', 'treemap', 'model-stats', 'correlation', 'image-grid',
      'details', 'training-curves', 'digit-filter', 'sort-by',
      'difficulty-range', 'checkpoint-slider', 'model-selector'
    ];

    requiredElements.forEach(id => {
      assert(document.getElementById(id) !== null, `Element #${id} exists`);
    });

    // Test 2: Sections have correct structure
    assert(document.querySelector('#left-panel section') !== null, 'Left panel has sections');
    assert(document.querySelector('#center-panel section') !== null, 'Center panel has sections');
    assert(document.querySelector('#right-panel section') !== null, 'Right panel has sections');

    console.log('\n--- JavaScript Module Tests ---\n');

    // Test 3: Load and validate JS modules
    const mockData = generateMockData();

    // Create state and events for modules
    const state = {
      data: mockData,
      selectedImage: null,
      highlightedImages: new Set(),
      filters: {
        digit: 'all',
        sort: 'difficulty-asc',
        minDifficulty: 0
      }
    };

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

    const tooltip = {
      el: document.getElementById('tooltip'),
      show(html, x, y) {},
      hide() {}
    };

    // Test 4: Module imports (check files exist and are valid)
    const modules = ['forceGraph.js', 'correlation.js', 'imageGrid.js', 'details.js', 'trainingCurves.js'];
    modules.forEach(module => {
      try {
        const code = readFileSync(join(__dirname, 'js', module), 'utf-8');
        assert(code.includes('export function'), `${module} exports a function`);
        assert(code.includes('state.data'), `${module} uses state.data`);
      } catch (e) {
        assert(false, `${module} is readable and valid: ${e.message}`);
      }
    });

    // Test 5: Validate details.js specifically
    const detailsCode = readFileSync(join(__dirname, 'js/details.js'), 'utf-8');
    assert(detailsCode.includes('state.data.images'), 'details.js references state.data.images (not test_images)');
    assert(detailsCode.includes("getElementById('model-stats')"), 'details.js accesses model-stats element');

    // Test 6: Validate forceGraph.js
    const forceGraphCode = readFileSync(join(__dirname, 'js/forceGraph.js'), 'utf-8');
    assert(forceGraphCode.includes('state.data.images'), 'forceGraph.js references state.data.images (not test_images)');

    // Test 7: Validate imageGrid.js
    const imageGridCode = readFileSync(join(__dirname, 'js/imageGrid.js'), 'utf-8');
    assert(imageGridCode.includes('state.data.images'), 'imageGrid.js references state.data.images (not test_images)');

    // Test 8: Validate main.js
    const mainCode = readFileSync(join(__dirname, 'js/main.js'), 'utf-8');
    assert(mainCode.includes('state.data.images.length'), 'main.js references state.data.images.length');
    assert(!mainCode.includes('test_images'), 'main.js has no references to test_images');

    console.log('\n--- Data Structure Tests ---\n');

    // Test 9: Mock data structure is valid
    assert(mockData.config !== undefined, 'Mock data has config');
    assert(mockData.models !== undefined, 'Mock data has models');
    assert(mockData.images !== undefined, 'Mock data has images (not test_images)');
    assert(Array.isArray(mockData.models), 'Models is an array');
    assert(Array.isArray(mockData.images), 'Images is an array');
    assert(mockData.models.length > 0, 'Has models');
    assert(mockData.images.length > 0, 'Has images');

    // Test 10: Model structure
    const model = mockData.models[0];
    assert(model.id !== undefined, 'Model has id');
    assert(Array.isArray(model.training_indices), 'Model has training_indices array');
    assert(Array.isArray(model.checkpoints), 'Model has checkpoints array');
    assert(model.test_accuracy !== undefined, 'Model has test_accuracy');

    // Test 11: Checkpoint structure
    const checkpoint = model.checkpoints[0];
    assert(checkpoint.epoch !== undefined, 'Checkpoint has epoch');
    assert(checkpoint.train_acc !== undefined, 'Checkpoint has train_acc');
    assert(checkpoint.eval_acc !== undefined, 'Checkpoint has eval_acc');
    assert(Array.isArray(checkpoint.sample_correct), 'Checkpoint has sample_correct array');

    // Test 12: Image structure
    const image = mockData.images[0];
    assert(image.id !== undefined, 'Image has id');
    assert(image.digit !== undefined, 'Image has digit');
    assert(image.difficulty !== undefined, 'Image has difficulty');
    assert(Array.isArray(image.correct_by), 'Image has correct_by array');
    assert(image.image !== undefined, 'Image has base64 data');

    // Test 13: Training indices are valid
    model.training_indices.forEach((idx, i) => {
      assert(idx >= 0 && idx < mockData.images.length,
        `Model training_indices[${i}] (${idx}) is valid for images array (${mockData.images.length})`);
    });

    console.log('\n--- CSS Tests ---\n');

    // Test 14: CSS file exists and has required classes
    const css = readFileSync(join(__dirname, 'css/style.css'), 'utf-8');
    assert(css.includes('.grid-image'), 'CSS has .grid-image class');
    assert(css.includes('.checkpoint-correct') || css.includes('checkpoint'), 'CSS has checkpoint styles');
    assert(css.includes('--success') || css.includes('--error'), 'CSS has color variables');

  } catch (e) {
    console.error('\n❌ Fatal error:', e.message);
    console.error(e.stack);
    results.failed++;
  }

  // Print summary
  console.log('\n=== Test Summary ===\n');
  console.log(`Passed: ${results.passed}`);
  console.log(`Failed: ${results.failed}`);

  if (results.failed > 0) {
    console.log('\nFailed tests:');
    results.errors.forEach(err => console.log(`  - ${err}`));
    process.exit(1);
  } else {
    console.log('\n✓ All tests passed!\n');
    process.exit(0);
  }
}

runTests();
