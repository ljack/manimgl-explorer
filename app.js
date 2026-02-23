/* ============================================================
   ManimGL Explorer — app.js
   Main application logic for the code explorer web app.
   Expects window.MANIM_DATA to be defined by data.js.
   ============================================================ */

(function () {
  'use strict';

  // ── Data reference ──────────────────────────────────────────
  const DATA = window.MANIM_DATA || { tree: [], files: {}, architectureContext: '' };

  // ── DOM references ──────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    fileTree:       $('#fileTree'),
    welcome:        $('#welcome'),
    codeView:       $('#codeView'),
    codeBody:       $('#codeBody'),
    codeBreadcrumb: $('#codeBreadcrumb'),
    codeDescription:$('#codeDescription'),
    btnAnnotations: $('#btnAnnotations'),
    searchInput:    $('#searchInput'),
    searchBar:      $('#searchBar'),
    chatPanel:      $('#chatPanel'),
    chatMessages:   $('#chatMessages'),
    chatInput:      $('#chatInput'),
    chatSuggestions:$('#chatSuggestions'),
    btnSend:        $('#btnSend'),
    btnChat:        $('#toggleChat'),
    apiKeyInput:    $('#apiKeyInput'),
  };

  // ── Lazy-loading infrastructure ─────────────────────────────
  const loadedChunks = new Set();

  function loadChunk(url) {
    if (loadedChunks.has(url)) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url;
      script.onload = () => {
        loadedChunks.add(url);
        resolve();
      };
      script.onerror = () => reject(new Error('Failed to load ' + url));
      document.head.appendChild(script);
    });
  }

  // ── App state ───────────────────────────────────────────────
  const state = {
    activeFilePath: null,
    activeFileData: null,
    annotationsVisible: false,
    chatVisible: true,
    chatHistory: [],       // { role, content } pairs for API
    isSending: false,
    searchResults: null,   // cached overlay element
    isLoading: false,      // true while a chunk is being loaded
  };

  // ── Initialization ──────────────────────────────────────────
  function init() {
    renderFileTree();
    setupSearch();
    setupChat();
    setupAnnotationsToggle();
    restoreApiKey();
  }

  // ============================================================
  //  FILE TREE
  // ============================================================

  function renderFileTree() {
    const tree = DATA.tree;
    if (!tree || !tree.length) {
      dom.fileTree.innerHTML = '<div style="padding:12px;color:var(--text-muted);">No files loaded.</div>';
      return;
    }
    const fragment = document.createDocumentFragment();
    tree.forEach((yearGroup) => {
      // Year header
      const yearEl = document.createElement('div');
      yearEl.className = 'tree-year';
      yearEl.textContent = yearGroup.label || yearGroup.year;
      fragment.appendChild(yearEl);

      // Projects under this year
      if (yearGroup.children) {
        yearGroup.children.forEach((node) => {
          fragment.appendChild(buildTreeNode(node, String(yearGroup.year)));
        });
      }
    });
    dom.fileTree.appendChild(fragment);

    // Event delegation for the whole tree
    dom.fileTree.addEventListener('click', handleTreeClick);
  }

  /**
   * Recursively build a tree DOM node.
   * node = { name, type:'folder'|'file', children?, path? }
   * pathPrefix is used to build breadcrumb paths.
   */
  function buildTreeNode(node, pathPrefix) {
    if (node.type === 'file') {
      const el = document.createElement('div');
      el.className = 'tree-file';
      el.dataset.path = node.path || (pathPrefix + '/' + node.label || node.name);
      el.innerHTML = '<span class="file-icon">&#128220;</span>' + escapeHTML(node.label || node.name);
      return el;
    }

    // Folder
    const folder = document.createElement('div');
    folder.className = 'tree-folder open'; // start expanded

    const label = document.createElement('div');
    label.className = 'tree-folder-label';
    label.innerHTML =
      '<span class="chevron">&#9654;</span>' +
      '<span>' + escapeHTML(node.label || node.name) + '</span>';
    folder.appendChild(label);

    const childContainer = document.createElement('div');
    childContainer.className = 'tree-folder-children';

    if (node.children) {
      const childPath = pathPrefix + '/' + node.label || node.name;
      node.children.forEach((child) => {
        childContainer.appendChild(buildTreeNode(child, childPath));
      });
    }
    folder.appendChild(childContainer);
    return folder;
  }

  function handleTreeClick(e) {
    // Folder toggle
    const folderLabel = e.target.closest('.tree-folder-label');
    if (folderLabel) {
      const folder = folderLabel.parentElement;
      folder.classList.toggle('open');
      return;
    }

    // File click
    const fileEl = e.target.closest('.tree-file');
    if (fileEl) {
      const path = fileEl.dataset.path;
      loadFile(path);  // async, but we don't need to await here
    }
  }

  function setActiveTreeItem(path) {
    $$('.tree-file.active').forEach((el) => el.classList.remove('active'));
    const target = dom.fileTree.querySelector(`.tree-file[data-path="${CSS.escape(path)}"]`);
    if (target) target.classList.add('active');
  }

  // ============================================================
  //  CODE VIEWER
  // ============================================================

  async function loadFile(path) {
    state.activeFilePath = path;
    state.annotationsVisible = false;
    dom.btnAnnotations.classList.remove('active');

    setActiveTreeItem(path);

    // Show code view, hide welcome
    dom.welcome.style.display = 'none';
    dom.codeView.style.display = 'flex';

    // Breadcrumb
    const parts = path.split('/').filter(Boolean);
    dom.codeBreadcrumb.innerHTML = parts
      .map((p, i) => {
        const cls = i === parts.length - 1 ? 'current' : '';
        return (i > 0 ? '<span class="sep">/</span>' : '') +
               '<span class="' + cls + '">' + escapeHTML(p) + '</span>';
      })
      .join('');

    // Check if file data is already loaded
    let fileData = DATA.files ? DATA.files[path] : null;

    // If not loaded, try lazy-loading the chunk
    if (!fileData && DATA.fileIndex && DATA.fileIndex[path]) {
      const chunkUrl = DATA.fileIndex[path];
      dom.codeDescription.textContent = '';
      dom.codeDescription.style.display = 'none';
      dom.codeBody.innerHTML =
        '<div style="padding:48px;text-align:center;color:var(--text-muted);">' +
        '<div class="loading-spinner"></div>' +
        '<p style="font-size:13px;margin-top:12px;">Loading...</p>' +
        '</div>';
      state.isLoading = true;

      try {
        await loadChunk(chunkUrl);
        fileData = DATA.files[path] || null;
      } catch (err) {
        dom.codeBody.innerHTML =
          '<div style="padding:48px;text-align:center;color:var(--text-muted);">' +
          '<p style="font-size:15px;margin-bottom:8px;">Failed to load</p>' +
          '<p style="font-size:13px;">' + escapeHTML(err.message) + '</p>' +
          '</div>';
        state.isLoading = false;
        return;
      }
      state.isLoading = false;

      // Check if user navigated away while loading
      if (state.activeFilePath !== path) return;
    }

    state.activeFileData = fileData;

    if (!fileData || !fileData.code) {
      dom.codeDescription.textContent = '';
      dom.codeDescription.style.display = 'none';
      dom.codeBody.innerHTML =
        '<div style="padding:48px;text-align:center;color:var(--text-muted);">' +
        '<p style="font-size:15px;margin-bottom:8px;">Code not loaded</p>' +
        '<p style="font-size:13px;">This file\'s source code has not been embedded in the explorer data.</p>' +
        '</div>';
      return;
    }

    // Description
    if (fileData.description) {
      dom.codeDescription.innerHTML = renderDescriptionHTML(fileData.description);
      dom.codeDescription.style.display = '';
    } else {
      dom.codeDescription.textContent = '';
      dom.codeDescription.style.display = 'none';
    }

    // Render code
    renderCode(fileData);
  }

  /**
   * Render the description text with basic inline formatting.
   * Supports **bold** and `code`.
   */
  function renderDescriptionHTML(text) {
    let html = escapeHTML(text);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    return html;
  }

  /**
   * Render the code table with line numbers, syntax highlighting,
   * and annotation markers.
   */
  function renderCode(fileData) {
    const code = fileData.code;
    const annotations = fileData.annotations || {};
    const lines = code.split('\n');

    // Build annotation lookup: lineNumber -> { id, text }
    // annotations can be: { "42": "explanation...", ... }
    // or: [ { line: 42, text: "...", id: 1 }, ... ]
    const annotationMap = buildAnnotationMap(annotations);

    const rows = [];
    for (let i = 0; i < lines.length; i++) {
      const lineNum = i + 1;
      const raw = lines[i];
      const highlighted = highlightPython(raw);
      const ann = annotationMap[lineNum];

      let markerHTML = '';
      if (ann) {
        markerHTML = '<span class="annotation-marker" data-ann-line="' + lineNum + '">' +
                     ann.id + '</span>';
      }

      rows.push(
        '<tr class="code-line" data-line="' + lineNum + '">' +
          '<td class="line-num">' + lineNum + '</td>' +
          '<td class="line-content">' + highlighted + markerHTML + '</td>' +
        '</tr>'
      );

      // Annotation block (hidden by default)
      if (ann) {
        rows.push(
          '<tr class="annotation-row" data-ann-line="' + lineNum + '">' +
            '<td></td>' +
            '<td>' +
              '<div class="annotation-block" data-ann-line="' + lineNum + '">' +
                renderAnnotationHTML(ann.text) +
              '</div>' +
            '</td>' +
          '</tr>'
        );
      }
    }

    dom.codeBody.innerHTML =
      '<table class="code-table"><tbody>' + rows.join('') + '</tbody></table>';

    // Event delegation for annotation markers
    dom.codeBody.addEventListener('click', handleCodeBodyClick);
  }

  function buildAnnotationMap(annotations) {
    const map = {};
    if (!annotations) return map;

    if (Array.isArray(annotations)) {
      annotations.forEach((ann, idx) => {
        const line = ann.line;
        map[line] = { id: ann.id || idx + 1, text: ann.text };
      });
    } else {
      // Object keyed by line number
      let idx = 1;
      for (const lineStr of Object.keys(annotations)) {
        const line = parseInt(lineStr, 10);
        if (isNaN(line)) continue;
        const val = annotations[lineStr];
        const text = typeof val === 'string' ? val : (val.text || '');
        map[line] = { id: val.id || idx, text: text };
        idx++;
      }
    }
    return map;
  }

  function renderAnnotationHTML(text) {
    let html = escapeHTML(text);
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Convert newlines
    html = html.replace(/\n/g, '<br>');
    return html;
  }

  function handleCodeBodyClick(e) {
    const marker = e.target.closest('.annotation-marker');
    if (marker) {
      const lineNum = marker.dataset.annLine;
      toggleAnnotation(lineNum);
    }
  }

  function toggleAnnotation(lineNum) {
    const block = dom.codeBody.querySelector(
      '.annotation-block[data-ann-line="' + lineNum + '"]'
    );
    if (block) {
      block.classList.toggle('visible');
    }
  }

  function setupAnnotationsToggle() {
    dom.btnAnnotations.addEventListener('click', () => {
      state.annotationsVisible = !state.annotationsVisible;
      dom.btnAnnotations.classList.toggle('active', state.annotationsVisible);

      const blocks = dom.codeBody.querySelectorAll('.annotation-block');
      blocks.forEach((block) => {
        if (state.annotationsVisible) {
          block.classList.add('visible');
        } else {
          block.classList.remove('visible');
        }
      });
    });
  }

  // ============================================================
  //  PYTHON SYNTAX HIGHLIGHTER
  // ============================================================

  const PY_KEYWORDS = new Set([
    'class', 'def', 'if', 'for', 'while', 'return', 'import', 'from',
    'with', 'as', 'in', 'not', 'and', 'or', 'is', 'lambda', 'try',
    'except', 'raise', 'yield', 'pass', 'break', 'continue', 'elif',
    'else', 'finally', 'True', 'False', 'None', 'self', 'del', 'global',
    'nonlocal', 'assert', 'async', 'await',
  ]);

  const PY_SELF = new Set(['self']);

  const MANIM_CONSTANTS = new Set([
    'BLUE', 'RED', 'GREEN', 'YELLOW', 'WHITE', 'GREY', 'UP', 'DOWN',
    'LEFT', 'RIGHT', 'OUT', 'IN', 'ORIGIN', 'TAU', 'PI',
    'FRAME_WIDTH', 'FRAME_HEIGHT', 'DEGREES', 'DEG', 'UL', 'UR', 'DL', 'DR',
    'BLUE_E', 'BLUE_A', 'BLUE_B', 'BLUE_C', 'BLUE_D',
    'GREY_A', 'GREY_B', 'GREY_C', 'GREY_D', 'GREY_E',
    'GRAY', 'GRAY_A', 'GRAY_B', 'GRAY_C', 'GRAY_D', 'GRAY_E',
    'RED_E', 'RED_A', 'RED_B', 'RED_C', 'RED_D',
    'GREEN_A', 'GREEN_B', 'GREEN_C', 'GREEN_D', 'GREEN_E',
    'YELLOW_A', 'YELLOW_B', 'YELLOW_C', 'YELLOW_D', 'YELLOW_E',
    'TEAL', 'TEAL_A', 'TEAL_B', 'TEAL_C', 'TEAL_D', 'TEAL_E',
    'MAROON', 'MAROON_A', 'MAROON_B', 'MAROON_C', 'MAROON_D', 'MAROON_E',
    'PURPLE', 'PURPLE_A', 'PURPLE_B', 'PURPLE_C', 'PURPLE_D', 'PURPLE_E',
    'PINK', 'ORANGE', 'BLACK',
    'SMALL_BUFF', 'MED_SMALL_BUFF', 'MED_LARGE_BUFF', 'LARGE_BUFF',
  ]);

  const MANIM_CLASSES = new Set([
    'Scene', 'InteractiveScene', 'ThreeDAxes', 'NumberPlane', 'Sphere',
    'Surface', 'SurfaceMesh', 'VMobject', 'VGroup', 'Group', 'Mobject',
    'Dot', 'TrueDot', 'GlowDot', 'DotCloud', 'Arrow', 'Line', 'Square',
    'Circle', 'Arc', 'Tex', 'Text', 'Integer', 'SVGMobject', 'ImageMobject',
    'Clock', 'Point', 'VectorField', 'StreamLines', 'AnimatedStreamLines',
    'ParametricCurve', 'Vector', 'ValueTracker',
    'Axes', 'ComplexPlane', 'CoordinateSystem', 'NumberLine',
    'Rectangle', 'Polygon', 'Triangle', 'Annulus', 'AnnularSector',
    'TexText', 'MathTex', 'DecimalNumber', 'Matrix',
    'BarChart', 'SampleSpace', 'Brace', 'BraceLabel',
    'FullScreenRectangle', 'ScreenRectangle', 'SurroundingRectangle',
    'BackgroundRectangle', 'Cross', 'Exponent',
    'CubicBezier', 'Polyline', 'DashedLine', 'TangentLine',
    'DoubleArrow', 'Elbow', 'CurvedArrow', 'FunctionGraph',
    'ParametricFunction', 'ImplicitFunction',
  ]);

  const MANIM_FUNCTIONS = new Set([
    'ShowCreation', 'Write', 'FadeIn', 'FadeOut', 'Transform',
    'TransformFromCopy', 'Restore', 'GrowArrow', 'VFadeIn',
    'VFadeInThenOut', 'MoveAlongPath', 'Rotate', 'ClockPassesTime',
    'cycle_animation', 'color_gradient', 'normalize', 'rotate_vector',
    'interpolate', 'z_to_vector', 'normalize_along_axis',
    'ReplacementTransform', 'MoveToTarget', 'ApplyMethod',
    'AnimationGroup', 'Succession', 'LaggedStart', 'LaggedStartMap',
    'DrawBorderThenFill', 'GrowFromCenter', 'GrowFromPoint',
    'SpinInFromNothing', 'ShrinkToCenter', 'FadeInFromPoint',
    'FadeTransform', 'CountInFrom', 'Indicate', 'Flash',
    'ShowPassingFlash', 'ShowCreationThenDestruction',
    'ShowCreationThenFadeOut', 'ApplyWave', 'WiggleOutThenIn',
    'TurnInsideOut', 'FlashAround', 'FlashUnder',
    'Uncreate', 'UpdateFromFunc', 'UpdateFromAlphaFunc',
    'MaintainPositionRelativeTo', 'always_redraw', 'always_shift',
  ]);

  /**
   * Tokenize and highlight a single line of Python code.
   * Returns an HTML string with <span class="syn-*"> wrappers.
   *
   * This is a linear scanner -- no external dependencies.
   */
  function highlightPython(line) {
    // We'll keep track of whether we're inside a multiline string context
    // For single-line highlighting, we handle triple-quotes that start and end on the same line.
    const tokens = tokenizePython(line);
    return tokens.map(renderToken).join('');
  }

  /**
   * Tokenize a single line into an array of { type, value } tokens.
   */
  function tokenizePython(line) {
    const tokens = [];
    let i = 0;
    const len = line.length;

    while (i < len) {
      const ch = line[i];
      const rest = line.slice(i);

      // ── Comments ──
      if (ch === '#') {
        tokens.push({ type: 'comment', value: line.slice(i) });
        break;
      }

      // ── Decorators ──
      if (ch === '@' && (i === 0 || /^\s*$/.test(line.slice(0, i)))) {
        // Decorator: consume the rest of the line up to (
        const match = rest.match(/^@[\w.]+/);
        if (match) {
          tokens.push({ type: 'decorator', value: match[0] });
          i += match[0].length;
          continue;
        }
      }

      // ── Strings ──
      // Raw strings: r"...", r'...', R"...", R'...'
      // Also f-strings: f"...", b"..."
      if (/^[rRbBfFuU]{0,2}["']/.test(rest)) {
        const strResult = consumeString(line, i);
        if (strResult) {
          tokens.push({ type: 'string', value: strResult.value });
          i = strResult.end;
          continue;
        }
      }

      // ── Numbers ──
      if (/[0-9]/.test(ch) || (ch === '.' && i + 1 < len && /[0-9]/.test(line[i + 1]))) {
        const numMatch = rest.match(/^(?:0[xXoObB][\da-fA-F_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[jJ]?)/);
        if (numMatch) {
          tokens.push({ type: 'number', value: numMatch[0] });
          i += numMatch[0].length;
          continue;
        }
      }

      // ── Identifiers / Keywords ──
      if (/[a-zA-Z_]/.test(ch)) {
        const idMatch = rest.match(/^[a-zA-Z_]\w*/);
        if (idMatch) {
          const word = idMatch[0];
          let type = 'identifier';

          if (PY_SELF.has(word)) {
            type = 'self';
          } else if (PY_KEYWORDS.has(word)) {
            type = 'keyword';
          } else if (MANIM_CONSTANTS.has(word)) {
            type = 'constant';
          } else if (MANIM_CLASSES.has(word)) {
            type = 'class';
          } else if (MANIM_FUNCTIONS.has(word)) {
            type = 'builtin';
          } else {
            // Check if followed by ( — then it's a function call
            const afterWord = line.slice(i + word.length);
            if (/^\s*\(/.test(afterWord)) {
              type = 'function';
            }
          }

          tokens.push({ type, value: word });
          i += word.length;
          continue;
        }
      }

      // ── Operators / punctuation ──
      const opMatch = rest.match(/^(?:>>>=?|<<=?|[+\-*/%&|^~]=?|={1,2}|!=|<=?|>=?|->|\*\*=?|\/\/=?|:=|[()[\]{},;:.])/);
      if (opMatch) {
        tokens.push({ type: 'operator', value: opMatch[0] });
        i += opMatch[0].length;
        continue;
      }

      // ── Whitespace ──
      if (/\s/.test(ch)) {
        const wsMatch = rest.match(/^\s+/);
        tokens.push({ type: 'whitespace', value: wsMatch[0] });
        i += wsMatch[0].length;
        continue;
      }

      // ── Fallback: single character ──
      tokens.push({ type: 'plain', value: ch });
      i++;
    }

    return tokens;
  }

  /**
   * Consume a string literal starting at position i.
   * Handles triple-quoted, single-quoted, raw strings, f-strings, etc.
   */
  function consumeString(line, i) {
    let start = i;

    // Skip optional prefix
    const prefixMatch = line.slice(i).match(/^[rRbBfFuU]{1,2}/);
    if (prefixMatch) {
      i += prefixMatch[0].length;
    }

    if (i >= line.length) return null;

    const quoteChar = line[i];
    if (quoteChar !== '"' && quoteChar !== "'") return null;

    // Check for triple quotes
    const triple = line.slice(i, i + 3);
    if (triple === '"""' || triple === "'''") {
      // Triple-quoted string
      const endQuote = triple;
      let j = i + 3;
      while (j < line.length) {
        if (line[j] === '\\') {
          j += 2; // skip escaped char
          continue;
        }
        if (line.slice(j, j + 3) === endQuote) {
          return { value: line.slice(start, j + 3), end: j + 3 };
        }
        j++;
      }
      // Unterminated triple-quote on this line -- consume to end
      return { value: line.slice(start), end: line.length };
    }

    // Single-line string
    let j = i + 1;
    while (j < line.length) {
      if (line[j] === '\\') {
        j += 2;
        continue;
      }
      if (line[j] === quoteChar) {
        return { value: line.slice(start, j + 1), end: j + 1 };
      }
      j++;
    }
    // Unterminated -- consume to end
    return { value: line.slice(start), end: line.length };
  }

  /**
   * Render a single token as an HTML span.
   */
  function renderToken(token) {
    const escaped = escapeHTML(token.value);
    switch (token.type) {
      case 'keyword':    return '<span class="syn-kw">' + escaped + '</span>';
      case 'self':       return '<span class="syn-self">' + escaped + '</span>';
      case 'string':     return '<span class="syn-str">' + escaped + '</span>';
      case 'number':     return '<span class="syn-num">' + escaped + '</span>';
      case 'comment':    return '<span class="syn-cmt">' + escaped + '</span>';
      case 'decorator':  return '<span class="syn-dec">' + escaped + '</span>';
      case 'constant':   return '<span class="syn-const">' + escaped + '</span>';
      case 'class':      return '<span class="syn-cls">' + escaped + '</span>';
      case 'builtin':    return '<span class="syn-builtin">' + escaped + '</span>';
      case 'function':   return '<span class="syn-fn">' + escaped + '</span>';
      case 'operator':   return '<span class="syn-op">' + escaped + '</span>';
      default:           return escaped;
    }
  }

  // ============================================================
  //  SEARCH
  // ============================================================

  let searchOverlay = null;

  function setupSearch() {
    // Create the overlay element
    searchOverlay = document.createElement('div');
    searchOverlay.className = 'search-results';
    document.body.appendChild(searchOverlay);

    dom.searchInput.addEventListener('input', debounce(onSearchInput, 150));
    dom.searchInput.addEventListener('focus', () => {
      if (dom.searchInput.value.trim()) onSearchInput();
    });

    // Close overlay on click outside
    document.addEventListener('click', (e) => {
      if (!searchOverlay.contains(e.target) && !dom.searchBar.contains(e.target)) {
        searchOverlay.classList.remove('visible');
      }
    });

    // Close on Escape
    dom.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        searchOverlay.classList.remove('visible');
        dom.searchInput.blur();
      }
    });

    // Event delegation for result clicks
    searchOverlay.addEventListener('click', async (e) => {
      const result = e.target.closest('.search-result');
      if (result) {
        const path = result.dataset.path;
        const line = result.dataset.line;
        searchOverlay.classList.remove('visible');
        dom.searchInput.value = '';
        if (path) {
          await loadFile(path);
          if (line) {
            // Scroll to the matching line after rendering
            requestAnimationFrame(() => {
              scrollToLine(parseInt(line, 10));
            });
          }
        }
      }
    });
  }

  function onSearchInput() {
    const query = dom.searchInput.value.trim().toLowerCase();
    if (!query) {
      searchOverlay.classList.remove('visible');
      return;
    }

    const results = performSearch(query);

    if (results.length === 0) {
      searchOverlay.innerHTML =
        '<div style="padding:16px;text-align:center;color:var(--text-muted);font-size:13px;">No results found</div>';
    } else {
      searchOverlay.innerHTML = results
        .slice(0, 30)
        .map((r) => {
          const icon = r.type === 'file' ? '&#128196;' :
                       r.type === 'class' ? '&#128302;' : '&#128295;';
          return '<div class="search-result" data-path="' + escapeAttr(r.path) + '"' +
                 (r.line ? ' data-line="' + r.line + '"' : '') + '>' +
                   '<span class="search-result-icon">' + icon + '</span>' +
                   '<div>' +
                     '<div class="search-result-text">' + highlightSearchMatch(r.name, query) + '</div>' +
                     '<div class="search-result-path">' + escapeHTML(r.path) + '</div>' +
                   '</div>' +
                 '</div>';
        })
        .join('');
    }

    searchOverlay.classList.add('visible');
  }

  /**
   * Search across file names and class/function definitions.
   * Uses searchIndex for files not yet loaded, live code for loaded files.
   */
  function performSearch(query) {
    const results = [];

    // Gather all known file paths from fileIndex (covers all files)
    const allPaths = DATA.fileIndex ? Object.keys(DATA.fileIndex) : [];

    // Also include any files already in DATA.files that might not be in fileIndex
    if (DATA.files) {
      for (const path of Object.keys(DATA.files)) {
        if (!allPaths.includes(path)) allPaths.push(path);
      }
    }

    for (const path of allPaths) {
      const fileName = path.split('/').pop();
      const fileData = DATA.files ? DATA.files[path] : null;

      // Search file name
      if (fileName.toLowerCase().includes(query)) {
        results.push({ type: 'file', name: fileName, path, line: null });
      }

      // For loaded files, search actual code
      if (fileData && fileData.code) {
        const lines = fileData.code.split('\n');
        for (let i = 0; i < lines.length; i++) {
          const line = lines[i];
          const classMatch = line.match(/^\s*class\s+(\w+)/);
          if (classMatch && classMatch[1].toLowerCase().includes(query)) {
            results.push({ type: 'class', name: classMatch[1], path, line: i + 1 });
          }
          const funcMatch = line.match(/^\s*def\s+(\w+)/);
          if (funcMatch && funcMatch[1].toLowerCase().includes(query)) {
            results.push({ type: 'function', name: funcMatch[1], path, line: i + 1 });
          }
        }
      } else if (DATA.searchIndex && DATA.searchIndex[path]) {
        // For unloaded files, use the pre-built search index
        const idx = DATA.searchIndex[path];
        if (idx.classes) {
          for (const name of idx.classes) {
            if (name.toLowerCase().includes(query)) {
              results.push({ type: 'class', name, path, line: null });
            }
          }
        }
        if (idx.functions) {
          for (const name of idx.functions) {
            if (name.toLowerCase().includes(query)) {
              results.push({ type: 'function', name, path, line: null });
            }
          }
        }
        if (idx.methods) {
          for (const name of idx.methods) {
            if (name.toLowerCase().includes(query)) {
              results.push({ type: 'function', name, path, line: null });
            }
          }
        }
      }
    }

    // Sort: file matches first, then by name relevance
    results.sort((a, b) => {
      const aExact = a.name.toLowerCase() === query ? 0 : 1;
      const bExact = b.name.toLowerCase() === query ? 0 : 1;
      if (aExact !== bExact) return aExact - bExact;
      const aStarts = a.name.toLowerCase().startsWith(query) ? 0 : 1;
      const bStarts = b.name.toLowerCase().startsWith(query) ? 0 : 1;
      return aStarts - bStarts;
    });

    return results;
  }

  function highlightSearchMatch(text, query) {
    const escaped = escapeHTML(text);
    const idx = text.toLowerCase().indexOf(query);
    if (idx < 0) return escaped;
    const before = escapeHTML(text.slice(0, idx));
    const match = escapeHTML(text.slice(idx, idx + query.length));
    const after = escapeHTML(text.slice(idx + query.length));
    return before + '<span class="search-result-match">' + match + '</span>' + after;
  }

  function scrollToLine(lineNum) {
    const row = dom.codeBody.querySelector('tr.code-line[data-line="' + lineNum + '"]');
    if (row) {
      row.scrollIntoView({ behavior: 'smooth', block: 'center' });
      row.classList.add('highlighted');
      setTimeout(() => row.classList.remove('highlighted'), 2000);
    }
  }

  // ============================================================
  //  AI CHAT
  // ============================================================

  function setupChat() {
    // Toggle chat panel
    dom.btnChat.addEventListener('click', toggleChatPanel);

    // Send message
    dom.btnSend.addEventListener('click', sendMessage);

    // Textarea: Enter to send, Shift+Enter for newline, auto-resize
    dom.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    dom.chatInput.addEventListener('input', autoResizeTextarea);

    // Suggestion buttons (event delegation)
    dom.chatSuggestions.addEventListener('click', (e) => {
      const btn = e.target.closest('.suggestion');
      if (btn) {
        const question = btn.dataset.q;
        dom.chatInput.value = question;
        autoResizeTextarea();
        dom.chatInput.focus();
      }
    });

    // API key save/restore
    dom.apiKeyInput.addEventListener('input', () => {
      localStorage.setItem('manimgl_explorer_api_key', dom.apiKeyInput.value);
    });
  }

  function restoreApiKey() {
    const saved = localStorage.getItem('manimgl_explorer_api_key');
    if (saved) {
      dom.apiKeyInput.value = saved;
    }
  }

  function toggleChatPanel() {
    state.chatVisible = !state.chatVisible;
    dom.chatPanel.classList.toggle('hidden', !state.chatVisible);
    dom.btnChat.classList.toggle('active', state.chatVisible);
  }

  function autoResizeTextarea() {
    const ta = dom.chatInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
  }

  async function sendMessage() {
    const text = dom.chatInput.value.trim();
    if (!text || state.isSending) return;

    const apiKey = dom.apiKeyInput.value.trim();
    if (!apiKey) {
      appendChatMessage('assistant', 'Please enter your Anthropic API key in the field above.');
      return;
    }

    // Add user message to UI and history
    appendChatMessage('user', text);
    state.chatHistory.push({ role: 'user', content: text });
    dom.chatInput.value = '';
    autoResizeTextarea();

    // Show typing indicator
    const typingEl = showTypingIndicator();
    state.isSending = true;
    dom.btnSend.disabled = true;

    try {
      const response = await callClaudeAPI(apiKey, state.chatHistory);
      removeTypingIndicator(typingEl);
      appendChatMessage('assistant', response);
      state.chatHistory.push({ role: 'assistant', content: response });
    } catch (err) {
      removeTypingIndicator(typingEl);
      const errMsg = err.message || 'An error occurred while contacting the API.';
      appendChatMessage('assistant', 'Error: ' + errMsg);
    } finally {
      state.isSending = false;
      dom.btnSend.disabled = false;
    }
  }

  async function callClaudeAPI(apiKey, messages) {
    // Build system prompt
    let systemPrompt = DATA.architectureContext || 'You are an expert on ManimGL, the math animation library by 3Blue1Brown.';

    // Add current file context
    if (state.activeFileData && state.activeFileData.code) {
      systemPrompt += '\n\n--- Current File: ' + state.activeFilePath + ' ---\n' +
                      state.activeFileData.code;
    }

    const body = {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      system: systemPrompt,
      messages: messages,
    };

    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      const errMessage = errData && errData.error ? errData.error.message : 'API request failed with status ' + res.status;
      throw new Error(errMessage);
    }

    const data = await res.json();
    // Extract text from content blocks
    if (data.content && data.content.length > 0) {
      return data.content
        .filter((block) => block.type === 'text')
        .map((block) => block.text)
        .join('');
    }
    return 'No response received.';
  }

  function appendChatMessage(role, content) {
    // Remove the welcome message on first real message
    const welcomeEl = dom.chatMessages.querySelector('.chat-welcome');
    if (welcomeEl) welcomeEl.remove();

    const msgEl = document.createElement('div');
    msgEl.className = 'chat-msg ' + role;
    msgEl.innerHTML =
      '<div class="chat-msg-role">' + (role === 'user' ? 'You' : 'Claude') + '</div>' +
      '<div class="chat-msg-body">' +
        (role === 'user' ? escapeHTML(content) : renderMarkdown(content)) +
      '</div>';
    dom.chatMessages.appendChild(msgEl);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
  }

  function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'chat-msg assistant';
    el.innerHTML =
      '<div class="chat-msg-role">Claude</div>' +
      '<div class="chat-msg-body">' +
        '<div class="typing-indicator"><span></span><span></span><span></span></div>' +
      '</div>';
    dom.chatMessages.appendChild(el);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return el;
  }

  function removeTypingIndicator(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  // ============================================================
  //  MARKDOWN RENDERER (basic)
  // ============================================================

  /**
   * Render a subset of Markdown to HTML.
   * Supports: code blocks (```), inline code (`), bold (**), paragraphs.
   */
  function renderMarkdown(text) {
    if (!text) return '';

    // Split into blocks by code fences
    const parts = text.split(/(```[\s\S]*?```)/g);
    let html = '';

    for (const part of parts) {
      if (part.startsWith('```') && part.endsWith('```')) {
        // Code block
        const inner = part.slice(3, -3);
        // Remove optional language identifier on first line
        const firstNewline = inner.indexOf('\n');
        const codeContent = firstNewline >= 0 ? inner.slice(firstNewline + 1) : inner;
        html += '<pre><code>' + escapeHTML(codeContent) + '</code></pre>';
      } else {
        // Regular text: split into paragraphs
        const paragraphs = part.split(/\n{2,}/);
        for (const para of paragraphs) {
          const trimmed = para.trim();
          if (!trimmed) continue;
          let processed = escapeHTML(trimmed);
          // Bold
          processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
          // Inline code
          processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
          // Line breaks within paragraph
          processed = processed.replace(/\n/g, '<br>');
          html += '<p>' + processed + '</p>';
        }
      }
    }

    return html;
  }

  // ============================================================
  //  UTILITIES
  // ============================================================

  function escapeHTML(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escapeAttr(str) {
    return escapeHTML(str);
  }

  function debounce(fn, ms) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // ── Boot ────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
