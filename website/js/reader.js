/**
 * Illumine Lingao Online Reader
 * Loads chapters dynamically from content files with progress tracking
 */

class LingaoReader {
    constructor() {
        this.chapters = []; // All chapter metadata
        this.loadedParts = new Set(); // Track which parts are loaded
        this.currentChapter = 0;
        this.totalChapters = 852;
        this.settings = this.loadSettings();
        this.progress = this.loadProgress();

        // DOM Elements
        this.elements = this.cacheElements();

        this.init();
    }

    cacheElements() {
        return {
            loadingScreen: document.getElementById('loading-screen'),
            loadingStatus: document.getElementById('loading-status'),
            loadingProgress: document.getElementById('loading-progress'),
            readerInterface: document.getElementById('reader-interface'),
            chapterContent: document.getElementById('chapter-content'),
            tocList: document.getElementById('toc-list'),
            sidebar: document.getElementById('sidebar'),
            settingsPanel: document.getElementById('settings-panel'),
            searchPanel: document.getElementById('search-panel'),
            overlay: document.getElementById('overlay'),
            currentChapterTitle: document.getElementById('current-chapter-title'),
            chapterNumber: document.getElementById('chapter-number'),
            progressBar: document.getElementById('progress-bar'),
            progressText: document.getElementById('progress-text'),
            percentText: document.getElementById('percent-text'),
            prevBtn: document.getElementById('prev-chapter'),
            nextBtn: document.getElementById('next-chapter'),
            searchInput: document.getElementById('search-input'),
            searchResults: document.getElementById('search-results'),
            searchStatus: document.getElementById('search-status'),
            continueBtn: document.getElementById('continue-reading'),
            themeIconLight: document.getElementById('theme-icon-light'),
            themeIconDark: document.getElementById('theme-icon-dark')
        };
    }

    async init() {
        this.bindEvents();
        this.applySettings();

        // Load chapter index first
        await this.loadChapterIndex();

        // Check for saved progress or start from beginning
        const savedChapter = this.progress.chapter || 0;
        this.showReader();
        await this.loadChapter(savedChapter);

        // Restore scroll position after a short delay
        if (this.progress.scroll > 0) {
            setTimeout(() => {
                window.scrollTo(0, this.progress.scroll);
            }, 100);
        }
    }

    bindEvents() {
        // Navigation
        document.getElementById('menu-toggle')?.addEventListener('click', () => this.toggleSidebar());
        document.getElementById('sidebar-close')?.addEventListener('click', () => this.closeSidebar());
        document.getElementById('toc-toggle')?.addEventListener('click', () => this.toggleSidebar());
        document.getElementById('settings-toggle')?.addEventListener('click', () => this.openSettings());
        document.getElementById('settings-close')?.addEventListener('click', () => this.closeSettings());
        document.getElementById('search-toggle')?.addEventListener('click', () => this.openSearch());
        document.getElementById('search-close')?.addEventListener('click', () => this.closeSearch());
        document.getElementById('theme-toggle')?.addEventListener('click', () => this.toggleTheme());

        this.elements.overlay.addEventListener('click', () => {
            this.closeSidebar();
            this.closeSettings();
            this.closeSearch();
        });

        // Chapter navigation
        this.elements.prevBtn.addEventListener('click', () => this.prevChapter());
        this.elements.nextBtn.addEventListener('click', () => this.nextChapter());
        this.elements.continueBtn?.addEventListener('click', () => this.nextChapter());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' && !e.ctrlKey && !e.metaKey) {
                // Don't trigger if in input
                if (document.activeElement?.tagName !== 'INPUT') {
                    this.prevChapter();
                }
            } else if (e.key === 'ArrowRight' && !e.ctrlKey && !e.metaKey) {
                if (document.activeElement?.tagName !== 'INPUT') {
                    this.nextChapter();
                }
            }
        });

        // Settings
        this.bindSettings();

        // Scroll progress tracking
        window.addEventListener('scroll', this.throttle(() => {
            this.updateScrollProgress();
            this.saveProgress();
        }, 250));

        // Before unload - save position
        window.addEventListener('beforeunload', () => {
            this.saveProgress();
        });
    }

    bindSettings() {
        // Theme
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const theme = btn.dataset.theme;
                this.setTheme(theme);
                document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        // Font size
        document.getElementById('font-decrease')?.addEventListener('click', () => this.changeFontSize(-1));
        document.getElementById('font-increase')?.addEventListener('click', () => this.changeFontSize(1));

        // Font family
        document.getElementById('font-family')?.addEventListener('change', (e) => {
            this.setFontFamily(e.target.value);
        });

        // Line height
        document.getElementById('line-height')?.addEventListener('input', (e) => {
            this.setLineHeight(e.target.value);
        });

        // Reset progress
        document.getElementById('reset-progress')?.addEventListener('click', () => {
            this.resetProgress();
        });

        // Search
        document.getElementById('search-btn')?.addEventListener('click', () => this.performSearch());
        this.elements.searchInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });
    }

    // ==================== Content Loading ====================

    async loadChapterIndex() {
        this.updateLoadingStatus('Loading chapter index...', 10);

        try {
            // Load the index script dynamically
            await this.loadScript('content/chapters_index.js');

            if (typeof CHAPTERS_INDEX !== 'undefined') {
                this.chapters = CHAPTERS_INDEX.chapterList;
                this.totalChapters = CHAPTERS_INDEX.totalChapters;
                this.renderTOC();
            } else {
                throw new Error('CHAPTERS_INDEX not defined after loading script');
            }
        } catch (error) {
            console.error('Error loading chapter index:', error);
            this.updateLoadingStatus('Error loading content: ' + error.message, 0);
            throw error;
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            // Check if script already exists
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed to load ${src}`));
            document.head.appendChild(script);
        });
    }

    async loadChapterPart(partNum) {
        if (this.loadedParts.has(partNum)) return;

        this.updateLoadingStatus(`Loading chapters (part ${partNum}/10)...`, 20 + (partNum * 5));

        try {
            await this.loadScript(`content/chapters_part_${partNum}.js`);
            this.loadedParts.add(partNum);
        } catch (error) {
            console.error(`Error loading part ${partNum}:`, error);
            throw error;
        }
    }

    async loadChapter(chapterNum) {
        if (chapterNum < 0 || chapterNum >= this.totalChapters) return;

        this.currentChapter = chapterNum;

        // Determine which part this chapter is in
        const partNum = Math.floor(chapterNum / 86) + 1;

        // Load the part if not already loaded
        if (!this.loadedParts.has(partNum)) {
            this.showLoading();
            await this.loadChapterPart(partNum);
        }

        // Get chapter content
        let chapterContent = '';
        let chapterTitle = '';

        chapterContent = this.getChapterFromPart(partNum, chapterNum);

        if (!chapterContent) {
            // Try loading all parts until we find it
            for (let i = 1; i <= 10; i++) {
                if (!this.loadedParts.has(i)) {
                    await this.loadChapterPart(i);
                }
                chapterContent = this.getChapterFromPart(i, chapterNum);
                if (chapterContent) break;
            }
        }

        // Render content
        this.renderChapter(chapterContent, this.currentChapter);
        this.updateUI();
        this.hideLoading();

        // Save progress
        this.saveProgress();

        // Scroll to top
        window.scrollTo(0, 0);
    }

    getChapterFromPart(partNum, chapterNum) {
        const partVar = window[`CHAPTERS_PART_${partNum}`];
        if (!partVar || !Array.isArray(partVar)) return null;
        const chapter = partVar.find(c => c.number === chapterNum);
        return chapter ? chapter.content : null;
    }

    renderChapter(content, chapterNum) {
        // Get title from chapters index
        const chapterInfo = this.chapters.find(c => c.number === chapterNum);
        const title = chapterInfo ? chapterInfo.title : `Chapter ${chapterNum}`;

        if (!content) {
            this.elements.chapterContent.innerHTML = `
                <h1>Chapter ${chapterNum}: ${this.escapeHtml(title)}</h1>
                <p>Content not available. Please try refreshing.</p>
            `;
            return;
        }

        // Process content - convert plain text to HTML paragraphs
        const paragraphs = content.split('\n\n').filter(p => p.trim());
        const htmlContent = paragraphs.map(p => {
            // Check if it looks like a chapter heading
            if (p.match(/^Chapter\s+\d+:/i)) {
                return `<h1>${this.escapeHtml(p)}</h1>`;
            }
            // Regular paragraph
            return `<p>${this.escapeHtml(p)}</p>`;
        }).join('\n');

        this.elements.chapterContent.innerHTML = htmlContent;

        // Show/hide continue button
        if (this.elements.continueBtn) {
            if (chapterNum < this.totalChapters - 1) {
                this.elements.continueBtn.classList.remove('hidden');
                this.elements.continueBtn.textContent = `Continue to Chapter ${chapterNum + 1}`;
            } else {
                this.elements.continueBtn.classList.add('hidden');
            }
        }
    }

    renderTOC() {
        this.elements.tocList.innerHTML = '';

        this.chapters.forEach((chapter, index) => {
            const tocItem = document.createElement('div');
            tocItem.className = 'toc-item';
            tocItem.innerHTML = `
                <span class="chapter-number">Chapter ${chapter.number}</span>
                ${this.escapeHtml(chapter.title)}
            `;
            tocItem.addEventListener('click', () => {
                this.loadChapter(chapter.number);
                // Scroll this item into view in the TOC
                tocItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                if (window.innerWidth <= 1024) this.closeSidebar();
            });
            this.elements.tocList.appendChild(tocItem);
        });
    }

    // ==================== Navigation ====================

    nextChapter() {
        if (this.currentChapter < this.totalChapters - 1) {
            this.loadChapter(this.currentChapter + 1);
        }
    }

    prevChapter() {
        if (this.currentChapter > 0) {
            this.loadChapter(this.currentChapter - 1);
        }
    }

    // ==================== Progress & Settings ====================

    loadProgress() {
        const defaults = {
            chapter: 0,
            scroll: 0,
            lastRead: null
        };

        const saved = localStorage.getItem('lingao-progress');
        return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
    }

    saveProgress() {
        const progress = {
            chapter: this.currentChapter,
            scroll: window.scrollY,
            lastRead: new Date().toISOString()
        };
        localStorage.setItem('lingao-progress', JSON.stringify(progress));

        // Also save as cookie for backup/easy access
        document.cookie = `lingao_chapter=${this.currentChapter}; path=/; max-age=31536000`;
        document.cookie = `lingao_scroll=${window.scrollY}; path=/; max-age=31536000`;
    }

    resetProgress() {
        localStorage.removeItem('lingao-progress');
        document.cookie = 'lingao_chapter=0; path=/; max-age=0';
        document.cookie = 'lingao_scroll=0; path=/; max-age=0';
        this.progress = { chapter: 0, scroll: 0 };
        this.loadChapter(0);
        alert('Progress reset! Starting from Chapter 0.');
    }

    loadSettings() {
        const defaults = {
            theme: 'dark',
            fontSize: 18,
            fontFamily: 'Crimson Text',
            lineHeight: 1.6
        };

        const saved = localStorage.getItem('lingao-settings');
        return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
    }

    saveSettings() {
        localStorage.setItem('lingao-settings', JSON.stringify(this.settings));
    }

    applySettings() {
        this.setTheme(this.settings.theme);
        this.setFontSize(this.settings.fontSize);
        this.setFontFamily(this.settings.fontFamily);
        this.setLineHeight(this.settings.lineHeight);

        // Update UI controls
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.theme === this.settings.theme);
        });

        const fontFamilySelect = document.getElementById('font-family');
        if (fontFamilySelect) fontFamilySelect.value = this.settings.fontFamily;

        const lineHeightInput = document.getElementById('line-height');
        if (lineHeightInput) {
            lineHeightInput.value = this.settings.lineHeight;
            document.getElementById('line-height-value').textContent = this.settings.lineHeight;
        }
    }

    // ==================== Theme & Display ====================

    toggleTheme() {
        const themes = ['light', 'sepia', 'dark'];
        const currentIndex = themes.indexOf(this.settings.theme);
        const nextTheme = themes[(currentIndex + 1) % themes.length];
        this.setTheme(nextTheme);
    }

    setTheme(theme) {
        this.settings.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);

        if (theme === 'dark') {
            this.elements.themeIconLight?.classList.add('hidden');
            this.elements.themeIconDark?.classList.remove('hidden');
        } else {
            this.elements.themeIconLight?.classList.remove('hidden');
            this.elements.themeIconDark?.classList.add('hidden');
        }

        this.saveSettings();
    }

    changeFontSize(delta) {
        const newSize = Math.max(12, Math.min(32, this.settings.fontSize + delta));
        this.setFontSize(newSize);
    }

    setFontSize(size) {
        this.settings.fontSize = size;
        document.getElementById('font-size-value').textContent = `${size}px`;
        this.elements.chapterContent.style.fontSize = `${size}px`;
        this.saveSettings();
    }

    setFontFamily(font) {
        this.settings.fontFamily = font;
        this.elements.chapterContent.style.fontFamily = font;
        this.saveSettings();
    }

    setLineHeight(height) {
        this.settings.lineHeight = height;
        document.getElementById('line-height-value').textContent = height;
        this.elements.chapterContent.style.lineHeight = height;
        this.saveSettings();
    }

    // ==================== UI State ====================

    showLoading() {
        this.elements.loadingScreen.classList.remove('hidden');
    }

    hideLoading() {
        this.elements.loadingScreen.classList.add('hidden');
    }

    showReader() {
        this.elements.loadingScreen.classList.add('hidden');
        this.elements.readerInterface.classList.remove('hidden');
    }

    updateLoadingStatus(text, percent) {
        this.elements.loadingStatus.textContent = text;
        this.elements.loadingProgress.style.width = `${percent}%`;
    }

    updateUI() {
        // Title
        const chapter = this.chapters[this.currentChapter];
        const title = chapter ? chapter.title : `Chapter ${this.currentChapter}`;
        this.elements.currentChapterTitle.textContent = `Chapter ${this.currentChapter}: ${title}`;
        this.elements.chapterNumber.textContent = `${this.currentChapter + 1} / ${this.totalChapters}`;

        // Progress text
        this.elements.progressText.textContent = `Chapter ${this.currentChapter} of ${this.totalChapters - 1}`;
        const percent = Math.round((this.currentChapter / (this.totalChapters - 1)) * 100);
        this.elements.percentText.textContent = `${percent}%`;

        // Navigation buttons
        this.elements.prevBtn.disabled = this.currentChapter === 0;
        this.elements.nextBtn.disabled = this.currentChapter >= this.totalChapters - 1;

        // TOC active state - only scroll into view if user clicked from TOC
        document.querySelectorAll('.toc-item').forEach((item, index) => {
            const isActive = this.chapters[index]?.number === this.currentChapter;
            item.classList.toggle('active', isActive);
        });
    }

    updateScrollProgress() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        this.elements.progressBar.style.width = `${progress}%`;
    }

    // ==================== Sidebar & Panels ====================

    toggleSidebar() {
        this.elements.sidebar.classList.toggle('open');
        if (window.innerWidth <= 1024) {
            this.elements.overlay.classList.toggle('show');
        }
    }

    closeSidebar() {
        this.elements.sidebar.classList.remove('open');
        this.elements.overlay.classList.remove('show');
    }

    openSettings() {
        this.elements.settingsPanel.classList.add('open');
        this.elements.overlay.classList.add('show');
    }

    closeSettings() {
        this.elements.settingsPanel.classList.remove('open');
        this.elements.overlay.classList.remove('show');
    }

    openSearch() {
        this.elements.searchPanel.classList.add('open');
        this.elements.overlay.classList.add('show');
        setTimeout(() => this.elements.searchInput.focus(), 100);
    }

    closeSearch() {
        this.elements.searchPanel.classList.remove('open');
        this.elements.overlay.classList.remove('show');
    }

    // ==================== Search ====================

    async performSearch() {
        const query = this.elements.searchInput.value.trim().toLowerCase();
        if (!query || query.length < 3) {
            this.elements.searchStatus.textContent = 'Enter at least 3 characters';
            return;
        }

        this.elements.searchStatus.textContent = 'Searching...';
        this.elements.searchResults.innerHTML = '';

        const results = [];

        // Load all parts if not loaded
        for (let i = 1; i <= 10; i++) {
            if (!this.loadedParts.has(i)) {
                await this.loadChapterPart(i);
            }

            try {
                const partData = window[`CHAPTERS_PART_${i}`];
                if (!partData || !Array.isArray(partData)) continue;
                partData.forEach(chapter => {
                    if (chapter.content.toLowerCase().includes(query)) {
                        // Find context
                        const index = chapter.content.toLowerCase().indexOf(query);
                        const start = Math.max(0, index - 50);
                        const end = Math.min(chapter.content.length, index + query.length + 50);
                        const context = chapter.content.substring(start, end);

                        results.push({
                            chapter: chapter.number,
                            title: chapter.title,
                            context: context,
                            query: query
                        });
                    }
                });
            } catch (e) {}
        }

        this.displaySearchResults(results);
    }

    displaySearchResults(results) {
        if (results.length === 0) {
            this.elements.searchStatus.textContent = 'No results found';
            return;
        }

        this.elements.searchStatus.textContent = `${results.length} result(s) found`;

        results.forEach(result => {
            const div = document.createElement('div');
            div.className = 'search-result';

            const highlightedContext = result.context.replace(
                new RegExp(`(${result.query})`, 'gi'),
                '<mark>$1</mark>'
            );

            div.innerHTML = `
                <div class="result-page">Chapter ${result.chapter}: ${this.escapeHtml(result.title)}</div>
                <div class="result-text">...${highlightedContext}...</div>
            `;

            div.addEventListener('click', () => {
                this.loadChapter(result.chapter);
                this.closeSearch();
            });

            this.elements.searchResults.appendChild(div);
        });
    }

    // ==================== Utilities ====================

    throttle(func, limit) {
        let inThrottle;
        return (...args) => {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new LingaoReader();
});
