/**
 * AWAI - Sukabumi Intelligent Traffic Analytics
 * Frontend Interactive Controller
 */

// Initialize state variables
let API_URL = localStorage.getItem('AWAI_API_URL') || (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://127.0.0.1:8000' 
        : 'https://awai-backend.onrender.com'
);

let API_KEY = localStorage.getItem('AWAI_API_KEY') || 'awai_api_key_rev2026';

// Helper to make authenticated fetch calls to the backend
async function fetchWithAuth(url, options = {}) {
    const headers = options.headers || {};
    if (API_KEY) {
        headers['x-api-key'] = API_KEY;
    }
    return fetch(url, { ...options, headers });
}

let map = null;
let roadsData = [];
let polylines = {};
let selectedRoadId = null;
let forecastChart = null;
let updateInterval = null;
let statusInterval = null;

// DOM Elements
const docElements = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    viewPanels: document.querySelectorAll('.view-panel'),
    viewTitle: document.getElementById('view-title'),
    viewSubtitle: document.getElementById('view-subtitle'),
    
    // Sidebar Status
    healthDot: document.getElementById('health-dot'),
    healthLabel: document.getElementById('health-label'),
    activeModelName: document.getElementById('active-model-name'),
    currentTimeWib: document.getElementById('current-time-wib'),
    btnRefreshData: document.getElementById('btn-refresh-data'),
    
    // Dashboard Stats
    valAvgSpeed: document.getElementById('val-avg-speed'),
    txtSpeedTrend: document.getElementById('txt-speed-trend'),
    valCongestedRoads: document.getElementById('val-congested-roads'),
    txtCongestedPercentage: document.getElementById('txt-congested-percentage'),
    valQualityScore: document.getElementById('val-quality-score'),
    txtQualityStatus: document.getElementById('txt-quality-status'),
    valPredMode: document.getElementById('val-pred-mode'),
    txtPredCache: document.getElementById('txt-pred-cache'),
    
    // Sidebar Detail Panel
    telemetryEmptyState: document.getElementById('road-list-panel'),
    telemetryActiveState: document.getElementById('telemetry-active-state'),
    roadListPanel: document.getElementById('road-list-panel'),
    roadListScroll: document.getElementById('road-list-scroll'),
    roadSearchInput: document.getElementById('road-search-input'),
    roadCountBadge: document.getElementById('road-count-badge'),
    btnBackToList: document.getElementById('btn-back-to-list'),
    segmentName: document.getElementById('segment-name'),
    segmentId: document.getElementById('segment-id'),
    segmentCongestion: document.getElementById('segment-congestion'),
    segmentCurrentSpeed: document.getElementById('segment-current-speed'),
    segmentConfidence: document.getElementById('segment-confidence'),
    segmentConfidenceBar: document.getElementById('segment-confidence-bar'),
    segmentWeight: document.getElementById('segment-weight'),
    segmentLastUpdate: document.getElementById('segment-last-update'),
    speedRing: document.getElementById('speed-ring'),
    
    // Predictions Summary list
    pred15m: document.getElementById('pred-15m'),
    pred30m: document.getElementById('pred-30m'),
    pred45m: document.getElementById('pred-45m'),
    pred60m: document.getElementById('pred-60m'),
    
    // Manual Ingest Form
    formManualIngest: document.getElementById('form-manual-ingest'),
    ingestRoadId: document.getElementById('ingest-road-id'),
    ingestSpeed: document.getElementById('ingest-speed'),
    ingestConfidence: document.getElementById('ingest-confidence'),
    btnDemoIngest: document.getElementById('btn-demo-ingest'),
    btnSubmitIngest: document.getElementById('btn-submit-ingest'),
    ingestAlertError: document.getElementById('ingest-alert-error'),
    ingestErrorText: document.getElementById('ingest-error-text'),
    ingestAlertSuccess: document.getElementById('ingest-alert-success'),
    
    // System Status Page Lists
    readyDetailsList: document.getElementById('ready-details-list'),
    schedulerDetailsList: document.getElementById('scheduler-details-list'),
    auditCompleteness: document.getElementById('audit-completeness'),
    auditCompTxt: document.getElementById('audit-comp-txt'),
    auditStaleCount: document.getElementById('audit-stale-count'),
    auditMissingCount: document.getElementById('audit-missing-count'),
    auditDriftStatus: document.getElementById('audit-drift-status'),
    qualityDonut: document.getElementById('quality-donut'),
    
    // Fullscreen Map elements
    mapContainerCard: document.getElementById('map-container-card'),
    btnMapFullscreen: document.getElementById('btn-map-fullscreen')
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize navigation hooks
    initNavigation();
    
    // 2. Initialize Leaflet Map
    initMap();
    initMapFullscreen();
    
    // 3. Start clocks & status loops
    startClock();
    checkSystemStatus();
    
    // 4. Initial dynamic data fetching
    refreshData();
    
    // 5. Connect UI events
    docElements.btnRefreshData.addEventListener('click', refreshData);
    docElements.formManualIngest.addEventListener('submit', handleManualIngest);
    docElements.btnDemoIngest.addEventListener('click', handleDemoNetworkIngest);

    // Road list back button
    if (docElements.btnBackToList) {
        docElements.btnBackToList.addEventListener('click', showRoadList);
    }

    // Road list search filter
    if (docElements.roadSearchInput) {
        docElements.roadSearchInput.addEventListener('input', (e) => {
            filterRoadList(e.target.value.trim().toLowerCase());
        });
    }
    
    // Set periodic refresh timers
    statusInterval = setInterval(checkSystemStatus, 10000); // 10s health checks
    updateInterval = setInterval(refreshData, 30000);       // 30s telemetry pull
    
    // Populate API setting card on system metrics page
    injectApiUrlSettingCard();
});

// Setup Navigation Switcher
function initNavigation() {
    docElements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active classes
            docElements.navItems.forEach(nav => nav.classList.remove('active'));
            docElements.viewPanels.forEach(panel => panel.classList.remove('active'));
            
            // Add active class
            item.classList.add('active');
            const targetTab = item.getAttribute('data-tab');
            const targetPanel = document.getElementById(`panel-${targetTab}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
            
            // Update Headers
            if (targetTab === 'dashboard') {
                docElements.viewTitle.innerText = "Traffic Telemetry & Predictions";
                docElements.viewSubtitle.innerText = "Real-time congestion map and LSTM prediction bounds for Sukabumi's 50 road segments";
            } else if (targetTab === 'manual-ingest') {
                docElements.viewTitle.innerText = "Data Ingestion Controls";
                docElements.viewSubtitle.innerText = "Feed current speed records directly into the state buffer and postgres database";
            } else if (targetTab === 'system-metrics') {
                docElements.viewTitle.innerText = "System Health & Audits";
                docElements.viewSubtitle.innerText = "Real-time logs of LSTM pipelines, in-process scheduler tasks, and data drift diagnostics";
            }
            
            // Recalculate Leaflet map sizes if returning to dashboard
            if (targetTab === 'dashboard' && map) {
                setTimeout(() => map.invalidateSize(), 100);
            }
        });
    });
}

// Clock Indicator
function startClock() {
    const updateWibClock = () => {
        const now = new Date();
        // Convert to WIB (UTC+7)
        const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
        const wibTime = new Date(utc + (3600000 * 7));
        
        let hrs = wibTime.getHours().toString().padStart(2, '0');
        let mins = wibTime.getMinutes().toString().padStart(2, '0');
        let secs = wibTime.getSeconds().toString().padStart(2, '0');
        
        docElements.currentTimeWib.textContent = `${hrs}:${mins}:${secs} WIB`;
    };
    updateWibClock();
    setInterval(updateWibClock, 1000);
}

// Leaflet Map Init
function initMap() {
    // Default Sukabumi coordinates center
    map = L.map('traffic-map', {
        zoomControl: false,
        attributionControl: false
    }).setView([-6.915, 106.85], 13);
    
    // Modern Sleek Dark Mode Map Tiles (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 20
    }).addTo(map);
    
    // Zoom control position
    L.control.zoom({
        position: 'bottomright'
    }).addTo(map);
}

// Toggle Fullscreen Map
function initMapFullscreen() {
    const btn = docElements.btnMapFullscreen;
    const container = docElements.mapContainerCard;
    
    if (!btn || !container) return;
    
    const toggleFullscreen = () => {
        const isFullscreen = container.classList.toggle('is-fullscreen');
        
        // Update Lucide Icon
        btn.innerHTML = isFullscreen 
            ? '<i data-lucide="minimize-2"></i>' 
            : '<i data-lucide="maximize-2"></i>';
        
        if (window.lucide) {
            window.lucide.createIcons();
        }
        
        // Dynamic resizing for Leaflet
        if (map) {
            map.invalidateSize();
            // Invalidate size again after a brief timeout to let CSS transitions complete perfectly
            setTimeout(() => {
                map.invalidateSize();
            }, 150);
        }
    };
    
    btn.addEventListener('click', toggleFullscreen);
    
    // Add escape key handler to exit fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && container.classList.contains('is-fullscreen')) {
            toggleFullscreen();
        }
    });
}

// Dynamic API URL settings injection
function injectApiUrlSettingCard() {
    const parentContainer = docElements.readyDetailsList.parentElement.parentElement;
    
    // Check if card is already injected
    if (document.getElementById('api-settings-card')) return;
    
    const card = document.createElement('div');
    card.id = 'api-settings-card';
    card.className = 'metric-card glass';
    card.style.gridColumn = '1 / -1';
    card.style.marginTop = '1rem';
    
    card.innerHTML = `
        <div class="card-header">
            <h3><i data-lucide="settings"></i> Render API Configuration</h3>
        </div>
        <div style="padding: 1.25rem 1.5rem; display: flex; flex-direction: column; gap: 1rem;">
            <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4; margin: 0;">
                Provide the Render backend API base URL and the API Secret Key to securely authenticate requests.
            </p>
            <div style="display: flex; flex-direction: column; gap: 0.75rem; width: 100%;">
                <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                    <label for="setting-api-url" style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted);">API Base URL</label>
                    <input type="text" id="setting-api-url" class="glass-input" style="padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.85rem;" value="${API_URL}">
                </div>
                <div style="display: flex; flex-direction: column; gap: 0.25rem;">
                    <label for="setting-api-key" style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted);">API Secret Key (X-API-KEY)</label>
                    <input type="password" id="setting-api-key" class="glass-input" style="padding: 0.6rem 1rem; border-radius: 8px; font-size: 0.85rem;" value="${API_KEY}">
                </div>
                <button id="btn-save-api-url" class="btn btn-primary" style="padding: 0.6rem 1.2rem; border-radius: 8px; font-size: 0.85rem; width: fit-content; align-self: flex-end; display: flex; align-items: center; gap: 0.35rem;">
                    <i data-lucide="save" style="width: 14px; height: 14px;"></i> Save Configuration
                </button>
            </div>
            <span id="api-save-status" style="font-size: 0.75rem; font-weight: 600; display: none;"></span>
        </div>
    `;
    
    parentContainer.appendChild(card);
    if (window.lucide) window.lucide.createIcons();
    
    // Bind Save Action
    document.getElementById('btn-save-api-url').addEventListener('click', () => {
        const inputUrl = document.getElementById('setting-api-url').value.trim();
        const inputKey = document.getElementById('setting-api-key').value.trim();
        
        if (inputUrl) {
            API_URL = inputUrl;
            localStorage.setItem('AWAI_API_URL', API_URL);
            
            API_KEY = inputKey;
            localStorage.setItem('AWAI_API_KEY', API_KEY);
            
            const statusLabel = document.getElementById('api-save-status');
            statusLabel.innerText = "API configuration updated successfully! Re-initiating connection checks...";
            statusLabel.style.color = "var(--color-green)";
            statusLabel.style.display = "inline-block";
            
            checkSystemStatus();
            refreshData();
            
            setTimeout(() => {
                statusLabel.style.display = "none";
            }, 3000);
        }
    });
}

// ----------------------------------------------------
// TELEMETRY REFRESH & BATCH PREDICTION IMPLEMENTATION
// ----------------------------------------------------

async function refreshData() {
    try {
        docElements.btnRefreshData.classList.add('loading-spin');
        
        // 1. Fetch active roads if we haven't loaded them yet
        if (roadsData.length === 0) {
            await fetchRoads();
        }
        
        if (roadsData.length === 0) {
            docElements.btnRefreshData.classList.remove('loading-spin');
            return; // API is likely offline or roads empty
        }
        
        // 2. Fetch General Metrics
        await fetchGeneralMetrics();
        
        // 3. Batch Predict all 50 segments for dynamic coloration
        await refreshNetworkPredictions();
        
        // 4. Fetch System Audit Diagnostics
        await fetchSystemAudits();
        
        // 5. If we have a selected segment, refresh its detail panel
        if (selectedRoadId) {
            refreshSegmentDetails(selectedRoadId);
        }
        
        docElements.btnRefreshData.classList.remove('loading-spin');
    } catch (err) {
        console.error("Telemetry refresh failed:", err);
        docElements.btnRefreshData.classList.remove('loading-spin');
    }
}

// Fetch Road segments
async function fetchRoads() {
    try {
        const res = await fetchWithAuth(`${API_URL}/roads`);
        if (!res.ok) throw new Error("HTTP " + res.status);
        
        roadsData = await res.json();
        
        // Clear old map layers
        Object.values(polylines).forEach(poly => map.removeLayer(poly));
        polylines = {};
        
        // Populate manual ingestion dropdown
        docElements.ingestRoadId.innerHTML = '<option value="" disabled selected>Select a segment...</option>';
        
        const mapBounds = [];
        
        // Plot road polylines
        roadsData.forEach(road => {
            // Populate select option
            const opt = document.createElement('option');
            opt.value = road.road_id;
            opt.innerText = `${road.road_name} (${road.road_id})`;
            docElements.ingestRoadId.appendChild(opt);
            
            // Build polyline coordinates
            const startLat = road.start_lat || -6.915;
            const startLon = road.start_lon || 106.85;
            const endLat = road.end_lat || -6.915;
            const endLon = road.end_lon || 106.85;
            
            let coords = [[startLat, startLon], [endLat, endLon]];
            if (road.mid_lat && road.mid_lon) {
                coords = [[startLat, startLon], [road.mid_lat, road.mid_lon], [endLat, endLon]];
            }
            
            mapBounds.push([startLat, startLon]);
            mapBounds.push([endLat, endLon]);
            
            // Base polyline
            const poly = L.polyline(coords, {
                color: 'var(--text-muted)',
                weight: 5,
                opacity: 0.6,
                smoothFactor: 1.0
            }).addTo(map);
            
            poly.on('mouseover', () => poly.setStyle({ weight: 8, opacity: 0.95 }));
            poly.on('mouseout', () => poly.setStyle({
                weight: selectedRoadId === road.road_id ? 8 : 5,
                opacity: selectedRoadId === road.road_id ? 0.95 : 0.6
            }));
            poly.on('click', () => selectSegment(road.road_id));
            
            poly.bindTooltip(`${road.road_name || 'Unnamed Segment'} (${road.road_id})`, {
                sticky: true,
                className: 'custom-tooltip'
            });
            
            polylines[road.road_id] = poly;
        });
        
        // Auto zoom bounds dynamically
        if (mapBounds.length > 0) {
            map.fitBounds(L.latLngBounds(mapBounds), { padding: [30, 30] });
        }
        
        // Populate road list sidebar
        populateRoadList(roadsData);
        
        if (window.lucide) window.lucide.createIcons();
    } catch (err) {
        console.error("Failed to load road segments:", err);
    }
}

// Populate the sidebar road list from roadsData
function populateRoadList(roads) {
    if (!docElements.roadListScroll) return;

    docElements.roadListScroll.innerHTML = '';

    if (docElements.roadCountBadge) {
        docElements.roadCountBadge.textContent = `${roads.length} jalan`;
    }

    roads.forEach(road => {
        const item = document.createElement('div');
        item.className = 'road-list-item';
        item.dataset.roadId = road.road_id;
        item.dataset.roadName = (road.road_name || '').toLowerCase();
        item.innerHTML = `
            <div class="road-item-dot" id="dot-${road.road_id}"></div>
            <div class="road-item-info">
                <div class="road-item-name">${road.road_name || road.road_id}</div>
                <div class="road-item-id">${road.road_id}</div>
            </div>
            <div class="road-item-speed" id="speed-badge-${road.road_id}">
                <span>--</span>
                <span class="speed-unit-small">km/h</span>
            </div>
            <div class="road-item-chevron"><i data-lucide="chevron-right"></i></div>
        `;
        item.addEventListener('click', () => selectSegment(road.road_id));
        docElements.roadListScroll.appendChild(item);
    });

    if (window.lucide) window.lucide.createIcons();
}

// Filter list items by search query
function filterRoadList(query) {
    if (!docElements.roadListScroll) return;
    const items = docElements.roadListScroll.querySelectorAll('.road-list-item');
    let visibleCount = 0;

    items.forEach(item => {
        const name = item.dataset.roadName || '';
        const id   = (item.dataset.roadId || '').toLowerCase();
        const match = !query || name.includes(query) || id.includes(query);
        item.style.display = match ? '' : 'none';
        if (match) visibleCount++;
    });

    // Show/hide empty state
    let emptyEl = docElements.roadListScroll.querySelector('.road-list-empty');
    if (visibleCount === 0) {
        if (!emptyEl) {
            emptyEl = document.createElement('div');
            emptyEl.className = 'road-list-empty';
            emptyEl.innerHTML = `<i data-lucide="search-x"></i><p>Jalan "${query}" tidak ditemukan</p>`;
            docElements.roadListScroll.appendChild(emptyEl);
            if (window.lucide) window.lucide.createIcons();
        }
    } else {
        if (emptyEl) emptyEl.remove();
    }
}

// Fetch general metrics
async function fetchGeneralMetrics() {
    try {
        const res = await fetchWithAuth(`${API_URL}/metrics`);
        if (!res.ok) throw new Error("HTTP " + res.status);
        
        const data = await res.json();
        
        // Uptime info
        const uptimeHrs = (data.uptime_seconds / 3600).toFixed(1);
        docElements.txtSpeedTrend.innerText = `Uptime: ${uptimeHrs} hrs | Buffer: ${(data.buffer_average_fill_rate * 100).toFixed(0)}%`;
        
        // Prediction mode: lebih detail
        if (data.model_loaded) {
            docElements.valPredMode.innerText = 'LSTM Active';
            docElements.valPredMode.style.color = 'var(--color-green)';
        } else {
            docElements.valPredMode.innerText = 'Fallback';
            docElements.valPredMode.style.color = 'var(--color-amber)';
        }
        docElements.txtPredCache.innerText = `v${data.model_version || 'N/A'} | Cache: ${data.prediction_cache_size} entries`;
        
        // Data quality
        if (data.data_quality_status) {
            const statusUpper = data.data_quality_status.toUpperCase();
            docElements.txtQualityStatus.innerText = `Status: ${statusUpper}`;
            if (docElements.valQualityScore) {
                docElements.valQualityScore.innerText = data.data_quality_score != null ? (data.data_quality_score * 100).toFixed(0) : '--';
            }
        }
        
    } catch (err) {
        console.warn("Failed to fetch general system metrics:", err);
    }
}

// Batch predictions to color-code the map network flow
// Model prediksi LSTM adalah sumber utama; live speed ditampilkan sebagai data pendamping
async function refreshNetworkPredictions() {
    if (roadsData.length === 0) return;
    
    try {
        // ---- 1. Build correct PredictionBatchRequest format ----
        // Backend expects: { predictions: [{road_id, horizon_minutes}, ...] }
        const batchPayload = {
            predictions: roadsData.map(r => ({
                road_id: r.road_id,
                horizon_minutes: 15
            }))
        };

        const predRes = await fetchWithAuth(`${API_URL}/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(batchPayload)
        });
        
        // ---- 2. Parse PredictionBatchResponse correctly ----
        // Backend returns: { predictions: [...], successful_count, failed_count, ... }
        // Convert to map {road_id -> prediction} for easy lookup
        let predMap = {};
        if (predRes.ok || predRes.status === 206) {
            const batchResp = await predRes.json();
            if (batchResp.predictions && Array.isArray(batchResp.predictions)) {
                batchResp.predictions.forEach(p => {
                    predMap[p.road_id] = p;
                });
            }
        }
        
        // ---- 3. Fetch LIVE speeds (monitoring pendamping, bukan sumber utama) ----
        const liveRes = await fetchWithAuth(`${API_URL}/roads/live`);
        let liveMap = {};
        if (liveRes.ok) {
            const liveData = await liveRes.json();
            liveData.forEach(item => { liveMap[item.road_id] = item; });
        }
        
        let totalPredSpeed = 0;
        let activeCongestedCount = 0;
        let successCount = 0;
        let lstmCount = 0;
        let fallbackCount = 0;
        
        // ---- 4. Update map berdasarkan hasil prediksi ----
        roadsData.forEach(road => {
            const roadId = road.road_id;
            const pred = predMap[roadId];
            const live = liveMap[roadId];
            const poly = polylines[roadId];
            
            if (!pred) return; // skip jika prediksi gagal untuk road ini
            
            const predictedSpeed = pred.predicted_speed;
            const freeFlow = pred.free_flow_speed || road.free_flow_speed || 35.0;
            
            // Prioritaskan current_speed untuk monitoring aktual, gunakan prediksi jika current tidak ada
            const actualSpeed = (live && live.current_speed) ? live.current_speed : predictedSpeed;
            const speedRatio = actualSpeed / freeFlow;
            
            const predMethod = pred.prediction_method || '';
            const isLSTM = predMethod === 'live_lstm_runtime';
            
            if (isLSTM) lstmCount++; else fallbackCount++;
            
            // Klasifikasi berdasarkan speed aktual (monitoring)
            let strokeColor = 'var(--color-green)';
            let dotClass = 'green';
            let congText = 'Free Flow';
            let isCongested = false;
            
            // Threshold lebih akurat: < 0.30 Severe, < 0.50 Congested, < 0.75 Moderate
            if (speedRatio < 0.30) {
                strokeColor = 'var(--color-dark-red)'; dotClass = 'dark-red';
                congText = 'Severe Congestion'; isCongested = true;
            } else if (speedRatio < 0.50) {
                strokeColor = 'var(--color-red)'; dotClass = 'red';
                congText = 'Congested'; isCongested = true;
            } else if (speedRatio < 0.75) {
                strokeColor = 'var(--color-amber)'; dotClass = 'amber';
                congText = 'Moderate';
            }
            
            if (isCongested && poly) activeCongestedCount++;
            
            const roadName = road.road_name || roadId;
            const liveSpeedStr = live ? `${live.current_speed.toFixed(1)} km/h` : 'N/A';
            const staleBadge = (live && live.is_stale) ? ` ⚠` : '';
            const methodBadge = isLSTM ? '🧠 LSTM' : '📊 Est.';
            
            if (poly) {
                totalPredSpeed += predictedSpeed;
                successCount++;
                poly.setStyle({ color: strokeColor, weight: selectedRoadId === roadId ? 8 : 5 });
                poly.bindTooltip(`
                    <div class="map-tooltip-content">
                        <strong>${roadName}</strong>
                        <span style="font-size:0.7rem;color:var(--text-secondary);margin-left:4px">${roadId}</span><br/>
                        <div style="margin-top:5px;display:flex;flex-direction:column;gap:3px">
                            <span style="color:${strokeColor};font-weight:700;font-size:0.9rem">
                                Live now: ${liveSpeedStr}${staleBadge}
                            </span>
                            <span style="font-size:0.78rem;color:var(--text-secondary)">${congText}</span>
                            <span style="font-size:0.75rem;color:var(--text-muted)">
                                Pred +15m: ${predictedSpeed.toFixed(1)} km/h &nbsp;•&nbsp; ${methodBadge}
                            </span>
                        </div>
                    </div>
                `, { sticky: true, className: 'custom-tooltip' });
            }

            // Sidebar road list: tampilkan kecepatan aktual
            const dot = document.getElementById(`dot-${roadId}`);
            const badge = document.getElementById(`speed-badge-${roadId}`);
            if (dot) dot.className = `road-item-dot ${dotClass}`;
            if (badge) {
                badge.innerHTML = `<span>${actualSpeed.toFixed(1)}</span><span class="speed-unit-small">km/h</span>`;
                badge.style.color = strokeColor;
                badge.title = `Live: ${liveSpeedStr} | Pred +15m: ${predictedSpeed.toFixed(1)} km/h [${methodBadge}]`;
            }
        });
        
        // ---- 5. Update dashboard stats ----
        if (successCount > 0) {
            const avgPred = (totalPredSpeed / successCount).toFixed(1);
            docElements.valAvgSpeed.innerText = avgPred;
            docElements.valCongestedRoads.innerText = activeCongestedCount;
            const pctCongested = ((activeCongestedCount / successCount) * 100).toFixed(0);
            docElements.txtCongestedPercentage.innerText = `${pctCongested}% of ${successCount} segments`;
        }

        // Update prediction mode display
        const totalPred = lstmCount + fallbackCount;
        if (totalPred > 0) {
            const lstmPct = Math.round((lstmCount / totalPred) * 100);
            docElements.valPredMode.innerText = lstmCount > 0 ? `LSTM ${lstmPct}%` : 'Fallback';
            docElements.txtPredCache.innerText = `${lstmCount} LSTM · ${fallbackCount} Fallback dari ${totalPred} jalan`;
        }
        
    } catch (err) {
        console.error("Network predictions refresh failed:", err);
    }
}

// ----------------------------------------------------
// DYNAMIC SEGMENT telemetry SIDEBAR DETAILS
// ----------------------------------------------------

async function selectSegment(roadId) {
    // Reset weights of all polylines
    Object.keys(polylines).forEach(id => polylines[id].setStyle({ weight: 5 }));
    
    selectedRoadId = roadId;
    
    // Highlight selected segment on map
    if (polylines[roadId]) {
        polylines[roadId].setStyle({ weight: 8 });
    }

    // Update active state on list items
    if (docElements.roadListScroll) {
        docElements.roadListScroll.querySelectorAll('.road-list-item').forEach(el => {
            el.classList.toggle('active', el.dataset.roadId === roadId);
        });
    }
    
    // Show telemetry, hide road list
    if (docElements.roadListPanel)  docElements.roadListPanel.classList.add('hidden');
    if (docElements.telemetryActiveState) docElements.telemetryActiveState.classList.remove('hidden');
    
    // Show loading indicator
    docElements.segmentName.innerText = "Loading details...";
    docElements.segmentId.innerText = roadId;
    docElements.segmentCurrentSpeed.innerText = "--";
    
    await refreshSegmentDetails(roadId);
}

// Return to road list view
function showRoadList() {
    selectedRoadId = null;

    // Deselect all polylines
    Object.keys(polylines).forEach(id => polylines[id].setStyle({ weight: 5 }));

    // Remove active highlight from list items
    if (docElements.roadListScroll) {
        docElements.roadListScroll.querySelectorAll('.road-list-item').forEach(el => {
            el.classList.remove('active');
        });
    }

    // Show road list, hide telemetry
    if (docElements.roadListPanel)  docElements.roadListPanel.classList.remove('hidden');
    if (docElements.telemetryActiveState) docElements.telemetryActiveState.classList.add('hidden');

    // Clear search input
    if (docElements.roadSearchInput) {
        docElements.roadSearchInput.value = '';
        filterRoadList('');
    }
}

async function refreshSegmentDetails(roadId) {
    try {
        const roadObj = roadsData.find(r => r.road_id === roadId);
        if (!roadObj) return;
        
        // Render road details instantly
        docElements.segmentName.innerText = roadObj.road_name || 'Unnamed Segment';
        docElements.segmentId.innerText = roadId;
        docElements.segmentWeight.innerText = roadObj.road_weight ? roadObj.road_weight.toFixed(2) : '1.00';
        
        // 1. Fetch all 4 forecast horizons concurrently
        const horizons = [15, 30, 45, 60];
        const forecasts = {};
        
        const reqs = horizons.map(h => {
            return fetchWithAuth(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ road_id: roadId, horizon_minutes: h })
            }).then(async r => {
                if (!r.ok) throw new Error("Horizon failed " + h);
                return { horizon: h, data: await r.json() };
            });
        });
        
        const results = await Promise.allSettled(reqs);
        results.forEach(res => {
            if (res.status === 'fulfilled') {
                forecasts[res.value.horizon] = res.value.data;
            }
        });
        
        // 2. Prioritize LIVE speed for the main display ring (monitoring aktual), and show PREDICTED speed as secondary
        const pred15 = forecasts[15];
        let liveSpeedVal = pred15?.current_speed;
        let displaySpeedVal = pred15?.predicted_speed ?? 35.0;
        let confidenceScore = pred15?.confidence_score ?? 0.95;
        const freeFlowSpeed = pred15?.free_flow_speed ?? roadObj.free_flow_speed ?? 35.0;
        
        // 3. Classify berdasarkan LIVE speed untuk monitoring yang aktual
        const actualSpeed = liveSpeedVal != null ? liveSpeedVal : displaySpeedVal;
        const speedRatio = actualSpeed / freeFlowSpeed;
        
        let segmentCong = 'free';
        let segmentText = 'Free Flow';
        let ringColor = 'var(--color-green)';
        
        if (speedRatio < 0.40) {
            segmentCong = 'congested';
            segmentText = 'Severe Congestion';
            ringColor = 'var(--color-red)';
        } else if (speedRatio < 0.60) {
            segmentCong = 'congested';
            segmentText = 'Congested';
            ringColor = 'var(--color-red)';
        } else if (speedRatio < 0.80) { // Naikkan threshold ke 0.80 agar lebih toleran
            segmentCong = 'moderate';
            segmentText = 'Moderate Flow';
            ringColor = 'var(--color-amber)';
        }
        
        // 4. Update current speed ring display (showing LIVE speed)
        docElements.segmentCurrentSpeed.innerText = actualSpeed.toFixed(1);
        
        // Show data source indicator under the speed value
        const predMethod15 = pred15?.prediction_method || '';
        const isLSTM15 = predMethod15 === 'live_lstm_runtime';
        const sourceLabel = document.getElementById('segment-speed-source');
        if (sourceLabel) {
            sourceLabel.innerText = liveSpeedVal != null ? 'Live Speed' : 'Estimated Base';
            sourceLabel.style.color = liveSpeedVal != null ? 'var(--text-primary)' : 'var(--color-amber)';
        }
        
        const pctConf = (confidenceScore * 100).toFixed(0);
        docElements.segmentConfidence.innerText = `${pctConf}%`;
        docElements.segmentConfidenceBar.style.width = `${pctConf}%`;
        
        // Tampilkan Prediksi di last update field
        docElements.segmentLastUpdate.innerText = isLSTM15 
            ? `Pred +15m: ${displaySpeedVal.toFixed(1)} km/h` 
            : 'Pred: N/A';
        
        // 5. Set congestion badge & ring color
        const badge = docElements.segmentCongestion;
        badge.className = 'congestion-badge';
        badge.classList.add(segmentCong);
        badge.innerText = segmentText;
        
        const ring = docElements.speedRing;
        ring.style.borderTopColor = ringColor;
        
        // 6. Render prediction forecast cards & chart
        const yPredList = [];
        const yLowerList = [];
        const yUpperList = [];
        const labelsList = ['15 min', '30 min', '45 min', '60 min'];
        
        horizons.forEach(h => {
            const details = forecasts[h];
            const summaryVal = document.getElementById(`pred-${h}m`);
            
            if (details) {
                summaryVal.innerText = `${details.predicted_speed.toFixed(1)} km/h`;
                yPredList.push(details.predicted_speed);
                yLowerList.push(details.uncertainty_lower);
                yUpperList.push(details.uncertainty_upper);
            } else {
                summaryVal.innerText = '--';
                const fallbackSpeed = liveSpeedVal ?? displaySpeedVal;
                yPredList.push(fallbackSpeed);
                yLowerList.push(fallbackSpeed * 0.8);
                yUpperList.push(fallbackSpeed * 1.2);
            }
        });
        
        // Re-draw forecast chart with live speed as baseline reference (garis putus-putus)
        renderForecastChart(labelsList, yPredList, yLowerList, yUpperList, liveSpeedVal);
        
    } catch (err) {
        console.error("Failed to load segment specifics:", err);
    }
}

// Forecast Graph in Chart.js with translucent bounds + current speed reference line
function renderForecastChart(labels, predictions, lowerBounds, upperBounds, currentSpeed) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Destroy previous Chart instance
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    const rootStyles = getComputedStyle(document.documentElement);
    const violetAccent = rootStyles.getPropertyValue('--violet-accent').trim() || '#8b5cf6';
    
    // Build current speed reference line (flat) if available
    const currentSpeedLine = currentSpeed != null
        ? labels.map(() => currentSpeed)
        : null;
    
    const datasets = [
        {
            label: 'Predicted Speed',
            data: predictions,
            borderColor: violetAccent,
            backgroundColor: 'rgba(139, 92, 246, 0.2)',

                    borderWidth: 3,
                    tension: 0.35,
                    fill: false,
                    z: 10
                },
                {
                    label: 'Upper Uncertainty Bound',
                    data: upperBounds,
                    borderColor: 'rgba(139, 92, 246, 0.15)',
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    tension: 0.35,
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'Lower Uncertainty Bound',
                    data: lowerBounds,
                    borderColor: 'rgba(139, 92, 246, 0.15)',
                    backgroundColor: 'rgba(139, 92, 246, 0.08)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    tension: 0.35,
                    fill: '-1', // Shade the area between lower bound and upper bound
                    pointRadius: 0
                },
                ...(currentSpeedLine ? [{
                    label: 'Current Speed (Now)',
                    data: currentSpeedLine,
                    borderColor: 'rgba(52, 211, 153, 0.7)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [4, 3],
                    tension: 0,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 0
                }] : [])
            ];
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false // Hide default legends
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(16, 22, 37, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(1) + ' km/h';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            family: 'Plus Jakarta Sans',
                            size: 10,
                            weight: '600'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#64748b',
                        font: {
                            family: 'Plus Jakarta Sans',
                            size: 10,
                            weight: '600'
                        }
                    },
                    suggestedMin: 5,
                    suggestedMax: 60
                }
            }
        }
    });
}


// ----------------------------------------------------
// MANUAL TRAFFIC OBSERVATIONS INGESTION
// ----------------------------------------------------

async function handleManualIngest(e) {
    e.preventDefault();
    
    // Lock submission button
    docElements.btnSubmitIngest.disabled = true;
    docElements.btnSubmitIngest.innerHTML = `<span class="spinner" style="width: 14px; height: 14px; display: inline-block;"></span> <span>Sending...</span>`;
    
    docElements.ingestAlertError.classList.add('hidden');
    docElements.ingestAlertSuccess.classList.add('hidden');
    
    const roadId = docElements.ingestRoadId.value;
    const speed = parseFloat(docElements.ingestSpeed.value);
    const confidence = parseFloat(docElements.ingestConfidence.value);
    
    if (!roadId || isNaN(speed) || isNaN(confidence)) {
        showIngestError("Please fill in all required form values.");
        resetIngestButton();
        return;
    }
    
    try {
        const payload = {
            records: [
                {
                    road_id: roadId,
                    current_speed: speed,
                    confidence: confidence,
                    timestamp: new Date().toISOString()
                }
            ]
        };
        
        const res = await fetchWithAuth(`${API_URL}/ingest/manual`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const errBody = await res.json();
            throw new Error(errBody.message || `HTTP ${res.status}`);
        }
        
        const result = await res.json();
        
        // Show Success Alert
        docElements.ingestAlertSuccess.classList.remove('hidden');
        docElements.formManualIngest.reset();
        
        // Instant Map Update
        refreshData();
        
        setTimeout(() => {
            docElements.ingestAlertSuccess.classList.add('hidden');
        }, 5000);
        
    } catch (err) {
        showIngestError(err.message || "Network request failed. Is the Render API active?");
    } finally {
        resetIngestButton();
    }
}

// Trigger bulk demo network traffic ingestion (all 50 segments populated randomly)
async function handleDemoNetworkIngest() {
    if (roadsData.length === 0) {
        alert("Roads details not loaded yet. Is the backend online?");
        return;
    }
    
    if (!confirm("This will generate and ingest simulated live traffic speed records (randomized between 10 km/h and 60 km/h) for all 50 Sukabumi segments, flushing the prediction cache. Continue?")) {
        return;
    }
    
    const oldBtnText = docElements.btnDemoIngest.innerText;
    docElements.btnDemoIngest.disabled = true;
    docElements.btnDemoIngest.innerText = "Simulating bulk network updates...";
    
    try {
        const payloadRecords = roadsData.map(road => {
            // Generate normal random speeds centered around weight limits
            const weightCoeff = road.road_weight || 1.0;
            const baseCenter = 35.0 * weightCoeff;
            const randSpeed = Math.max(5.0, Math.min(100.0, baseCenter + (Math.random() * 24 - 12)));
            
            return {
                road_id: road.road_id,
                current_speed: parseFloat(randSpeed.toFixed(1)),
                confidence: parseFloat((0.85 + (Math.random() * 0.15)).toFixed(2)),
                timestamp: new Date().toISOString()
            };
        });
        
        const res = await fetchWithAuth(`${API_URL}/ingest/manual`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ records: payloadRecords })
        });
        
        if (!res.ok) {
            const errBody = await res.json();
            throw new Error(errBody.message || `HTTP ${res.status}`);
        }
        
        alert(`Simulated traffic network updated successfully! Ingested speeds for all 50 roads.`);
        
        // Refresh map visual
        refreshData();
        
    } catch (err) {
        alert("Simulation failed: " + err.message);
    } finally {
        docElements.btnDemoIngest.disabled = false;
        docElements.btnDemoIngest.innerText = oldBtnText;
    }
}

function showIngestError(msg) {
    docElements.ingestErrorText.innerText = msg;
    docElements.ingestAlertError.classList.remove('hidden');
}

function resetIngestButton() {
    docElements.btnSubmitIngest.disabled = false;
    docElements.btnSubmitIngest.innerHTML = `<i data-lucide="send"></i> <span>Ingest Speed Record</span>`;
    if (window.lucide) window.lucide.createIcons();
}

// ----------------------------------------------------
// SYSTEM HEALTH, API READINESS, AND DIALS
// ----------------------------------------------------

async function checkSystemStatus() {
    try {
        const res = await fetchWithAuth(`${API_URL}/ready`);
        if (!res.ok) throw new Error("Ready Check degraded");
        
        const data = await res.json();
        
        // Update Sidebar Indicators
        const dot = docElements.healthDot;
        const label = docElements.healthLabel;
        
        if (data.ready) {
            dot.className = "status-dot pulsing";
            label.innerText = "System Online";
        } else {
            dot.className = "status-dot degraded";
            label.innerText = "Degraded (No model)";
        }
        
        // Populate API readiness list card dynamically
        let rHTML = '';
        Object.entries(data.resources).forEach(([key, val]) => {
            const statusIcon = val.ready === true || val.status === 'available' || val.status === 'loaded' || val.status === 'configured' || val.available === true || val.healthy === true 
                ? '<span style="color: var(--color-green); display: flex; align-items: center; gap: 0.35rem;"><i data-lucide="check-circle2" style="width: 14px; height: 14px;"></i> Available</span>'
                : '<span style="color: var(--color-red); display: flex; align-items: center; gap: 0.35rem;"><i data-lucide="x-circle" style="width: 14px; height: 14px;"></i> Unavailable</span>';
            
            rHTML += `
                <div class="metric-item" style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color); font-size: 0.85rem;">
                    <span style="font-weight: 500; text-transform: capitalize;">${key.replace('_', ' ')}:</span>
                    <strong>${statusIcon}</strong>
                </div>
            `;
        });
        docElements.readyDetailsList.innerHTML = rHTML || '<p style="font-size: 0.8rem; color: var(--text-muted);">No system assets listed</p>';
        
        // Get model status
        const mRes = await fetchWithAuth(`${API_URL}/model/version`);
        if (mRes.ok) {
            const mData = await mRes.json();
            docElements.activeModelName.innerText = mData.model_loaded ? mData.model_version : "No model active";
        }
        
        // Fetch scheduler statuses
        await fetchSchedulerStatus();
        
        if (window.lucide) window.lucide.createIcons();
    } catch (err) {
        console.warn("Health check connection lost:", err);
        docElements.healthDot.className = "status-dot failed";
        docElements.healthLabel.innerText = "Connection Offline";
        docElements.activeModelName.innerText = "Offline";
    }
}

async function fetchSchedulerStatus() {
    try {
        const res = await fetchWithAuth(`${API_URL}/scheduler/status`);
        if (!res.ok) throw new Error("Scheduler offline");
        
        const data = await res.json();
        
        let sHTML = '';
        if (data.jobs && Object.keys(data.jobs).length > 0) {
            Object.entries(data.jobs).forEach(([name, status]) => {
                const badgeClass = status.enabled ? 'green' : 'red';
                const countText = `Runs: ${status.run_count} | Fails: ${status.failure_count}`;
                
                sHTML += `
                    <div style="padding: 0.6rem 0; border-bottom: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 0.2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem;">
                            <span style="font-weight: 600;">${name}</span>
                            <span class="badge" style="background: rgba(16, 185, 129, 0.1); color: var(--color-green); font-size: 0.7rem; padding: 0.15rem 0.4rem; border-radius: 4px;">Active</span>
                        </div>
                        <span style="font-size: 0.75rem; color: var(--text-muted);">${countText} (Interval: ${status.interval_seconds}s)</span>
                    </div>
                `;
            });
        } else {
            sHTML = '<p style="font-size: 0.8rem; color: var(--text-muted); padding: 1rem 0;">No active background scheduler jobs detected</p>';
        }
        
        docElements.schedulerDetailsList.innerHTML = sHTML;
    } catch (err) {
        docElements.schedulerDetailsList.innerHTML = '<p style="font-size: 0.8rem; color: var(--color-red); padding: 1rem 0;">Failed to fetch job statuses.</p>';
    }
}

async function fetchSystemAudits() {
    try {
        const res = await fetchWithAuth(`${API_URL}/data-quality`);
        if (!res.ok) throw new Error("Data quality failed");
        
        const data = await res.json();
        
        // Data Completeness & Quality Donut
        const completenessScore = data.completeness * 100;
        docElements.valQualityScore.innerText = completenessScore.toFixed(0);
        docElements.auditCompleteness.innerText = `${completenessScore.toFixed(0)}%`;
        docElements.auditCompTxt.innerText = `${data.completeness * 50} / 50 segments synced`;
        
        // Count quality issues
        docElements.auditStaleCount.innerText = data.stale_roads.length;
        docElements.auditMissingCount.innerText = data.missing_roads.length;
        
        // Set drift status
        const driftBadge = docElements.auditDriftStatus;
        driftBadge.innerText = data.status === 'optimal' ? 'No Drift' : 'Action Required';
        driftBadge.style.background = data.status === 'optimal' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)';
        driftBadge.style.color = data.status === 'optimal' ? 'var(--color-green)' : 'var(--color-amber)';
        
        // Animate quality donut border styling using standard conic-gradient parameters
        const deg = (completenessScore / 100) * 360;
        docElements.qualityDonut.style.background = `conic-gradient(var(--violet-accent) 0deg ${deg}deg, rgba(255,255,255,0.05) ${deg}deg 360deg)`;
        
    } catch (err) {
        console.warn("Failed to fetch data quality stats:", err);
    }
}
