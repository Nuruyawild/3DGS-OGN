/**
 * Object-Goal Navigation Web App - 3D 可视化与图表模块
 * 提供: 3D 高斯场景、导航路径、语义地图、指标图表
 */
(function() {
    'use strict';
    try {

    var vizScene = null, vizCamera = null, vizRenderer = null, vizControls = null;
    var gaussianPoints = [], pathLine = null, floorMesh = null;
    var metricsChart = null;
    var viewMode = 'default';

    // ===== 3D 可视化 =====
    function initViz3D() {
        var container = document.getElementById('canvas3d');
        var hint = document.getElementById('viz3dHint');
        if (!container || !window.THREE) return;

        var width = Math.max(container.clientWidth || 600, 400);
        var height = Math.max(container.clientHeight || 450, 300);

        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1d28);

        var camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        camera.position.set(5, 5, 5);
        camera.lookAt(0, 0, 0);

        var renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        if (window.THREE.OrbitControls) {
            var controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            vizControls = controls;
        }

        var gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
        scene.add(gridHelper);

        vizScene = scene;
        vizCamera = camera;
        vizRenderer = renderer;

        window.addEventListener('resize', function() {
            if (!container || !vizRenderer || !vizCamera) return;
            var w = Math.max(container.clientWidth || 600, 400);
            var h = Math.max(container.clientHeight || 450, 300);
            vizCamera.aspect = w / h;
            vizCamera.updateProjectionMatrix();
            vizRenderer.setSize(w, h);
        });

        function animate() {
            requestAnimationFrame(animate);
            if (vizControls) vizControls.update();
            renderer.render(scene, camera);
        }
        animate();

        window._resetCamera = function() {
            if (vizCamera && vizControls) {
                vizCamera.position.set(5, 5, 5);
                vizCamera.lookAt(0, 0, 0);
                vizControls.target.set(0, 0, 0);
            }
        };

        window._setViewMode = function(mode) {
            viewMode = mode || 'default';
            if (vizCamera && mode === 'topdown') {
                vizCamera.position.set(0, 15, 0);
                vizCamera.lookAt(0, 0, 0);
                if (vizControls) vizControls.target.set(0, 0, 0);
            }
        };

        window._updateViz = function(data) {
            if (!vizScene || !window.THREE) return;
            if (hint) hint.style.display = 'none';

            var gaussians = data.gaussians || [];
            var pathHistory = data.path_history || [];
            var mapGrid = data.map_grid || [];

            if (floorMesh) { vizScene.remove(floorMesh); floorMesh = null; }
            if (mapGrid.length > 0) {
                var rows = mapGrid.length, cols = mapGrid[0] ? mapGrid[0].length : 0;
                if (cols > 0) {
                    var cvs = document.createElement('canvas');
                    cvs.width = cols;
                    cvs.height = rows;
                    var cctx = cvs.getContext('2d');
                    for (var r = 0; r < rows; r++) {
                        for (var c = 0; c < cols; c++) {
                            var v = mapGrid[r][c];
                            var col = MAP_PALETTE[Math.min(v, MAP_PALETTE.length - 1)] || MAP_PALETTE[0];
                            cctx.fillStyle = 'rgb(' + Math.round(col[0]*255) + ',' + Math.round(col[1]*255) + ',' + Math.round(col[2]*255) + ')';
                            cctx.fillRect(c, r, 1, 1);
                        }
                    }
                    var tex = new THREE.CanvasTexture(cvs);
                    tex.needsUpdate = true;
                    var planeGeom = new THREE.PlaneGeometry(24, 24);
                    var planeMat = new THREE.MeshBasicMaterial({
                        map: tex,
                        side: THREE.DoubleSide
                    });
                    floorMesh = new THREE.Mesh(planeGeom, planeMat);
                    floorMesh.rotation.x = -Math.PI / 2;
                    floorMesh.position.y = -0.5;
                    vizScene.add(floorMesh);
                }
            }

            // 更新高斯点
            gaussianPoints.forEach(function(m) { vizScene.remove(m); });
            gaussianPoints = [];

            var geom = new THREE.BufferGeometry();
            var positions = [];
            var colors = [];
            for (var i = 0; i < gaussians.length; i++) {
                var c = gaussians[i].center || [0, 0, 0];
                var col = gaussians[i].color || [0.5, 0.5, 0.5];
                positions.push(c[0], c[1], c[2]);
                colors.push(col[0], col[1], col[2]);
            }
            if (positions.length > 0) {
                geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                geom.computeBoundingSphere();
                var mat = new THREE.PointsMaterial({
                    size: 0.08, vertexColors: true, sizeAttenuation: true
                });
                var pts = new THREE.Points(geom, mat);
                vizScene.add(pts);
                gaussianPoints.push(pts);
            }

            // 更新路径线
            if (pathLine) vizScene.remove(pathLine);
            pathLine = null;
            if (pathHistory.length >= 2) {
                var pathGeom = new THREE.BufferGeometry();
                var pathPos = [];
                for (var j = 0; j < pathHistory.length; j++) {
                    var p = pathHistory[j];
                    pathPos.push(p[0] || 0, p[1] || 0, (p[2] || 0) * 0.01);
                }
                pathGeom.setAttribute('position', new THREE.Float32BufferAttribute(pathPos, 3));
                var pathMat = new THREE.LineBasicMaterial({ color: 0x4a9eff, linewidth: 2 });
                pathLine = new THREE.Line(pathGeom, pathMat);
                vizScene.add(pathLine);
            }
        }

        // 包装 _updateViz 以同时更新 2D 路径、语义地图和指标图表
        var _innerUpdateViz = window._updateViz;
        window._updateViz = function(d) {
            if (_innerUpdateViz) _innerUpdateViz(d);
            drawPathCanvas(d);
            drawSemanticCanvas(d);
            if (d && d.metrics && window._updateMetricsChart) window._updateMetricsChart(d.metrics);
        };
    }

    // ===== 导航监控 - 指标图表 =====
    function initMetricsChart() {
        var canvas = document.getElementById('chartMetrics');
        var hint = document.getElementById('chartHint');
        if (!canvas || !window.Chart) return;

        var ctx = canvas.getContext('2d');
        var chartData = { labels: [], sr: [], spl: [], dtg: [] };
        var maxPoints = 50;

        metricsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: [
                    { label: 'SR', data: chartData.sr, borderColor: '#34d399', fill: false, tension: 0.3 },
                    { label: 'SPL', data: chartData.spl, borderColor: '#4a9eff', fill: false, tension: 0.3 },
                    { label: 'DTG', data: chartData.dtg, borderColor: '#f87171', fill: false, tension: 0.3 }
                ]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
                plugins: { legend: { display: true } },
                scales: {
                    x: { display: true, maxTicksLimit: 10 },
                    y: { min: 0, max: 1.2 }
                }
            }
        });

        window._updateMetricsChart = function(m) {
            if (!metricsChart || !m) return;
            if (hint) hint.style.display = 'none';
            var t = new Date().toLocaleTimeString();
            chartData.labels.push(t);
            chartData.sr.push(m.sr != null ? m.sr : 0);
            chartData.spl.push(m.spl != null ? m.spl : 0);
            chartData.dtg.push(m.dtg != null ? m.dtg : 0);
            if (chartData.labels.length > maxPoints) {
                chartData.labels.shift();
                chartData.sr.shift();
                chartData.spl.shift();
                chartData.dtg.shift();
            }
            metricsChart.data.labels = chartData.labels;
            metricsChart.data.datasets[0].data = chartData.sr;
            metricsChart.data.datasets[1].data = chartData.spl;
            metricsChart.data.datasets[2].data = chartData.dtg;
            metricsChart.update('none');
        };
    }

    var MAP_PALETTE = [
        [0.08, 0.09, 0.12],
        [0.25, 0.25, 0.28],
        [0.2, 0.4, 0.8], [0.8, 0.3, 0.2], [0.2, 0.8, 0.4], [0.8, 0.6, 0.2],
        [0.6, 0.2, 0.8], [0.3, 0.8, 0.8], [0.9, 0.5, 0.2], [0.4, 0.6, 0.9],
        [0.7, 0.9, 0.5], [0.5, 0.3, 0.7], [0.9, 0.7, 0.3], [0.3, 0.5, 0.9],
        [0.8, 0.4, 0.6], [0.5, 0.8, 0.6], [0.6, 0.5, 0.9], [0.9, 0.6, 0.4]
    ];

    function drawMapBackground(ctx, mapGrid, w, h, cellW, cellH, offX, offY) {
        if (!mapGrid || !mapGrid.length) return;
        var rows = mapGrid.length, cols = mapGrid[0] ? mapGrid[0].length : 0;
        if (!cols) return;
        var cw = cellW || w / cols, ch = cellH || h / rows;
        var ox = offX || 0, oy = offY || 0;
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                var v = mapGrid[r][c];
                var idx = Math.min(v, MAP_PALETTE.length - 1);
                var col = MAP_PALETTE[idx] || MAP_PALETTE[0];
                ctx.fillStyle = 'rgb(' + Math.round(col[0]*255) + ',' + Math.round(col[1]*255) + ',' + Math.round(col[2]*255) + ')';
                ctx.fillRect(ox + c * cw, oy + r * ch, cw + 1, ch + 1);
            }
        }
    }

    // ===== 导航路径 2D 画布 =====
    function drawPathCanvas(data) {
        var canvas = document.getElementById('pathCanvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var w = canvas.width, h = canvas.height;
        ctx.fillStyle = '#1a1d28';
        ctx.fillRect(0, 0, w, h);

        var mapGrid = data.map_grid || [];
        var path = data.path_history || [];

        if (mapGrid.length > 0) {
            drawMapBackground(ctx, mapGrid, w, h);
        }

        if (path.length < 2) return;

        var pad = 15;
        var mapRows = mapGrid.length, mapCols = mapGrid[0] ? mapGrid[0].length : 0;
        var mapSize = 24;
        var pathScale, pathOffX, pathOffY;
        if (mapRows > 0 && mapCols > 0) {
            var scale = Math.min((w - 2 * pad) / mapCols, (h - 2 * pad) / mapRows);
            pathOffX = (w - mapCols * scale) / 2;
            pathOffY = (h - mapRows * scale) / 2;
            pathScale = scale * mapCols / mapSize;
        } else {
            var xs = path.map(function(p) { return p[0]; });
            var ys = path.map(function(p) { return p[1]; });
            var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
            var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
            var rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
            pathScale = Math.min((w - 2 * pad) / rangeX, (h - 2 * pad) / rangeY);
            pathOffX = pad - minX * pathScale;
            pathOffY = pad - minY * pathScale;
        }

        ctx.strokeStyle = '#4a9eff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (var i = 0; i < path.length; i++) {
            var px = pathOffX + (path[i][0] || 0) * pathScale;
            var py = h - pathOffY - (path[i][1] || 0) * pathScale;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.fillStyle = '#34d399';
        if (path.length > 0) {
            var lx = pathOffX + (path[path.length - 1][0] || 0) * pathScale;
            var ly = h - pathOffY - (path[path.length - 1][1] || 0) * pathScale;
            ctx.beginPath();
            ctx.arc(lx, ly, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ===== 语义地图画布 =====
    function drawSemanticCanvas(data) {
        var canvas = document.getElementById('semCanvas');
        var heatCanvas = document.getElementById('heatCanvas');
        if (!canvas) return;

        var w = canvas.width, h = canvas.height;
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#1a1d28';
        ctx.fillRect(0, 0, w, h);

        var mapGrid = data.map_grid || [];
        var semInfo = data.semantic_info || {};
        var pathHistory = data.path_history || [];

        if (mapGrid.length > 0) {
            drawMapBackground(ctx, mapGrid, w, h);
        } else {
            var hasSem = false;
            for (var k in semInfo) { hasSem = true; break; }
            if (hasSem) {
                var palette = MAP_PALETTE.slice(2);
                var maxR = 0, maxC = 0;
                for (var key in semInfo) {
                    var parts = key.split('_');
                    var r = parseInt(parts[0], 10), c = parseInt(parts[1], 10);
                    if (r > maxR) maxR = r;
                    if (c > maxC) maxC = c;
                }
                var range = Math.max(maxR, maxC, 1);
                var cellSize = Math.min(w / range, h / range, 8);
                for (var key in semInfo) {
                    var parts = key.split('_');
                    var r = parseInt(parts[0], 10), c = parseInt(parts[1], 10);
                    var v = semInfo[key] || 0;
                    var col = palette[v % palette.length] || palette[0];
                    ctx.fillStyle = 'rgb(' + Math.round(col[0]*255) + ',' + Math.round(col[1]*255) + ',' + Math.round(col[2]*255) + ')';
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize + 1, cellSize + 1);
                }
            }
        }

        if (pathHistory.length >= 2) {
            var mapRows = mapGrid.length, mapCols = mapGrid[0] ? mapGrid[0].length : 0;
            var pathScale, pathOffX, pathOffY;
            if (mapRows > 0 && mapCols > 0) {
                var cellScale = Math.min(w / mapCols, h / mapRows);
                pathScale = cellScale * mapCols / 24;
                pathOffX = (w - mapCols * cellScale) / 2;
                pathOffY = (h - mapRows * cellScale) / 2;
            } else {
                var scale = 20;
                var pathCells = pathHistory.map(function(p) {
                    return [p[1] * scale, p[0] * scale];
                });
                var xs = pathCells.map(function(p) { return p[0]; });
                var ys = pathCells.map(function(p) { return p[1]; });
                var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
                var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
                var rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
                pathScale = Math.min((w - 4) / rangeX, (h - 4) / rangeY);
                pathOffX = 2 - minX * pathScale;
                pathOffY = 2 - minY * pathScale;
            }
            ctx.strokeStyle = '#4a9eff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (var i = 0; i < pathHistory.length; i++) {
                var px, py;
                if (mapRows > 0 && mapCols > 0) {
                    px = pathOffX + (pathHistory[i][0] || 0) * pathScale;
                    py = h - pathOffY - (pathHistory[i][1] || 0) * pathScale;
                } else {
                    var pc = [pathHistory[i][1] * 20, pathHistory[i][0] * 20];
                    px = pathOffX + pc[0] * pathScale;
                    py = h - pathOffY - pc[1] * pathScale;
                }
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        if (heatCanvas && (mapGrid.length > 0 || (function(){ for(var k in semInfo) return true; return false; })())) {
            var hctx = heatCanvas.getContext('2d');
            var hw = heatCanvas.width, hh = heatCanvas.height;
            hctx.fillStyle = '#1a1d28';
            hctx.fillRect(0, 0, hw, hh);
            var count = {};
            if (mapGrid.length > 0) {
                for (var ri = 0; ri < mapGrid.length; ri++) {
                    for (var ci = 0; ci < (mapGrid[ri] || []).length; ci++) {
                        var v = mapGrid[ri][ci];
                        if (v >= 2) count[v - 2] = (count[v - 2] || 0) + 1;
                    }
                }
            }
            for (var k in semInfo) {
                var v = semInfo[k];
                count[v] = (count[v] || 0) + 1;
            }
            var vals = [];
            for (var kk in count) vals.push(count[kk]);
            var maxCount = (vals.length > 0 ? Math.max.apply(null, vals) : 0) || 1;
            var entries = [];
            for (var kk in count) entries.push([kk, count[kk]]);
            var barH = hh / (entries.length + 1);
            for (var i = 0; i < entries.length; i++) {
                var ent = entries[i];
                var val = ent[1] / maxCount;
                hctx.fillStyle = '#4a9eff';
                hctx.fillRect(0, i * barH, hw * val, barH - 2);
                hctx.fillStyle = '#e4e6f0';
                hctx.font = '12px sans-serif';
                hctx.fillText('Cat ' + ent[0] + ': ' + ent[1], 5, i * barH + barH / 2);
            }
        }
    }

    // ===== 数据集类别图表 =====
    function initCatChart() {
        var canvas = document.getElementById('chartCat');
        if (!canvas || !window.Chart) return;

        var ctx = canvas.getContext('2d');
        window._updateCatChart = function(stats) {
            if (!window.Chart || !stats) return;
            var keys = Object.keys(stats);
            if (keys.length === 0) return;

            if (window._catChartInstance) window._catChartInstance.destroy();
            window._catChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: keys.slice(0, 10),
                    datasets: [{
                        label: 'Count',
                        data: keys.slice(0, 10).map(function(k) { return stats[k].count || 0; }),
                        backgroundColor: 'rgba(74, 158, 255, 0.6)'
                    }]
                },
                options: {
                    responsive: false,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        };
    }

    // ===== 初始化 =====
    document.addEventListener('DOMContentLoaded', function() {
        try {
            initViz3D();
            initMetricsChart();
            initCatChart();
        } catch (e) {
            console.warn('Viz init:', e);
        }
    });
    } catch (e) {
        console.warn('App.js load error:', e);
    }
})();
