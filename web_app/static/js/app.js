/*
 * 可视化增强 — 仅负责 3D 场景和 Chart 图表
 * 如果 THREE 或 Chart 未加载，此文件静默跳过，不影响核心功能
 */
(function() {
    var scene, camera, renderer, controls, gGroup, pathLine, agentMesh;
    var metricsChart, catChart;
    var COLORS = ['#f0a030','#30f060','#30c0f0','#8030f0','#f03098','#f03050','#a0f030','#30e8f0'];

    function init3D() {
        if (typeof THREE === 'undefined') return;
        var c = document.getElementById('canvas3d');
        if (!c) return;
        var w = c.clientWidth || 600, h = c.clientHeight || 450;

        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f1117);
        camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
        camera.position.set(5, 8, 5);
        camera.lookAt(0, 0, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(w, h);
        c.appendChild(renderer.domElement);

        if (THREE.OrbitControls) {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
        }

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        var dl = new THREE.DirectionalLight(0xffffff, 0.8);
        dl.position.set(5, 10, 5);
        scene.add(dl);
        scene.add(new THREE.GridHelper(20, 40, 0x333333, 0x222222));

        gGroup = new THREE.Group();
        scene.add(gGroup);
        pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x34d399 }));
        scene.add(pathLine);

        var cone = new THREE.Mesh(
            new THREE.ConeGeometry(0.15, 0.4, 8),
            new THREE.MeshPhongMaterial({ color: 0x4a9eff })
        );
        cone.rotation.x = Math.PI;
        agentMesh = cone;
        scene.add(agentMesh);

        window.addEventListener('resize', onResize);
        (function animate() {
            requestAnimationFrame(animate);
            if (controls) controls.update();
            if (renderer && scene && camera) renderer.render(scene, camera);
        })();
    }

    function onResize() {
        var c = document.getElementById('canvas3d');
        if (!c || !renderer || !camera) return;
        var w = c.clientWidth || 600, h = c.clientHeight || 450;
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    window._resetCamera = function() {
        if (camera) { camera.position.set(5, 8, 5); camera.lookAt(0, 0, 0); }
    };

    window._setViewMode = function(v) {
        if (v === 'topdown' && camera) { camera.position.set(0, 15, 0); camera.lookAt(0, 0, 0); }
    };

    function initCharts() {
        if (typeof Chart === 'undefined') return;
        var mc = document.getElementById('chartMetrics');
        if (mc) {
            metricsChart = new Chart(mc.getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [
                    { label: 'SR', data: [], borderColor: '#34d399', fill: false },
                    { label: 'SPL', data: [], borderColor: '#4a9eff', fill: false },
                    { label: 'DTG', data: [], borderColor: '#f87171', fill: false }
                ]},
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
        var cc = document.getElementById('chartCat');
        if (cc) {
            catChart = new Chart(cc.getContext('2d'), {
                type: 'bar',
                data: { labels: [], datasets: [{ label: 'Count', data: [], backgroundColor: '#4a9eff' }] },
                options: { responsive: true }
            });
        }
    }

    window._updateViz = function(d) {
        if (agentMesh && d.agent_position) {
            agentMesh.position.set(d.agent_position[0], d.agent_position[2] || 0, d.agent_position[1]);
        }
        updateGaussians(d.gaussians || []);
        updatePath(d.path_history || []);
        drawPath(d.path_history || []);
        drawSemantic(d.semantic_info || {});
        drawHeatmap(d.semantic_info || {});
    };

    function updateGaussians(gs) {
        if (!gGroup) return;
        while (gGroup.children.length) gGroup.remove(gGroup.children[0]);
        var count = Math.min(gs.length, 300);
        for (var i = 0; i < count; i++) {
            var g = gs[i];
            var pos = g.center || g.position || [0, 0, 0];
            var geo = new THREE.SphereGeometry(0.08, 8, 6);
            var col = g.color || [0.5, 0.5, 0.5];
            var mat = new THREE.MeshPhongMaterial({ color: new THREE.Color(col[0], col[1], col[2]) });
            var m = new THREE.Mesh(geo, mat);
            m.position.set(pos[0], pos[2] || 0, pos[1]);
            gGroup.add(m);
        }
    }

    function updatePath(ph) {
        if (!pathLine || !ph || ph.length < 2) return;
        var pts = [];
        for (var i = 0; i < ph.length; i++) {
            pts.push(new THREE.Vector3(ph[i][0] || 0, ph[i][2] || 0, ph[i][1] || 0));
        }
        pathLine.geometry.setFromPoints(pts);
        pathLine.geometry.attributes.position.needsUpdate = true;
    }

    function drawPath(ph) {
        var c = document.getElementById('pathCanvas');
        if (!c) return;
        var ctx = c.getContext('2d');
        ctx.fillStyle = '#1a1d28';
        ctx.fillRect(0, 0, c.width, c.height);
        if (!ph || ph.length < 2) return;
        var pad = 30;
        var xs = [], ys = [];
        for (var i = 0; i < ph.length; i++) { xs.push(ph[i][0]); ys.push(ph[i][1]); }
        var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
        var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
        var r = Math.max(maxX - minX, maxY - minY, 1);
        var scale = (Math.min(c.width, c.height) - 2 * pad) / r;
        function toX(x) { return pad + (x - minX) * scale; }
        function toY(y) { return c.height - pad - (y - minY) * scale; }
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(toX(ph[0][0]), toY(ph[0][1]));
        for (var i = 1; i < ph.length; i++) ctx.lineTo(toX(ph[i][0]), toY(ph[i][1]));
        ctx.stroke();
        ctx.fillStyle = '#4a9eff';
        ctx.beginPath();
        ctx.arc(toX(ph[0][0]), toY(ph[0][1]), 5, 0, 2 * Math.PI);
        ctx.fill();
    }

    function drawSemantic(sem) {
        var cv = document.getElementById('semCanvas');
        if (!cv) return;
        var ctx = cv.getContext('2d');
        ctx.fillStyle = '#1a1d28';
        ctx.fillRect(0, 0, cv.width, cv.height);
        var keys = Object.keys(sem);
        if (keys.length === 0) {
            ctx.fillStyle = '#8b8fa3';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('运行导航以显示语义', cv.width / 2, cv.height / 2);
            return;
        }
        var cellSize = 8;
        for (var i = 0; i < keys.length; i++) {
            var parts = keys[i].split('_');
            var r = parseInt(parts[0]) || 0, col = parseInt(parts[1]) || 0;
            ctx.fillStyle = COLORS[(sem[keys[i]] || 0) % COLORS.length];
            ctx.fillRect(col * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
        }
    }

    function drawHeatmap(sem) {
        var cv = document.getElementById('heatCanvas');
        if (!cv) return;
        var ctx = cv.getContext('2d');
        ctx.fillStyle = '#1a1d28';
        ctx.fillRect(0, 0, cv.width, cv.height);
        var keys = Object.keys(sem);
        var cellSize = 12;
        for (var i = 0; i < keys.length; i++) {
            var parts = keys[i].split('_');
            var r = parseInt(parts[0]) || 0, col = parseInt(parts[1]) || 0;
            ctx.fillStyle = COLORS[(sem[keys[i]] || 0) % COLORS.length];
            ctx.fillRect(col * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
        }
    }

    try {
        init3D();
    } catch(e) {
        console.warn('[viz] 3D init skipped:', e.message || e);
    }

    try {
        initCharts();
    } catch(e) {
        console.warn('[viz] Chart init skipped:', e.message || e);
    }
})();
