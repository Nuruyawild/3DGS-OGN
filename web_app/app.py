"""
Object-Goal Navigation Web 应用与可视化系统
所有 API 均使用 GET 方法以兼容云平台代理
"""

import os
import sys
import re
import json
import glob
import threading
import subprocess
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request, Response
from constants import coco_categories, scenes

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"ok": False, "msg": str(e)}), 500


# ============ 共享状态 ============
WEB_STATE_FILE = os.path.join(BASE_DIR, "tmp", "web_state.json")


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.agent_position = [0, 0, 0]
        self.agent_heading = 0
        self.goal_category = ""
        self.path_history = []
        self.gaussians = []
        self.scene_graph = {"nodes": [], "edges": []}
        self.metrics = {"sr": 0, "spl": 0, "dtg": 0}
        self.semantic_info = {}
        self.map_grid = []

    def update(self, data):
        with self.lock:
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def _load_from_file(self):
        """从 run_enhanced 写入的文件读取状态（训练/评估运行时）"""
        try:
            if os.path.exists(WEB_STATE_FILE):
                with open(WEB_STATE_FILE) as f:
                    d = json.load(f)
                with self.lock:
                    self.agent_position = d.get("agent_position", self.agent_position)
                    self.agent_heading = d.get("agent_heading", self.agent_heading)
                    self.path_history = d.get("path_history", [])
                    self.gaussians = d.get("gaussians", [])
                    self.scene_graph = d.get("scene_graph", self.scene_graph)
                    self.metrics = d.get("metrics", self.metrics)
                    self.semantic_info = d.get("semantic_info", {})
                    self.map_grid = d.get("map_grid", [])
        except Exception:
            pass

    def get(self):
        self._load_from_file()
        with self.lock:
            return {
                "agent_position": self.agent_position,
                "agent_heading": self.agent_heading,
                "goal_category": self.goal_category,
                "path_history": self.path_history[-200:],
                "gaussians": self.gaussians[:500],
                "scene_graph": self.scene_graph,
                "metrics": self.metrics,
                "semantic_info": self.semantic_info,
                "map_grid": self.map_grid,
            }


state = AppState()


# ============ 任务运行器 ============
class JobRunner:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.logs = deque(maxlen=5000)
        self.metrics = {}
        self.status = "idle"
        self.job_type = ""

    def _parse(self, line):
        m = re.search(r"ObjectNav succ/spl/dtg:\s*([\d.]+)/([\d.]+)/([\d.]+)", line)
        if m:
            self.metrics.update(success=float(m.group(1)), spl=float(m.group(2)), dtg=float(m.group(3)))
        m = re.search(r"FPS\s+(\d+)", line)
        if m:
            self.metrics["fps"] = int(m.group(1))
        m = re.search(r"num timesteps\s+(\d+)", line)
        if m:
            self.metrics["steps"] = int(m.group(1))

    def _read(self, stream):
        for raw in iter(stream.readline, b""):
            if not raw:
                break
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                line = str(raw)
            with self.lock:
                self.logs.append(line.rstrip())
                self._parse(line)

    def _run_proc(self, cmd):
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            self._read(self.proc.stdout)
            self.proc.wait()
        except Exception as e:
            with self.lock:
                self.logs.append("[ERROR] Process failed: {}".format(str(e)))
        finally:
            with self.lock:
                self.status = "idle"

    def start_train(self, exp_name, n_proc=2):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False, "已有任务在运行"
            self.logs.clear()
            self.metrics.clear()
            self.status = "running"
            self.job_type = "train"
        cmd = [sys.executable, "-u", os.path.join(BASE_DIR, "run_enhanced.py"),
               "-d", os.path.join(BASE_DIR, "tmp"), "--exp_name", exp_name,
               "-n", str(n_proc), "--split", "train"]
        threading.Thread(target=self._run_proc, args=(cmd,), daemon=True).start()
        return True, "训练已启动"

    def start_eval(self, exp_name, model_path, n_ep=50, save_fig=0):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False, "已有任务在运行"
            if not model_path:
                return False, "请选择模型"
            self.logs.clear()
            self.metrics.clear()
            self.status = "running"
            self.job_type = "eval"
        cmd = [sys.executable, "-u", os.path.join(BASE_DIR, "run_enhanced.py"),
               "-d", os.path.join(BASE_DIR, "tmp"), "--exp_name", exp_name,
               "--split", "val", "--eval", "1", "--load", model_path,
               "--num_eval_episodes", str(n_ep)]
        if save_fig:
            cmd.extend(["--save_paper_figures", "1"])
        threading.Thread(target=self._run_proc, args=(cmd,), daemon=True).start()
        return True, "评估已启动"

    def stop(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                except Exception:
                    pass
                self.status = "idle"
                self.logs.append("[INFO] Job stopped by user")
                return True, "已停止"
            self.status = "idle"
        return False, "无运行中任务"

    def get_status(self):
        with self.lock:
            lines = list(self.logs)
        return {
            "status": self.status,
            "job_type": self.job_type,
            "metrics": dict(self.metrics),
            "logs": lines[-300:] if len(lines) > 300 else lines,
        }


runner = JobRunner()


# ============ 路由（全部使用 GET 以兼容云平台代理） ============
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(state.get())


@app.route("/api/state/update")
def api_state_update():
    d = request.args.to_dict()
    if d:
        state.update(d)
    return jsonify({"ok": True})


@app.route("/api/categories")
def api_categories():
    return jsonify({"categories": [{"id": v, "name": k} for k, v in coco_categories.items()]})


@app.route("/api/goal/set")
def api_goal_set():
    cat = request.args.get("category", "")
    if cat:
        state.goal_category = cat
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "No category"}), 400


@app.route("/api/jobs/start")
def api_jobs_start():
    t = request.args.get("type", "train")
    if t == "train":
        ok, msg = runner.start_train(
            request.args.get("exp_name", "web_exp"),
            int(request.args.get("n_proc", 2)),
        )
    elif t == "eval":
        ok, msg = runner.start_eval(
            request.args.get("exp_name", "web_eval"),
            request.args.get("model_path", ""),
            int(request.args.get("n_ep", 50)),
            int(request.args.get("save_fig", 0)),
        )
    else:
        return jsonify({"ok": False, "msg": "未知类型"})
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/jobs/stop")
def api_jobs_stop():
    ok, msg = runner.stop()
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/jobs/status")
def api_jobs_status():
    return jsonify(runner.get_status())


@app.route("/api/jobs/models")
def api_jobs_models():
    models = []
    for d in [os.path.join(BASE_DIR, "tmp", "models"), os.path.join(BASE_DIR, "tmp", "dump")]:
        if os.path.exists(d):
            for exp in os.listdir(d):
                p = os.path.join(d, exp)
                if os.path.isdir(p):
                    for f in glob.glob(os.path.join(p, "*.pth")):
                        models.append({"path": f, "label": "{}/{}".format(exp, os.path.basename(f))})
    pt = os.path.join(BASE_DIR, "pretrained_models", "sem_exp.pth")
    if os.path.exists(pt):
        models.insert(0, {"path": pt, "label": "pretrained/sem_exp.pth"})
    return jsonify({"models": models})


@app.route("/api/experiments")
def api_experiments():
    exps = []
    rd = os.path.join(BASE_DIR, "tmp", "dump")
    try:
        os.makedirs(rd, exist_ok=True)
    except OSError:
        pass
    if os.path.exists(rd):
        for name in sorted(os.listdir(rd)):
            p = os.path.join(rd, name)
            if not os.path.isdir(p):
                continue
            info = {"name": name, "created": datetime.fromtimestamp(os.path.getctime(p)).isoformat()}
            mf = os.path.join(p, "enhanced_metrics.json")
            if os.path.exists(mf):
                with open(mf) as f:
                    info["metrics"] = json.load(f)
            exps.append(info)
    return jsonify(exps)


@app.route("/api/experiments/<name>")
def api_experiment_detail(name):
    p = os.path.join(BASE_DIR, "tmp", "dump", name)
    if not os.path.exists(p):
        return jsonify({"error": "Not found"}), 404
    info = {"name": name}
    for f in os.listdir(p):
        if f.endswith(".json"):
            with open(os.path.join(p, f)) as fp:
                info[f.replace(".json", "")] = json.load(fp)
    return jsonify(info)


@app.route("/api/experiments/<name>/export")
def api_experiment_export(name):
    p = os.path.join(BASE_DIR, "tmp", "dump", name)
    if not os.path.exists(p):
        return jsonify({"error": "Not found"}), 404
    info = {"name": name, "exported_at": datetime.now().isoformat()}
    for f in os.listdir(p):
        if f.endswith(".json"):
            with open(os.path.join(p, f)) as fp:
                info[f.replace(".json", "")] = json.load(fp)
    return Response(
        json.dumps(info, indent=2, default=str),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename={}_export.json".format(name)},
    )


@app.route("/api/datasets")
def api_datasets():
    try:
        ds = {"gibson": {"train": list(scenes.get("train", [])), "val": list(scenes.get("val", []))}}
        mp = os.path.join(BASE_DIR, "data", "scene_datasets", "mp3d")
        if os.path.exists(mp):
            ds["mp3d"] = {"scenes": [x for x in os.listdir(mp) if os.path.isdir(os.path.join(mp, x))]}
    except Exception:
        ds = {"gibson": {"train": [], "val": []}}
    return jsonify(ds)


@app.route("/api/datasets/stats")
def api_datasets_stats():
    stats = {k: {"id": v, "count": 0} for k, v in coco_categories.items()}
    rd = os.path.join(BASE_DIR, "tmp", "dump")
    if os.path.exists(rd):
        try:
            for exp in os.listdir(rd):
                exp_dir = os.path.join(rd, exp)
                if not os.path.isdir(exp_dir):
                    continue
                for fp in glob.glob(os.path.join(exp_dir, "*_spl_per_cat*.json")):
                    try:
                        with open(fp) as f:
                            for k, vals in json.load(f).items():
                                if k in stats:
                                    stats[k]["count"] += len(vals)
                    except Exception:
                        pass
        except Exception:
            pass
    return jsonify(stats)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print("Web App: http://localhost:{}".format(args.port))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
