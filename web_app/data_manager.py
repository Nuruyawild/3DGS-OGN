"""
Data Management Tools for Object-Goal Navigation.

Implements:
- Experiment result storage and querying
- Result comparison and analysis
- Data export functionality
- Dataset browsing for Gibson and MP3D
- Object category statistics
"""

import os
import sys
import json
import csv
import glob
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import coco_categories, scenes


class ExperimentManager:
    """Manages experiment results: storage, querying, comparison, and export."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'tmp', 'dump')
        self.models_dir = os.path.join(base_dir, 'tmp', 'models')

    def list_experiments(self) -> List[Dict]:
        """List all experiments with metadata."""
        experiments = []
        if not os.path.exists(self.results_dir):
            return experiments

        for exp_name in sorted(os.listdir(self.results_dir)):
            exp_dir = os.path.join(self.results_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue
            experiments.append(self._load_experiment_info(exp_name))
        return experiments

    def _load_experiment_info(self, exp_name: str) -> Dict:
        """Load detailed info for a single experiment."""
        exp_dir = os.path.join(self.results_dir, exp_name)
        info = {
            'name': exp_name,
            'path': exp_dir,
            'created': datetime.fromtimestamp(
                os.path.getctime(exp_dir)).isoformat(),
            'metrics': None,
            'spl_per_category': {},
            'success_per_category': {},
            'model_checkpoints': [],
            'num_episodes_dirs': 0,
            'config': {}
        }

        metrics_file = os.path.join(exp_dir, 'enhanced_metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                info['metrics'] = json.load(f)

        for spl_file in glob.glob(os.path.join(exp_dir, '*_spl_per_cat*.json')):
            with open(spl_file, 'r') as f:
                data = json.load(f)
            split_name = os.path.basename(spl_file).split('_')[0]
            info['spl_per_category'][split_name] = data

        for succ_file in glob.glob(os.path.join(exp_dir, '*_success_per_cat*.json')):
            with open(succ_file, 'r') as f:
                data = json.load(f)
            split_name = os.path.basename(succ_file).split('_')[0]
            info['success_per_category'][split_name] = data

        info['model_checkpoints'] = [
            os.path.basename(f) for f in
            glob.glob(os.path.join(exp_dir, '*.pth'))
        ]

        model_dir = os.path.join(self.models_dir, exp_name)
        if os.path.exists(model_dir):
            info['model_checkpoints'] += [
                os.path.basename(f) for f in
                glob.glob(os.path.join(model_dir, '*.pth'))
            ]

        episodes_dir = os.path.join(exp_dir, 'episodes')
        if os.path.exists(episodes_dir):
            info['num_episodes_dirs'] = len(os.listdir(episodes_dir))

        return info

    def get_experiment(self, exp_name: str) -> Optional[Dict]:
        """Get detailed results for a specific experiment."""
        exp_dir = os.path.join(self.results_dir, exp_name)
        if not os.path.exists(exp_dir):
            return None
        return self._load_experiment_info(exp_name)

    def compare_experiments(self, exp_names: List[str]) -> Dict:
        """Compare multiple experiments side by side."""
        comparison = {
            'experiments': [],
            'metrics_comparison': {},
            'category_comparison': {}
        }

        for name in exp_names:
            info = self.get_experiment(name)
            if info:
                comparison['experiments'].append(info)

        metric_keys = ['success', 'spl', 'dtg', 'num_episodes']
        for key in metric_keys:
            comparison['metrics_comparison'][key] = {}
            for exp in comparison['experiments']:
                val = 0
                if exp['metrics'] and key in exp['metrics']:
                    val = exp['metrics'][key]
                comparison['metrics_comparison'][key][exp['name']] = val

        all_categories = set()
        for exp in comparison['experiments']:
            for split_data in exp['spl_per_category'].values():
                all_categories.update(split_data.keys())

        for cat in all_categories:
            comparison['category_comparison'][cat] = {}
            for exp in comparison['experiments']:
                spl_vals = []
                succ_vals = []
                for split_data in exp['spl_per_category'].values():
                    if cat in split_data:
                        spl_vals.extend(split_data[cat])
                for split_data in exp['success_per_category'].values():
                    if cat in split_data:
                        succ_vals.extend(split_data[cat])

                comparison['category_comparison'][cat][exp['name']] = {
                    'avg_spl': sum(spl_vals) / len(spl_vals) if spl_vals else 0,
                    'avg_success': sum(succ_vals) / len(succ_vals) if succ_vals else 0,
                    'count': len(spl_vals)
                }

        return comparison

    def export_experiment(self, exp_name: str, output_path: str,
                          format: str = 'json') -> str:
        """Export experiment results to file."""
        info = self.get_experiment(exp_name)
        if info is None:
            raise ValueError(f"Experiment '{exp_name}' not found")

        info['exported_at'] = datetime.now().isoformat()
        info.pop('path', None)

        if format == 'json':
            if not output_path.endswith('.json'):
                output_path += '.json'
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)

        elif format == 'csv':
            if not output_path.endswith('.csv'):
                output_path += '.csv'
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Name', info['name']])
                writer.writerow(['Created', info['created']])
                if info['metrics']:
                    for k, v in info['metrics'].items():
                        writer.writerow([k, v])
                writer.writerow([])
                writer.writerow(['Category', 'Avg SPL', 'Avg Success'])
                for split_data in info['spl_per_category'].values():
                    for cat, vals in split_data.items():
                        avg_spl = sum(vals) / len(vals) if vals else 0
                        succ_vals = []
                        for s_data in info['success_per_category'].values():
                            if cat in s_data:
                                succ_vals.extend(s_data[cat])
                        avg_succ = sum(succ_vals) / len(succ_vals) if succ_vals else 0
                        writer.writerow([cat, f"{avg_spl:.4f}", f"{avg_succ:.4f}"])

        return output_path

    def save_run_results(self, exp_name: str, metrics: Dict,
                         config: Dict = None):
        """Save results from a training/eval run."""
        exp_dir = os.path.join(self.results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'timestamp': timestamp,
            'metrics': metrics,
            'config': config or {}
        }

        results_file = os.path.join(exp_dir, f'results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        enhanced_file = os.path.join(exp_dir, 'enhanced_metrics.json')
        with open(enhanced_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return results_file


class DatasetBrowser:
    """Browse and analyze Gibson and MP3D datasets."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')

    def get_available_datasets(self) -> Dict:
        """Get list of available datasets and their scenes."""
        datasets = {
            'gibson': {
                'train': scenes.get('train', []),
                'val': scenes.get('val', []),
                'total_scenes': len(scenes.get('train', [])) +
                                len(scenes.get('val', []))
            }
        }

        mp3d_path = os.path.join(self.data_dir, 'scene_datasets', 'mp3d')
        if os.path.exists(mp3d_path):
            mp3d_scenes = [d for d in os.listdir(mp3d_path)
                           if os.path.isdir(os.path.join(mp3d_path, d))]
            datasets['mp3d'] = {
                'scenes': mp3d_scenes,
                'total_scenes': len(mp3d_scenes)
            }

        return datasets

    def get_scene_info(self, dataset: str, scene_name: str) -> Dict:
        """Get information about a specific scene."""
        info = {
            'name': scene_name,
            'dataset': dataset,
            'exists': False,
            'files': []
        }

        if dataset == 'gibson':
            scene_path = os.path.join(
                self.data_dir, 'scene_datasets', 'gibson_semantic', scene_name)
        elif dataset == 'mp3d':
            scene_path = os.path.join(
                self.data_dir, 'scene_datasets', 'mp3d', scene_name)
        else:
            return info

        if os.path.exists(scene_path):
            info['exists'] = True
            info['files'] = os.listdir(scene_path)
            info['size_mb'] = sum(
                os.path.getsize(os.path.join(scene_path, f))
                for f in info['files'] if os.path.isfile(
                    os.path.join(scene_path, f))
            ) / (1024 * 1024)

        return info

    def get_category_statistics(self) -> Dict:
        """Get object category statistics across all datasets."""
        stats = {}
        for cat_name, cat_id in coco_categories.items():
            stats[cat_name] = {
                'id': cat_id,
                'name': cat_name,
                'observation_count': 0,
                'scene_count': 0,
                'avg_spl': 0.0,
                'avg_success': 0.0,
            }

        results_dir = os.path.join(self.base_dir, 'tmp', 'dump')
        if os.path.exists(results_dir):
            for exp_name in os.listdir(results_dir):
                exp_dir = os.path.join(results_dir, exp_name)
                for spl_file in glob.glob(
                        os.path.join(exp_dir, '*_spl_per_cat*.json')):
                    with open(spl_file, 'r') as f:
                        spl_data = json.load(f)
                    for cat_name, values in spl_data.items():
                        if cat_name in stats:
                            stats[cat_name]['observation_count'] += len(values)
                            if values:
                                current_avg = stats[cat_name]['avg_spl']
                                new_avg = sum(values) / len(values)
                                stats[cat_name]['avg_spl'] = (
                                    current_avg + new_avg) / 2 if current_avg > 0 else new_avg

                for succ_file in glob.glob(
                        os.path.join(exp_dir, '*_success_per_cat*.json')):
                    with open(succ_file, 'r') as f:
                        succ_data = json.load(f)
                    for cat_name, values in succ_data.items():
                        if cat_name in stats and values:
                            current_avg = stats[cat_name]['avg_success']
                            new_avg = sum(values) / len(values)
                            stats[cat_name]['avg_success'] = (
                                current_avg + new_avg) / 2 if current_avg > 0 else new_avg

        return stats

    def get_episode_data(self, dataset: str, split: str) -> Dict:
        """Get episode data for a dataset split."""
        ep_path = os.path.join(
            self.data_dir, 'datasets', 'objectnav',
            dataset, 'v1.1', split)

        if not os.path.exists(ep_path):
            ep_path = os.path.join(
                self.data_dir, 'datasets', 'objectnav',
                dataset, 'v1', split)

        result = {'path': ep_path, 'exists': os.path.exists(ep_path), 'files': []}

        if result['exists']:
            result['files'] = os.listdir(ep_path)

        return result


def main():
    """CLI tool for data management."""
    import argparse

    parser = argparse.ArgumentParser(description='Data Management Tool')
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List experiments')

    detail_parser = subparsers.add_parser('detail', help='Show experiment detail')
    detail_parser.add_argument('exp_name', type=str)

    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiments', nargs='+', type=str)

    export_parser = subparsers.add_parser('export', help='Export experiment')
    export_parser.add_argument('exp_name', type=str)
    export_parser.add_argument('--output', '-o', type=str, default=None)
    export_parser.add_argument('--format', '-f', choices=['json', 'csv'],
                               default='json')

    datasets_parser = subparsers.add_parser('datasets', help='Browse datasets')
    stats_parser = subparsers.add_parser('stats', help='Category statistics')

    args = parser.parse_args()

    exp_mgr = ExperimentManager()
    ds_browser = DatasetBrowser()

    if args.command == 'list':
        experiments = exp_mgr.list_experiments()
        if not experiments:
            print("No experiments found.")
            return
        print(f"\n{'Name':<30} {'Success':>10} {'SPL':>10} {'DTG':>10} {'Episodes':>10}")
        print("-" * 75)
        for exp in experiments:
            m = exp.get('metrics') or {}
            print(f"{exp['name']:<30} "
                  f"{m.get('success', 0):>10.3f} "
                  f"{m.get('spl', 0):>10.3f} "
                  f"{m.get('dtg', 0):>10.3f} "
                  f"{m.get('num_episodes', 0):>10}")

    elif args.command == 'detail':
        exp = exp_mgr.get_experiment(args.exp_name)
        if exp is None:
            print(f"Experiment '{args.exp_name}' not found.")
            return
        print(json.dumps(exp, indent=2, default=str))

    elif args.command == 'compare':
        comparison = exp_mgr.compare_experiments(args.experiments)
        print(json.dumps(comparison, indent=2, default=str))

    elif args.command == 'export':
        output = args.output or f"{args.exp_name}_export"
        path = exp_mgr.export_experiment(args.exp_name, output, args.format)
        print(f"Exported to: {path}")

    elif args.command == 'datasets':
        datasets = ds_browser.get_available_datasets()
        print(json.dumps(datasets, indent=2))

    elif args.command == 'stats':
        stats = ds_browser.get_category_statistics()
        print(f"\n{'Category':<20} {'ID':>5} {'Observations':>15} "
              f"{'Avg SPL':>10} {'Avg Success':>12}")
        print("-" * 70)
        for name, data in stats.items():
            print(f"{name:<20} {data['id']:>5} "
                  f"{data['observation_count']:>15} "
                  f"{data['avg_spl']:>10.3f} "
                  f"{data['avg_success']:>12.3f}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
