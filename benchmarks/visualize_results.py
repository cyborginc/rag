#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization script for CyborgDB benchmark results.
Generates charts and graphs for the announcement blog.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Generate visualizations from benchmark results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load benchmark results from files"""
        # Try to load consolidated results first
        consolidated_path = self.results_dir / "consolidated_results.csv"
        if consolidated_path.exists():
            self.df = pd.read_csv(consolidated_path)
        else:
            # Load from raw JSON
            raw_path = self.results_dir / "raw_results.json"
            if raw_path.exists():
                with open(raw_path) as f:
                    data = json.load(f)
                self.df = pd.DataFrame(data)
            else:
                raise FileNotFoundError(f"No results found in {self.results_dir}")
    
    def create_latency_comparison_chart(self):
        """Create latency comparison bar chart"""
        latency_df = self.df[self.df['metric_type'] == 'query_latency']
        
        if latency_df.empty:
            print("No latency data found")
            return
        
        # Pivot data for plotting
        pivot = latency_df.pivot_table(
            values='value',
            index='metric_name',
            columns='vdb_type',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(kind='bar', ax=ax)
        
        ax.set_title('Query Latency Comparison: CyborgDB vs Milvus', fontsize=16, fontweight='bold')
        ax.set_xlabel('Latency Metric', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.legend(title='Vector Database', loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        output_path = self.results_dir / "latency_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved latency comparison chart to {output_path}")
    
    def create_qps_recall_chart(self):
        """Create QPS vs Recall scatter plot"""
        qps_df = self.df[self.df['metric_type'] == 'qps_recall']
        
        if qps_df.empty:
            print("No QPS/Recall data found")
            return
        
        fig = go.Figure()
        
        # Add traces for each VDB type
        for vdb in qps_df['vdb_type'].unique():
            vdb_data = qps_df[qps_df['vdb_type'] == vdb]
            
            # Extract QPS and recall values
            qps_metrics = vdb_data[vdb_data['metric_name'].str.contains('qps')]
            recall_metrics = vdb_data[vdb_data['metric_name'].str.contains('recall')]
            
            if not qps_metrics.empty and not recall_metrics.empty:
                fig.add_trace(go.Scatter(
                    x=recall_metrics['value'].values,
                    y=qps_metrics['value'].values,
                    mode='markers+lines',
                    name=vdb.title(),
                    marker=dict(size=10),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='QPS vs Recall Trade-off',
            xaxis_title='Recall@10',
            yaxis_title='Queries Per Second (QPS)',
            height=500,
            width=800,
            hovermode='x unified'
        )
        
        # Save figure
        output_path = self.results_dir / "qps_recall_chart.html"
        fig.write_html(str(output_path))
        print(f"Saved QPS vs Recall chart to {output_path}")
    
    def create_gpu_cpu_comparison(self):
        """Create GPU vs CPU performance comparison"""
        gpu_df = self.df[self.df['device'] == 'gpu']
        cpu_df = self.df[self.df['device'] == 'cpu']
        
        if gpu_df.empty or cpu_df.empty:
            print("Insufficient GPU/CPU data for comparison")
            return
        
        # Calculate speedup for each metric
        speedup_data = []
        
        for metric in gpu_df['metric_name'].unique():
            gpu_val = gpu_df[gpu_df['metric_name'] == metric]['value'].mean()
            cpu_val = cpu_df[cpu_df['metric_name'] == metric]['value'].mean()
            
            if cpu_val != 0:
                # For latency metrics, speedup is inverse
                if 'latency' in metric:
                    speedup = cpu_val / gpu_val
                else:
                    speedup = gpu_val / cpu_val
                
                speedup_data.append({
                    'metric': metric,
                    'gpu': gpu_val,
                    'cpu': cpu_val,
                    'speedup': speedup
                })
        
        speedup_df = pd.DataFrame(speedup_data)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Absolute Performance', 'GPU Speedup Factor'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Absolute performance comparison
        fig.add_trace(
            go.Bar(name='GPU', x=speedup_df['metric'], y=speedup_df['gpu']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='CPU', x=speedup_df['metric'], y=speedup_df['cpu']),
            row=1, col=1
        )
        
        # Speedup factors
        colors = ['green' if x > 1 else 'red' for x in speedup_df['speedup']]
        fig.add_trace(
            go.Bar(x=speedup_df['metric'], y=speedup_df['speedup'],
                   marker_color=colors, showlegend=False),
            row=1, col=2
        )
        
        # Add horizontal line at y=1 for speedup chart
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=2)
        
        fig.update_layout(
            title='GPU vs CPU Performance Comparison',
            height=500,
            width=1200
        )
        
        # Save figure
        output_path = self.results_dir / "gpu_cpu_comparison.html"
        fig.write_html(str(output_path))
        print(f"Saved GPU vs CPU comparison to {output_path}")
    
    def create_upsert_performance_chart(self):
        """Create upsert performance comparison"""
        upsert_df = self.df[self.df['metric_type'] == 'upsert']
        
        if upsert_df.empty:
            print("No upsert data found")
            return
        
        # Filter for embeddings_per_second metric
        eps_df = upsert_df[upsert_df['metric_name'] == 'embeddings_per_second']
        
        if not eps_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by VDB type and device
            grouped = eps_df.groupby(['vdb_type', 'device'])['value'].mean().unstack()
            grouped.plot(kind='bar', ax=ax)
            
            ax.set_title('Upsert Performance: Embeddings per Second', fontsize=16, fontweight='bold')
            ax.set_xlabel('Vector Database', fontsize=12)
            ax.set_ylabel('Embeddings/Second', fontsize=12)
            ax.legend(title='Device', loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f', padding=3)
            
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            # Save figure
            output_path = self.results_dir / "upsert_performance.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved upsert performance chart to {output_path}")
    
    def create_e2e_metrics_chart(self):
        """Create end-to-end metrics visualization"""
        e2e_df = self.df[self.df['metric_type'].str.contains('e2e')]
        
        if e2e_df.empty:
            print("No end-to-end data found")
            return
        
        # Create subplots for different E2E metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Time to First Token',
                'Total Latency',
                'Tokens per Second',
                'Accuracy Metrics'
            )
        )
        
        # Time to first token
        ttft_df = e2e_df[e2e_df['metric_name'] == 'time_to_first_token_ms']
        if not ttft_df.empty:
            for vdb in ttft_df['vdb_type'].unique():
                vdb_data = ttft_df[ttft_df['vdb_type'] == vdb]
                fig.add_trace(
                    go.Box(y=vdb_data['value'], name=vdb.title()),
                    row=1, col=1
                )
        
        # Total latency
        total_df = e2e_df[e2e_df['metric_name'] == 'total_latency_ms']
        if not total_df.empty:
            for vdb in total_df['vdb_type'].unique():
                vdb_data = total_df[total_df['vdb_type'] == vdb]
                fig.add_trace(
                    go.Box(y=vdb_data['value'], name=vdb.title()),
                    row=1, col=2
                )
        
        # Tokens per second
        tps_df = e2e_df[e2e_df['metric_name'] == 'tokens_per_second']
        if not tps_df.empty:
            for vdb in tps_df['vdb_type'].unique():
                vdb_data = tps_df[tps_df['vdb_type'] == vdb]
                fig.add_trace(
                    go.Box(y=vdb_data['value'], name=vdb.title()),
                    row=2, col=1
                )
        
        # Accuracy
        acc_df = e2e_df[e2e_df['metric_name'].str.contains('accuracy')]
        if not acc_df.empty:
            acc_pivot = acc_df.pivot_table(
                values='value',
                index='metric_name',
                columns='vdb_type',
                aggfunc='mean'
            )
            
            for vdb in acc_pivot.columns:
                fig.add_trace(
                    go.Bar(x=acc_pivot.index, y=acc_pivot[vdb], name=vdb.title()),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, width=1200, title='End-to-End Performance Metrics')
        
        # Save figure
        output_path = self.results_dir / "e2e_metrics.html"
        fig.write_html(str(output_path))
        print(f"Saved E2E metrics chart to {output_path}")
    
    def create_summary_dashboard(self):
        """Create a comprehensive dashboard with all metrics"""
        # Create a single page dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Query Latency (P50)', 'Query Latency (P99)', 'QPS Performance',
                'Upsert Rate', 'Index Build Time', 'GPU Speedup',
                'Time to First Token', 'Tokens/Second', 'Recall@10'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Helper function to add comparison bars
        def add_comparison_metric(metric_name, row, col, lower_better=True):
            metric_df = self.df[self.df['metric_name'] == metric_name]
            if not metric_df.empty:
                for vdb in metric_df['vdb_type'].unique():
                    vdb_data = metric_df[metric_df['vdb_type'] == vdb]
                    value = vdb_data['value'].mean()
                    color = 'lightblue' if vdb == 'cyborgdb' else 'lightcoral'
                    fig.add_trace(
                        go.Bar(x=[vdb.title()], y=[value], name=vdb.title(),
                               marker_color=color, showlegend=(row == 1 and col == 1)),
                        row=row, col=col
                    )
        
        # Add all metrics
        add_comparison_metric('p50_latency_ms_top10', 1, 1, True)
        add_comparison_metric('p99_latency_ms_top10', 1, 2, True)
        add_comparison_metric('qps_batch_10', 1, 3, False)
        add_comparison_metric('embeddings_per_second', 2, 1, False)
        add_comparison_metric('build_time_seconds', 2, 2, True)
        
        # GPU Speedup special case
        gpu_df = self.df[(self.df['device'] == 'gpu') & (self.df['vdb_type'] == 'cyborgdb')]
        cpu_df = self.df[(self.df['device'] == 'cpu') & (self.df['vdb_type'] == 'cyborgdb')]
        if not gpu_df.empty and not cpu_df.empty:
            speedup = cpu_df['value'].mean() / gpu_df['value'].mean()
            fig.add_trace(
                go.Bar(x=['GPU Speedup'], y=[speedup], marker_color='green', showlegend=False),
                row=2, col=3
            )
        
        add_comparison_metric('time_to_first_token_ms', 3, 1, True)
        add_comparison_metric('tokens_per_second', 3, 2, False)
        add_comparison_metric('recall_batch_10', 3, 3, False)
        
        fig.update_layout(
            height=900,
            width=1400,
            title='CyborgDB vs Milvus: Comprehensive Performance Dashboard',
            showlegend=True
        )
        
        # Save dashboard
        output_path = self.results_dir / "performance_dashboard.html"
        fig.write_html(str(output_path))
        print(f"Saved performance dashboard to {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualization charts"""
        print("Generating benchmark visualizations...")
        
        self.create_latency_comparison_chart()
        self.create_qps_recall_chart()
        self.create_gpu_cpu_comparison()
        self.create_upsert_performance_chart()
        self.create_e2e_metrics_chart()
        self.create_summary_dashboard()
        
        print("\nAll visualizations generated successfully!")
        print(f"Results saved in: {self.results_dir}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Visualize CyborgDB benchmark results")
    parser.add_argument("results_dir", help="Directory containing benchmark results")
    parser.add_argument("--format", choices=["png", "html", "both"], default="both",
                       help="Output format for visualizations")
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()