#!/usr/bin/env python3
"""
Machine Learning Log Analyzer for Dealmonitor
Analyzes hyperparameter tuning logs with focus on Top-3 Accuracy

Usage: python ml_log_analyzer.py <log_file_path>
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import sys
from datetime import datetime

class MLLogAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.content = self._read_log_file()
        self.progress_df = None
        self.model_df = None
        
    def _read_log_file(self):
        """Read the log file"""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print(f"✅ Log file loaded: {len(content)} characters")
            return content
        except FileNotFoundError:
            print(f"❌ Error: Log file '{self.log_file_path}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    
    def extract_data(self):
        """Extract progress and model data from logs"""
        print("🔍 Extracting data from logs...")
        
        # Extract progress lines
        progress_pattern = r'Tuning mlp:\s+(\d+)%\|.*?\|\s*(\d+)/540\s*\[(.+?)<(.+?),\s*(.*?)\]'
        progress_matches = re.findall(progress_pattern, self.content)
        
        # Extract model results
        model_pattern = r'✅ Model: mlp, Params: ({.*?})\r?\n▶ Top-3 Accuracy: ([\d.]+)\r?\n▶ Accuracy: ([\d.]+)\r?\n▶ F1 Score: ([\d.]+)'
        model_matches = re.findall(model_pattern, self.content, re.DOTALL)
        
        print(f"Found {len(progress_matches)} progress updates and {len(model_matches)} model results")
        
        # Create progress DataFrame
        self._create_progress_df(progress_matches)
        
        # Create model DataFrame
        self._create_model_df(model_matches)
        
    def _create_progress_df(self, progress_matches):
        """Create DataFrame from progress data"""
        progress_data = []
        for match in progress_matches:
            percent, current, elapsed, remaining, speed = match
            
            # Parse elapsed time
            elapsed_parts = elapsed.split(':')
            if len(elapsed_parts) == 3:
                hours, minutes, seconds = map(int, elapsed_parts)
                elapsed_total_minutes = hours * 60 + minutes + seconds / 60
            else:
                elapsed_total_minutes = 0
            
            progress_data.append({
                'percent': int(percent),
                'current': int(current),
                'elapsed_minutes': elapsed_total_minutes,
                'remaining': remaining,
                'speed': speed
            })
        
        self.progress_df = pd.DataFrame(progress_data)
        
    def _create_model_df(self, model_matches):
        """Create DataFrame from model results"""
        model_data = []
        for match in model_matches:
            params_str, top3_acc, accuracy, f1 = match
            
            try:
                # Parse parameters
                params = ast.literal_eval(params_str)
                
                model_data.append({
                    'top3_accuracy': float(top3_acc),
                    'accuracy': float(accuracy),
                    'f1_score': float(f1),
                    'activation': params.get('activation', ''),
                    'alpha': params.get('alpha', 0),
                    'hidden_layer_sizes': str(params.get('hidden_layer_sizes', '')),
                    'learning_rate_init': params.get('learning_rate_init', 0),
                    'max_iter': params.get('max_iter', 0),
                    'solver': params.get('solver', '')
                })
            except:
                # Skip if parsing fails
                continue
        
        self.model_df = pd.DataFrame(model_data)
        
    def analyze_top3_accuracy(self):
        """Detailed analysis focused on Top-3 Accuracy"""
        print("\n" + "=" * 60)
        print("TOP-3 ACCURACY FOKUSSIERTE ANALYSE")
        print("=" * 60)
        
        if self.model_df is None or len(self.model_df) == 0:
            print("❌ No model data available for analysis")
            return
            
        # Best results
        best_top3 = self.model_df.loc[self.model_df['top3_accuracy'].idxmax()]
        print(f"\n🏆 BESTE ERGEBNISSE (TOP-3 ACCURACY FOKUS):")
        print(f"• Höchste Top-3 Accuracy: {best_top3['top3_accuracy']:.1%}")
        print(f"• Accuracy: {best_top3['accuracy']:.1%}")
        print(f"• F1-Score: {best_top3['f1_score']:.1%}")
        print(f"• Konfiguration: {best_top3['hidden_layer_sizes']} mit {best_top3['solver']} solver")
        print(f"• Learning Rate: {best_top3['learning_rate_init']}")
        
        # Statistics
        print(f"\n📊 TOP-3 ACCURACY STATISTIKEN:")
        print(f"• Maximum: {self.model_df['top3_accuracy'].max():.1%}")
        print(f"• Durchschnitt: {self.model_df['top3_accuracy'].mean():.1%}")
        print(f"• Standardabweichung: {self.model_df['top3_accuracy'].std():.1%}")
        print(f"• Median: {self.model_df['top3_accuracy'].median():.1%}")
        
        # Parameter analysis
        self._analyze_parameters_top3()
        
        # Top 10 models
        self._show_top_models_top3()
        
        # Recommendations
        self._generate_recommendations_top3()
        
    def _analyze_parameters_top3(self):
        """Analyze parameters with focus on Top-3 Accuracy"""
        print(f"\n📈 PARAMETER-ANALYSE (TOP-3 ACCURACY):")
        
        # Solver comparison
        solver_stats = self.model_df.groupby('solver')['top3_accuracy'].agg(['mean', 'std', 'max', 'count']).round(3)
        print(f"\nTop-3 Accuracy nach Solver:")
        print(solver_stats)
        
        if 'adam' in solver_stats.index and 'sgd' in solver_stats.index:
            adam_top3 = solver_stats.loc['adam', 'mean']
            sgd_top3 = solver_stats.loc['sgd', 'mean']
            print(f"Adam vs SGD: {(adam_top3/sgd_top3-1)*100:.1f}% Unterschied")
        
        # Learning rate comparison
        lr_stats = self.model_df.groupby('learning_rate_init')['top3_accuracy'].agg(['mean', 'std', 'max', 'count']).round(3)
        print(f"\nTop-3 Accuracy nach Learning Rate:")
        print(lr_stats)
        
        # Architecture comparison
        layer_stats = self.model_df.groupby('hidden_layer_sizes')['top3_accuracy'].agg(['mean', 'std', 'max', 'count']).round(3)
        print(f"\nTop-3 Accuracy nach Architecture:")
        print(layer_stats.sort_values('mean', ascending=False))
        
    def _show_top_models_top3(self):
        """Show top 10 models by Top-3 Accuracy"""
        print(f"\n🥇 TOP 10 MODELLE (TOP-3 ACCURACY):")
        top10 = self.model_df.nlargest(10, 'top3_accuracy')[
            ['top3_accuracy', 'accuracy', 'f1_score', 'hidden_layer_sizes', 
             'solver', 'learning_rate_init', 'max_iter']
        ]
        print(top10.to_string(index=False))
        
    def _generate_recommendations_top3(self):
        """Generate recommendations based on Top-3 Accuracy"""
        best_arch = self.model_df.groupby('hidden_layer_sizes')['top3_accuracy'].mean().sort_values(ascending=False)
        best_solver = self.model_df.groupby('solver')['top3_accuracy'].mean().sort_values(ascending=False)
        best_lr = self.model_df.groupby('learning_rate_init')['top3_accuracy'].mean().sort_values(ascending=False)
        
        print(f"\n💡 EMPFEHLUNGEN (TOP-3 ACCURACY OPTIMIERT):")
        print(f"• Beste Architektur: {best_arch.index[0]} ({best_arch.iloc[0]:.1%})")
        print(f"• Bester Solver: {best_solver.index[0]} ({best_solver.iloc[0]:.1%})")
        print(f"• Beste Learning Rate: {best_lr.index[0]} ({best_lr.iloc[0]:.1%})")
        
        # Performance differences
        if len(best_arch) > 1:
            improvement = (best_arch.iloc[0] / best_arch.iloc[1] - 1) * 100
            print(f"• Architektur-Vorteil: {improvement:.1f}% besser als zweitbeste")
            
        if len(best_solver) > 1:
            solver_improvement = (best_solver.iloc[0] / best_solver.iloc[1] - 1) * 100
            print(f"• Solver-Vorteil: {solver_improvement:.1f}% besser als Alternative")
    
    def create_visualizations(self):
        """Create visualizations focused on Top-3 Accuracy"""
        if self.model_df is None or len(self.model_df) == 0:
            print("❌ No data available for visualizations")
            return
            
        print("\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Machine Learning Analysis - Top-3 Accuracy Focus', fontsize=16, fontweight='bold')
        
        # 1. Top-3 Accuracy by Solver
        solver_data = self.model_df.groupby('solver')['top3_accuracy'].agg(['mean', 'std'])
        ax1 = axes[0, 0]
        solver_data['mean'].plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'], 
                                yerr=solver_data['std'], capsize=5)
        ax1.set_title('Top-3 Accuracy by Solver')
        ax1.set_ylabel('Top-3 Accuracy')
        ax1.tick_params(axis='x', rotation=0)
        
        # 2. Top-3 Accuracy by Architecture
        arch_data = self.model_df.groupby('hidden_layer_sizes')['top3_accuracy'].mean().sort_values(ascending=False)
        ax2 = axes[0, 1]
        arch_data.plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Top-3 Accuracy by Architecture')
        ax2.set_ylabel('Top-3 Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Distribution of Top-3 Accuracy
        ax3 = axes[1, 0]
        self.model_df['top3_accuracy'].hist(bins=30, ax=ax3, color='gold', alpha=0.7)
        ax3.axvline(self.model_df['top3_accuracy'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.model_df["top3_accuracy"].mean():.3f}')
        ax3.set_title('Top-3 Accuracy Distribution')
        ax3.set_xlabel('Top-3 Accuracy')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Learning Rate vs Top-3 Accuracy
        ax4 = axes[1, 1]
        lr_data = self.model_df.groupby('learning_rate_init')['top3_accuracy'].agg(['mean', 'std'])
        lr_data['mean'].plot(kind='bar', ax=ax4, color='orange', 
                            yerr=lr_data['std'], capsize=5)
        ax4.set_title('Top-3 Accuracy by Learning Rate')
        ax4.set_ylabel('Top-3 Accuracy')
        ax4.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig('ml_analysis_top3_focus.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Visualizations saved as 'ml_analysis_top3_focus.png'")
    
    def export_results(self):
        """Export results to CSV files"""
        if self.model_df is not None:
            # All results
            self.model_df.to_csv('model_results_detailed.csv', index=False)
            
            # Top results by Top-3 Accuracy
            top_results = self.model_df.nlargest(20, 'top3_accuracy')
            top_results.to_csv('top_20_by_top3_accuracy.csv', index=False)
            
            # Summary statistics
            summary_stats = {
                'metric': ['top3_accuracy', 'accuracy', 'f1_score'],
                'mean': [self.model_df['top3_accuracy'].mean(), 
                        self.model_df['accuracy'].mean(),
                        self.model_df['f1_score'].mean()],
                'std': [self.model_df['top3_accuracy'].std(), 
                       self.model_df['accuracy'].std(),
                       self.model_df['f1_score'].std()],
                'max': [self.model_df['top3_accuracy'].max(), 
                       self.model_df['accuracy'].max(),
                       self.model_df['f1_score'].max()]
            }
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_csv('summary_statistics.csv', index=False)
            
            print(f"\n💾 Results exported:")
            print(f"• model_results_detailed.csv ({len(self.model_df)} entries)")
            print(f"• top_20_by_top3_accuracy.csv (20 best models)")
            print(f"• summary_statistics.csv (overall statistics)")
        
        if self.progress_df is not None:
            self.progress_df.to_csv('training_progress_detailed.csv', index=False)
            print(f"• training_progress_detailed.csv ({len(self.progress_df)} entries)")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting ML Log Analysis...")
        
        # Extract data
        self.extract_data()
        
        # Run Top-3 focused analysis
        self.analyze_top3_accuracy()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export results
        self.export_results()
        
        print(f"\n✅ Analysis completed!")
        print(f"   Focus: Top-3 Accuracy optimization")
        print(f"   Models analyzed: {len(self.model_df) if self.model_df is not None else 0}")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python ml_log_analyzer.py <log_file_path>")
        print("Example: python ml_log_analyzer.py paste.txt")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    # Create analyzer and run analysis
    analyzer = MLLogAnalyzer(log_file)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()