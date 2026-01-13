import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
import glob
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

class FinalVisualizer:
    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        
        self.results_dir = self.project_root / 'results'
        self.models_dir = self.project_root / 'models'
        self.data_path = self.project_root / 'data/processed/cleaned_data.csv'
        self.plots_dir = self.project_root / 'plots'
        
        self.plots_dir.mkdir(exist_ok=True)
        print(f"Project Root: {self.project_root}")
        print(f"Saving plots to: {self.plots_dir}")

    def load_resources(self):
        """Load all necessary files"""
        print("\nLoading resources...")
        
        # Load JSON report
        list_of_files = glob.glob(str(self.results_dir / 'benchmarking_report_*.json'))
        if not list_of_files:
            raise FileNotFoundError("No benchmarking report found!")
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            self.report = json.load(f)
        print(f"Report loaded: {Path(latest_file).name}")

        # Identify best model
        self.best_model_name = None
        for name, details in self.report.items():
            if details.get('is_best'):
                self.best_model_name = name
                break
        print(f"Champion Model: {self.best_model_name}")

        # Load model and vectorizer
        self.model = joblib.load(self.models_dir / 'best_fake_news_model.joblib')
        self.vectorizer = joblib.load(self.models_dir / 'tfidf_vectorizer.joblib')
        print("Model & Vectorizer loaded.")

        # Load and prepare data
        df = pd.read_csv(self.data_path)
        text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        df[text_col] = df[text_col].fillna('')
        
        print("Transforming data for plotting...")
        X = self.vectorizer.transform(df[text_col])
        y = df['label']
        
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data ready.")

    def plot_1_comparison(self):
        """Model comparison plot"""
        print("Generating Plot 1: Model Comparison...")
        data = []
        for model_name, details in self.report.items():
            metrics = details['metrics']
            data.append({'Model': model_name, 'Metric': 'F1-Score', 'Score': metrics[f'{model_name}_f1_mean']})
            data.append({'Model': model_name, 'Metric': 'Accuracy', 'Score': metrics[f'{model_name}_accuracy_mean']})
            
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Model", y="Score", hue="Metric", palette="viridis")
        plt.ylim(0.8, 1.0)
        plt.title("Model Comparison Results", fontweight='bold')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
            
        plt.savefig(self.plots_dir / '1_comparison.png')
        plt.close()
        print("Saved: 1_comparison.png")

    def plot_2_confusion_matrix(self):
        """Confusion matrix plot"""
        print(f"Generating Plot 2: Confusion Matrix for {self.best_model_name}...")
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title(f"Confusion Matrix: {self.best_model_name}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.savefig(self.plots_dir / '2_confusion_matrix.png')
        plt.close()
        print("Saved: 2_confusion_matrix.png")

    def plot_3_roc_curve(self):
        """ROC Curve plot"""
        print(f"Generating Plot 3: ROC Curve for {self.best_model_name}...")
        
        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, "decision_function"):
            print("Using decision_function for SVM ROC...")
            y_score = self.model.decision_function(self.X_test)
        else:
            print("Skipped ROC: Model doesn't support scoring.")
            return

        fpr, tpr, _ = roc_curve(self.y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#1f77b4', lw=3, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f"ROC Curve: {self.best_model_name}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        
        plt.savefig(self.plots_dir / '3_roc_curve.png')
        plt.close()
        print("Saved: 3_roc_curve.png")

    def plot_4_feature_importance(self):
        """Feature importance plot"""
        print(f"Generating Plot 4: Feature Importance...")
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = None
        if hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0]) if self.model.coef_.shape[0] == 1 else np.mean(np.abs(self.model.coef_), axis=0)
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

        if importances is not None:
            indices = np.argsort(importances)[::-1][:20]
            top_features = [feature_names[i] for i in indices]
            top_scores = importances[indices]

            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_scores, y=top_features, palette="mako")
            plt.title(f"Top 20 Features (Words) used by {self.best_model_name}")
            plt.xlabel("Importance")
            
            plt.savefig(self.plots_dir / '4_feature_importance.png')
            plt.close()
            print("Saved: 4_feature_importance.png")
        else:
            print("Skipped Feature Importance (Not available for this model).")

    def run(self):
        try:
            self.load_resources()
            self.plot_1_comparison()
            self.plot_2_confusion_matrix()
            self.plot_3_roc_curve()
            self.plot_4_feature_importance()
            print(f"All plots saved successfully in: {self.plots_dir}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    viz = FinalVisualizer()
    viz.run()
