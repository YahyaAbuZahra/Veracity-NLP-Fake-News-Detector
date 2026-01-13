import pandas as pd
import joblib
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
import wandb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from features import create_features

warnings.filterwarnings('ignore')


class FakeNewsModelBenchmark:
    """Professional system for comparing and evaluating fake news detection models"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.results_dir = Path('results')
        self.models_dir = Path('models')
        self._setup_directories()
        
    def _default_config(self):
        """Default enhanced configuration"""
        return {
            "cv_splits": 5,
            "max_features": 5000,
            "ngram_range": (1, 2),
            "random_state": 42,
            "n_jobs": -1,
            "test_size": 0.2,
            "wandb_project": "fake-news-detection",
            "save_cv_predictions": True
        }
        
    
    def _setup_directories(self):
        """Create required directories"""
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
    def get_models(self):
        """Define models with optimized hyperparameters"""
        return {
            "Logistic_Regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='saga',
                penalty='l2',
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state']
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state'],
                class_weight='balanced'
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=self.config['n_jobs'],
                random_state=self.config['random_state']
            ),
            "SVM": LinearSVC(
                C=1.0,
                max_iter=2000,
                dual=False,
                class_weight='balanced',
                random_state=self.config['random_state']
            )
        }
    
    def evaluate_model(self, model, X, y, model_name):
        """Comprehensive evaluation for a single model"""
        skf = StratifiedKFold(
            n_splits=self.config['cv_splits'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1_weighted',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'roc_auc': 'roc_auc'
        }
        
        print(f"Evaluating {model_name}...")
        
        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=scoring,
            n_jobs=self.config['n_jobs'],
            return_train_score=True,
            return_estimator=True
        )
        
        metrics = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            metrics[f'{model_name}_{metric}_mean'] = test_scores.mean()
            metrics[f'{model_name}_{metric}_std'] = test_scores.std()
            metrics[f'{model_name}_{metric}_train'] = train_scores.mean()
            metrics[f'{model_name}_{metric}_gap'] = (
                train_scores.mean() - test_scores.mean()
            )
        
        metrics[f'{model_name}_fit_time'] = cv_results['fit_time'].mean()
        metrics[f'{model_name}_score_time'] = cv_results['score_time'].mean()
        
        print(f" {model_name}:")
        print(f"   F1-Score: {metrics[f'{model_name}_f1_mean']:.4f} (±{metrics[f'{model_name}_f1_std']:.4f})")
        print(f"   Accuracy: {metrics[f'{model_name}_accuracy_mean']:.4f}")
        print(f"   ROC-AUC: {metrics[f'{model_name}_roc_auc_mean']:.4f}")
        print(f"   Training Time: {metrics[f'{model_name}_fit_time']:.2f}s")
        
        return metrics, cv_results
    
    def run_benchmarking(self, data_path='data/processed/cleaned_data.csv'):
        """Execute complete benchmarking pipeline"""
        
        run = wandb.init(
            project=self.config['wandb_project'],
            name=f"benchmarking-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config
        )
        
        print(" Loading features...")
        X, y = create_features(data_path)
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        wandb.log({
            "dataset_size": X.shape[0],
            "n_features": X.shape[1],
            "class_0": np.sum(y == 0),
            "class_1": np.sum(y == 1)
        })
        
        models = self.get_models()
        
        all_results = {}
        best_f1 = 0
        best_model_name = None
        best_model_obj = None
        
        print("\n Starting Professional Benchmarking...\n")
        print("=" * 70)
        
        for model_name, model in models.items():
            metrics, cv_results = self.evaluate_model(model, X, y, model_name)
            all_results[model_name] = {
                'metrics': metrics,
                'cv_results': cv_results
            }
            
            wandb.log(metrics)
            
            current_f1 = metrics[f'{model_name}_f1_mean']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_name = model_name
                best_model_obj = model
            
            print("-" * 70)
        
        self._display_summary(all_results, best_model_name)
        
        print(f"\n Best Model: {best_model_name}")
        print(f"   F1-Score: {best_f1:.4f}")
        self._save_best_model(best_model_obj, X, y, best_model_name, all_results)
        
        self._save_detailed_report(all_results, best_model_name)
        
        wandb.finish()
        print("\n Process complete! All artifacts saved.")
        
        return all_results, best_model_name
    
    def _display_summary(self, results, best_model_name):
        """Display comparative summary"""
        print("\n" + "=" * 70)
        print(" BENCHMARKING SUMMARY")
        print("=" * 70)
        
        comparison = []
        for model_name in results.keys():
            metrics = results[model_name]['metrics']
            comparison.append({
                'Model': model_name,
                'F1': f"{metrics[f'{model_name}_f1_mean']:.4f}",
                'Accuracy': f"{metrics[f'{model_name}_accuracy_mean']:.4f}",
                'ROC-AUC': f"{metrics[f'{model_name}_roc_auc_mean']:.4f}",
                'Time(s)': f"{metrics[f'{model_name}_fit_time']:.2f}"
            })
        
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))
        print("=" * 70)
    
    def _save_best_model(self, model, X, y, model_name, all_results):
        """Save best model with metadata"""
        print(f"\n Training final {model_name} on full dataset...")
        model.fit(X, y)
        
        model_path = self.models_dir / 'best_fake_news_model.joblib'
        joblib.dump(model, model_path)
        
        metadata = {
            'model_name': model_name,
            'trained_date': datetime.now().isoformat(),
            'dataset_size': X.shape[0],
            'n_features': X.shape[1],
            'metrics': all_results[model_name]['metrics'],
            'config': self.config
        }
        
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   Model saved: {model_path}")
        print(f"   Metadata saved: {metadata_path}")
    
    def _save_detailed_report(self, results, best_model_name):
        """Save detailed report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f'benchmarking_report_{timestamp}.json'
        
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                'metrics': data['metrics'],
                'is_best': model_name == best_model_name
            }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"   Report saved: {report_path}")


def main():
    """Main entry point"""
    #os.environ["WANDB_MODE"] = "disabled"
    custom_config = {
        "cv_splits": 5,
        "max_features": 5000,
        "ngram_range": (1, 2),
        "random_state": 42,
        "n_jobs": -1,
        "wandb_project": "fake-news-detection",
        "save_cv_predictions": True

    }
    
    benchmark = FakeNewsModelBenchmark(config=custom_config)
    results, best_model = benchmark.run_benchmarking()
    
    return results, best_model


if __name__ == "__main__":
    main()