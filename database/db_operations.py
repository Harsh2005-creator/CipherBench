"""
Database operations for CipherBench.
CRUD operations for storing and retrieving experiment results.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from database.db_config import DatabaseConfig


class ExperimentDB:
    """Database operations for experiments."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize database connection."""
        self.db_config = DatabaseConfig(config_path)

    def insert_experiment(self, model_name: str, task_type: str,
                         algorithm_pair: str, ciphertext_size: str,
                         accuracy: float, precision: float,
                         recall: float, f1_score: float,
                         training_time: float,
                         hyperparameters: Dict[str, Any],
                         confusion_matrix: np.ndarray) -> int:
        """
        Insert a new experiment result into the database.

        Args:
            model_name: Name of the model (e.g., "SVM", "HKNNRF", "MLP")
            task_type: "binary" or "multiclass"
            algorithm_pair: Algorithm pair name or "5-class"
            ciphertext_size: Size of ciphertext (e.g., "512KB")
            accuracy: Accuracy score
            precision: Precision score
            recall: Recall score
            f1_score: F1 score
            training_time: Training time in seconds
            hyperparameters: Dictionary of hyperparameters
            confusion_matrix: Confusion matrix as numpy array

        Returns:
            Inserted experiment ID
        """
        conn = self.db_config.get_connection()
        cursor = conn.cursor()

        try:
            # Convert confusion matrix to JSON
            cm_json = json.dumps(confusion_matrix.tolist())
            hparams_json = json.dumps(hyperparameters)

            query = """
                INSERT INTO experiments
                (model_name, task_type, algorithm_pair, ciphertext_size,
                 accuracy, precision_score, recall_score, f1_score,
                 training_time, hyperparameters, confusion_matrix)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                model_name, task_type, algorithm_pair, ciphertext_size,
                round(accuracy, 4), round(precision, 4),
                round(recall, 4), round(f1_score, 4),
                round(training_time, 2),
                hparams_json, cm_json
            )

            cursor.execute(query, values)
            conn.commit()

            experiment_id = cursor.lastrowid
            print(f"[OK] Inserted experiment ID: {experiment_id}")

            return experiment_id

        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Error inserting experiment: {e}")
            raise

        finally:
            cursor.close()
            conn.close()

    def get_experiments(self, model_name: Optional[str] = None,
                       task_type: Optional[str] = None,
                       ciphertext_size: Optional[str] = None) -> List[Dict]:
        """
        Retrieve experiments with optional filtering.

        Args:
            model_name: Filter by model name
            task_type: Filter by task type
            ciphertext_size: Filter by ciphertext size

        Returns:
            List of experiment dictionaries
        """
        conn = self.db_config.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []

            if model_name:
                query += " AND model_name = %s"
                params.append(model_name)

            if task_type:
                query += " AND task_type = %s"
                params.append(task_type)

            if ciphertext_size:
                query += " AND ciphertext_size = %s"
                params.append(ciphertext_size)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            results = cursor.fetchall()

            # Parse JSON fields
            for result in results:
                if result['hyperparameters']:
                    result['hyperparameters'] = json.loads(result['hyperparameters'])
                if result['confusion_matrix']:
                    result['confusion_matrix'] = json.loads(result['confusion_matrix'])

            return results

        finally:
            cursor.close()
            conn.close()

    def get_best_models(self, task_type: str, ciphertext_size: str,
                       top_n: int = 5) -> List[Dict]:
        """
        Get top N best performing models for a specific task and size.

        Args:
            task_type: "binary" or "multiclass"
            ciphertext_size: Ciphertext size
            top_n: Number of top models to return

        Returns:
            List of top experiments
        """
        conn = self.db_config.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = """
                SELECT model_name, accuracy, precision_score, recall_score, f1_score,
                       algorithm_pair, hyperparameters
                FROM experiments
                WHERE task_type = %s AND ciphertext_size = %s
                ORDER BY accuracy DESC
                LIMIT %s
            """

            cursor.execute(query, (task_type, ciphertext_size, top_n))
            results = cursor.fetchall()

            # Parse JSON
            for result in results:
                if result['hyperparameters']:
                    result['hyperparameters'] = json.loads(result['hyperparameters'])

            return results

        finally:
            cursor.close()
            conn.close()

    def get_confusion_matrix(self, experiment_id: int) -> Optional[np.ndarray]:
        """
        Get confusion matrix for a specific experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Confusion matrix as numpy array or None
        """
        conn = self.db_config.get_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            query = "SELECT confusion_matrix FROM experiments WHERE id = %s"
            cursor.execute(query, (experiment_id,))
            result = cursor.fetchone()

            if result and result['confusion_matrix']:
                cm_list = json.loads(result['confusion_matrix'])
                return np.array(cm_list)

            return None

        finally:
            cursor.close()
            conn.close()

    def get_model_comparison(self, task_type: str, ciphertext_size: str) -> Dict:
        """
        Compare all models for a specific task and size.

        Args:
            task_type: "binary" or "multiclass"
            ciphertext_size: Ciphertext size

        Returns:
            Dictionary with model comparison data
        """
        experiments = self.get_experiments(
            task_type=task_type,
            ciphertext_size=ciphertext_size
        )

        comparison = {}
        for exp in experiments:
            model = exp['model_name']
            if model not in comparison:
                comparison[model] = []

            comparison[model].append({
                'accuracy': exp['accuracy'],
                'precision': exp['precision_score'],
                'recall': exp['recall_score'],
                'f1_score': exp['f1_score'],
                'algorithm_pair': exp['algorithm_pair']
            })

        return comparison

    def delete_experiment(self, experiment_id: int):
        """Delete an experiment by ID."""
        conn = self.db_config.get_connection()
        cursor = conn.cursor()

        try:
            query = "DELETE FROM experiments WHERE id = %s"
            cursor.execute(query, (experiment_id,))
            conn.commit()
            print(f"[OK] Deleted experiment ID: {experiment_id}")

        except Exception as e:
            conn.rollback()
            print(f"[ERROR] Error deleting experiment: {e}")
            raise

        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    # Test database operations
    print("Testing database operations...")

    db = ExperimentDB()

    # Test insert
    print("\n1. Inserting test experiment...")
    exp_id = db.insert_experiment(
        model_name="SVM",
        task_type="multiclass",
        algorithm_pair="5-class",
        ciphertext_size="512KB",
        accuracy=0.95,
        precision=0.94,
        recall=0.93,
        f1_score=0.94,
        training_time=12.5,
        hyperparameters={'kernel': 'linear', 'gamma': 0.001},
        confusion_matrix=np.array([[20, 0], [1, 19]])
    )

    # Test retrieval
    print("\n2. Retrieving experiments...")
    experiments = db.get_experiments(model_name="SVM", task_type="multiclass")
    print(f"   Found {len(experiments)} experiments")

    # Test best models
    print("\n3. Getting best models...")
    best = db.get_best_models("multiclass", "512KB", top_n=3)
    for i, model in enumerate(best, 1):
        print(f"   {i}. {model['model_name']}: {model['accuracy']:.4f}")

    print("\n[OK] Database operations test completed!")
