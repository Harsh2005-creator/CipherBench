"""
Flask REST API for CipherBench.
Exposes endpoints to query experiment results from the database.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_operations import ExperimentDB

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Initialize database
db = ExperimentDB()


@app.route('/')
def index():
    """API information endpoint."""
    return jsonify({
        'name': 'CipherBench API',
        'version': '1.0.0',
        'description': 'REST API for cryptographic algorithm identification experiment results',
        'endpoints': {
            '/api/models': 'List all available models',
            '/api/results': 'Query experiment results (supports filtering)',
            '/api/best': 'Get best performing models',
            '/api/confusion_matrix/<id>': 'Get confusion matrix for experiment',
            '/api/comparison': 'Compare all models for a task and size'
        }
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of all models."""
    try:
        experiments = db.get_experiments()
        models = list(set([exp['model_name'] for exp in experiments]))
        return jsonify({
            'success': True,
            'models': sorted(models)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """
    Get experiment results with optional filtering.
    Query parameters: model, task, size
    """
    try:
        model_name = request.args.get('model')
        task_type = request.args.get('task')
        ciphertext_size = request.args.get('size')

        experiments = db.get_experiments(
            model_name=model_name,
            task_type=task_type,
            ciphertext_size=ciphertext_size
        )

        # Convert datetime to string
        for exp in experiments:
            if exp.get('timestamp'):
                exp['timestamp'] = exp['timestamp'].isoformat()

        return jsonify({
            'success': True,
            'count': len(experiments),
            'results': experiments
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/best', methods=['GET'])
def get_best_models():
    """
    Get best performing models.
    Query parameters: task (required), size (required), top_n (default: 5)
    """
    try:
        task_type = request.args.get('task')
        ciphertext_size = request.args.get('size')
        top_n = int(request.args.get('top_n', 5))

        if not task_type or not ciphertext_size:
            return jsonify({
                'success': False,
                'error': 'Both task and size parameters are required'
            }), 400

        best_models = db.get_best_models(task_type, ciphertext_size, top_n)

        return jsonify({
            'success': True,
            'task_type': task_type,
            'ciphertext_size': ciphertext_size,
            'top_n': top_n,
            'models': best_models
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/confusion_matrix/<int:experiment_id>', methods=['GET'])
def get_confusion_matrix(experiment_id):
    """Get confusion matrix for a specific experiment."""
    try:
        cm = db.get_confusion_matrix(experiment_id)

        if cm is None:
            return jsonify({
                'success': False,
                'error': f'Experiment {experiment_id} not found'
            }), 404

        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'confusion_matrix': cm.tolist()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    """
    Compare all models for a specific task and size.
    Query parameters: task (required), size (required)
    """
    try:
        task_type = request.args.get('task')
        ciphertext_size = request.args.get('size')

        if not task_type or not ciphertext_size:
            return jsonify({
                'success': False,
                'error': 'Both task and size parameters are required'
            }), 400

        comparison = db.get_model_comparison(task_type, ciphertext_size)

        return jsonify({
            'success': True,
            'task_type': task_type,
            'ciphertext_size': ciphertext_size,
            'comparison': comparison
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall statistics."""
    try:
        all_experiments = db.get_experiments()

        stats = {
            'total_experiments': len(all_experiments),
            'models': list(set([exp['model_name'] for exp in all_experiments])),
            'task_types': list(set([exp['task_type'] for exp in all_experiments])),
            'ciphertext_sizes': list(set([exp['ciphertext_size'] for exp in all_experiments]))
        }

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting CipherBench API server...")
    print("API Documentation: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
