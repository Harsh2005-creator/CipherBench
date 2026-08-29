-- CipherBench Database Schema
-- MySQL database for storing cryptographic algorithm identification experiment results

CREATE DATABASE IF NOT EXISTS cipherbench
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE cipherbench;

-- Main experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    task_type ENUM('binary', 'multiclass') NOT NULL,
    algorithm_pair VARCHAR(100) NOT NULL,  -- e.g., "AES and 3DES" or "5-class"
    ciphertext_size VARCHAR(10) NOT NULL,  -- "1KB", "8KB", "64KB", "256KB", "512KB"
    accuracy FLOAT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    training_time FLOAT,  -- Training time in seconds
    hyperparameters JSON,  -- Model hyperparameters as JSON
    confusion_matrix JSON,  -- Confusion matrix as JSON array
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model (model_name),
    INDEX idx_task (task_type),
    INDEX idx_size (ciphertext_size),
    INDEX idx_accuracy (accuracy DESC),
    INDEX idx_composite (model_name, task_type, ciphertext_size)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Sample queries:
-- Get best models per task and size:
-- SELECT model_name, task_type, ciphertext_size, MAX(accuracy) as best_accuracy
-- FROM experiments
-- GROUP BY task_type, ciphertext_size
-- ORDER BY best_accuracy DESC;

-- Compare all models on 512KB multiclass:
-- SELECT model_name, accuracy, precision_score, recall_score, f1_score
-- FROM experiments
-- WHERE task_type = 'multiclass' AND ciphertext_size = '512KB'
-- ORDER BY accuracy DESC;
