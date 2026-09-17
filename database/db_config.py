"""
Database configuration for CipherBench.
Handles MySQL connection setup.
"""

import mysql.connector
from mysql.connector import pooling
import yaml
import os
from typing import Optional


class DatabaseConfig:
    """Database configuration and connection manager."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database configuration.

        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'config.yaml'
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.db_config = config['database']
        # Prefer an environment variable for the password so real credentials
        # never need to be committed to config.yaml.
        env_password = os.environ.get('CIPHERBENCH_DB_PASSWORD')
        if env_password:
            self.db_config['password'] = env_password
        self.connection_pool = None

    def create_connection_pool(self, pool_name: str = "cipherbench_pool",
                              pool_size: int = 5):
        """
        Create a connection pool for better performance.

        Args:
            pool_name: Name of the connection pool
            pool_size: Number of connections in the pool
        """
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name=pool_name,
                pool_size=pool_size,
                pool_reset_session=True,
                host=self.db_config['host'],
                port=self.db_config.get('port', 3306),
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print(f"[OK] Connection pool '{pool_name}' created successfully")
        except mysql.connector.Error as err:
            print(f"[ERROR] Error creating connection pool: {err}")
            raise

    def get_connection(self):
        """
        Get a connection from the pool.

        Returns:
            MySQL connection object
        """
        if self.connection_pool is None:
            self.create_connection_pool()

        try:
            return self.connection_pool.get_connection()
        except mysql.connector.Error as err:
            print(f"[ERROR] Error getting connection: {err}")
            raise

    def close_pool(self):
        """Close all connections in the pool."""
        if self.connection_pool:
            # Connection pools don't have a direct close method
            # Connections are automatically returned to the pool
            print("Connection pool will be cleaned up automatically")


def test_connection(config_path: Optional[str] = None):
    """
    Test database connection.

    Args:
        config_path: Path to config.yaml file
    """
    try:
        db_config = DatabaseConfig(config_path)
        conn = db_config.get_connection()
        cursor = conn.cursor()

        # Test query
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"[OK] Connected to MySQL version: {version[0]}")

        # Check if database exists
        cursor.execute(f"SHOW DATABASES LIKE '{db_config.db_config['database']}'")
        result = cursor.fetchone()

        if result:
            print(f"[OK] Database '{db_config.db_config['database']}' exists")
        else:
            print(f"[ERROR] Database '{db_config.db_config['database']}' does not exist")
            print("  Run: mysql -u root -p < database/schema.sql")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
