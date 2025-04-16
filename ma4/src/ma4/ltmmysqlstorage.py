import json
import mysql.connector
from mysql.connector import Error
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from crewai.utilities import Printer
from crewai.utilities.paths import db_storage_path

class LTMMySQLStorage:
    """An custom MySQL storage class for LTM data storage."""

    def __init__(
        self, 
        host: str, 
        user: str, 
        password: str, 
        database: str # database sghould exist in the instance
    ) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """Initializes the MySQL database and creates LTM table."""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        task_description TEXT,
                        metadata TEXT,
                        datetime TEXT,
                        score FLOAT
                    )
                    """
                )
                conn.commit()
        except Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred during database initialization: {e}",
                color="red",
            )
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """Saves data to the LTM table with error handling."""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO long_term_memories (task_description, metadata, datetime, score)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (task_description, json.dumps(metadata), datetime, score),
                )
                conn.commit()
        except Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
                color="red",
            )
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Queries the LTM table by task description with error handling."""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = %s
                    ORDER BY datetime DESC, score ASC
                    LIMIT %s
                    """,
                    (task_description, latest_n),
                )
                rows = cursor.fetchall()
                if rows:
                    return [
                        {
                            "metadata": json.loads(row[0]),
                            "datetime": row[1],
                            "score": row[2],
                        }
                        for row in rows
                    ]
        except Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while querying LTM: {e}",
                color="red",
            )
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
        return None

    def reset(self) -> None:
        """Resets the LTM table with error handling."""
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute("DELETE FROM long_term_memories")
                conn.commit()
        except Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: An error occurred while deleting all rows in LTM: {e}",
                color="red",
            )
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()