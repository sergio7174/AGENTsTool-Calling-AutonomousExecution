# Tool-Calling& AutonExec\src\M2_DatabInspecScript.py

import os
import sys

# Ensure project root and src/ are discoverable in Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import psycopg
except ImportError:
    print("❌ 'psycopg' package not found. Install it using: pip install psycopg[binary]")
    sys.exit(1)

# Pull database URI from environment variables or fallback
DB_URI = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/ai_agents_db"
)

def inspect_agent_database():
    print(f"\n🔌 Connecting to PostgreSQL database...")
    try:
        with psycopg.connect(DB_URI) as conn:
            # Inspect the database for conversation threads, checkpoints, and writes
            # with conn.cursor() is a context manager that automatically handles closing the cursor. 
            with conn.cursor() as cur:
                
                # 1. List Active Conversation Threads
                print("\n========================================")
                print("📊 1. ACTIVE CONVERSATION THREADS")
                print("========================================")
                # Execute a query to fetch thread IDs and their latest checkpoint info
                cur.execute("""
                    SELECT thread_id, COUNT(*) AS total_checkpoints, MAX(checkpoint_id) AS latest_checkpoint
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY latest_checkpoint DESC;
                """)
                threads = cur.fetchall()
                if not threads:
                    print("⚠️ No conversation threads found in the checkpoints table yet.")
                else:
                    for row in threads:
                        print(f" • Thread ID: {row[0]}")
                        print(f"   - Total Checkpoints: {row[1]}")
                        print(f"   - Latest Checkpoint ID: {row[2]}")
                        print("-" * 40)

                # 2. Inspect Recent Checkpoints
                print("\n========================================")
                print("📜 2. RECENT CHECKPOINT HISTORY")
                print("========================================")
                cur.execute("""
                    SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_ns
                    FROM checkpoints
                    ORDER BY checkpoint_id DESC
                    LIMIT 5;
                """)
                checkpoints = cur.fetchall()
                if not checkpoints:
                    print("⚠️ No checkpoints found.")
                else:
                    for row in checkpoints:
                        print(f" • Thread: {row[0]}")
                        print(f"   - Checkpoint ID: {row[1]}")
                        print(f"   - Parent ID: {row[2]}")
                        print(f"   - Namespace: {row[3]}")
                        print("-" * 40)

                # 3. View Recent Task Writes
                print("\n========================================")
                print("✍️  3. RECENT TASK WRITES & STATE CHANGES")
                print("========================================")
                cur.execute("""
                    SELECT thread_id, checkpoint_id, task_id, channel
                    FROM checkpoint_writes
                    ORDER BY checkpoint_id DESC
                    LIMIT 5;
                """)
                writes = cur.fetchall()
                if not writes:
                    print("⚠️ No checkpoint writes found.")
                else:
                    for row in writes:
                        print(f" • Thread: {row[0]} | Checkpoint: {row[1]}")
                        print(f"   - Task ID: {row[2]} | Channel: {row[3]}")
                        print("-" * 40)

    except Exception as e:
        print(f"\n❌ Database inspection failed: {e}")

if __name__ == "__main__":
    inspect_agent_database()