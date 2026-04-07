import sqlite3

# Connect to the database
conn = sqlite3.connect('data/knowledge_base.db')
cursor = conn.cursor()

# Get the list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:")
for table in tables:
    print(f"- {table[0]}")

# Count records in videos table
cursor.execute("SELECT COUNT(*) FROM videos;")
count = cursor.fetchone()[0]
print(f"\nNumber of videos in database: {count}")

# Close the connection
conn.close()

print("\nDatabase access test completed successfully!") 