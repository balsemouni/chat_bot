# test_connection.py
from database import engine, SessionLocal, init_db
from sqlalchemy import text

def test_connection():
    print("Testing database connection...")
    
    # Test 1: Check if we can connect using engine
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Engine connection successful!")
            print(f"   Query result: {result.scalar()}")
    except Exception as e:
        print(f"❌ Engine connection failed: {e}")
        return False
    
    # Test 2: Test session creation
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).scalar()
        print("✅ Session creation successful!")
        print(f"   Query result: {result}")
        db.close()
    except Exception as e:
        print(f"❌ Session creation failed: {e}")
        return False
    
    # Test 3: Test database initialization
    try:
        init_db()
        print("✅ Database initialization successful!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    
    # Test 4: Get database version
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version()")).scalar()
            print(f"✅ Database version: {result[:50]}...")  # Show first 50 chars
    except Exception as e:
        print(f"❌ Could not get database version: {e}")
    
    print("\n🎉 All tests passed! Database connection is working!")
    return True

if __name__ == "__main__":
    test_connection()