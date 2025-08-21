#!/usr/bin/env python3
"""
Script để chạy migration và setup ML models
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from migrate import LogisticsMigration

def main():
    """Main function"""
    
    print("🚛 Logistics Optimization - Migration & ML Setup")
    print("=" * 60)
    
    try:
        # Run migration
        migration = LogisticsMigration()
        migration.run_full_migration()
        
        print("\n✅ Migration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run server: python run_server.py")
        print("2. Test API: python test_api.py")
        print("3. Check ML status: curl http://localhost:8000/ml/status")
        print("4. View API docs: http://localhost:8000/docs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
