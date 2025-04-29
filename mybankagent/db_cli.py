#!/usr/bin/env python
"""
Command-line interface for managing the Banking Knowledge Vector Database.
This tool allows you to rebuild, backup, restore, and manage the vector database
used by the Banking Agent RAG system.

Usage:
    python db_cli.py [command] [options]

Commands:
    info        - Show information about the database
    rebuild     - Rebuild the database from knowledge files
    backup      - Create a backup of the current database
    restore     - Restore the database from a backup
    delete      - Delete the database
    collections - List all collections in the database
"""

import os
import sys
import argparse
from mybankagent.src.mybankagent.tools.db_manager import BankingDBManager

def main():
    """Main entry point for the DB CLI tool"""
    parser = argparse.ArgumentParser(description="Banking Knowledge Database Management CLI")
    
    # Add common arguments
    parser.add_argument("--knowledge-dir", "-k", type=str, default="knowledge",
                        help="Directory containing knowledge text files")
    parser.add_argument("--db-path", "-d", type=str, default="vector_db",
                        help="Directory for the vector database")
    parser.add_argument("--collection", "-c", type=str, default="banking_knowledge",
                        help="Name of the collection in Qdrant")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about the database")
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild the database from knowledge files")
    rebuild_parser.add_argument("--force", "-f", action="store_true", 
                              help="Force rebuild even if no changes detected")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup of the current database")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore the database from a backup")
    restore_parser.add_argument("--backup-path", "-b", type=str, required=True,
                              help="Path to the backup directory")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete the database")
    delete_parser.add_argument("--confirm", action="store_true",
                             help="Confirm deletion")
    
    # Collections command
    collections_parser = subparsers.add_parser("collections", help="List all collections in the database")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create database manager
    db_manager = BankingDBManager(
        knowledge_dir=args.knowledge_dir,
        db_path=args.db_path,
        collection_name=args.collection
    )
    
    # Execute command
    if args.command == "info":
        # Show information about the database
        info = db_manager.get_collection_info()
        print("\n=== Banking Knowledge Database Info ===")
        print(f"Collection: {info['name']}")
        print(f"Status: {info['status']}")
        if info['status'] == 'active':
            print(f"Vector count: {info['vectors_count']}")
            print(f"Vector size: {info['vector_size']}")
        else:
            print(f"Error: {info.get('error', 'Unknown error')}")
        
        # Show knowledge directory info
        text_files = [f for f in os.listdir(args.knowledge_dir) if f.endswith('.txt')]
        print(f"\nKnowledge directory: {args.knowledge_dir}")
        print(f"Text files: {len(text_files)}")
        for file in text_files:
            path = os.path.join(args.knowledge_dir, file)
            size = os.path.getsize(path) / 1024  # KB
            modified = os.path.getmtime(path)
            from datetime import datetime
            modified_str = datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')
            print(f" - {file} ({size:.1f} KB, modified: {modified_str})")
    
    elif args.command == "rebuild":
        # Rebuild the database
        print(f"Rebuilding database at {args.db_path}...")
        result = db_manager.rebuild_database(force=args.force)
        
        if result["status"] == "success":
            print(f"Success: {result['message']}")
            print(f"Time taken: {result['time_taken']:.2f} seconds")
        else:
            print(f"Error: {result['message']}")
    
    elif args.command == "backup":
        # Create a backup
        print("Creating database backup...")
        result = db_manager.create_backup()
        print(result)
    
    elif args.command == "restore":
        # Restore from backup
        if not os.path.exists(args.backup_path):
            print(f"Error: Backup path not found: {args.backup_path}")
            return
        
        print(f"Restoring database from {args.backup_path}...")
        result = db_manager.restore_backup(args.backup_path)
        print(result)
    
    elif args.command == "delete":
        # Delete the database
        if not args.confirm:
            confirm = input(f"Are you sure you want to delete the database at {args.db_path}? (y/N): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled.")
                return
        
        print(f"Deleting database at {args.db_path}...")
        result = db_manager.delete_database()
        print(result)
    
    elif args.command == "collections":
        # List collections
        collections = db_manager.list_collections()
        print("\n=== Collections ===")
        if collections:
            for i, collection in enumerate(collections, 1):
                print(f"{i}. {collection}")
        else:
            print("No collections found.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()