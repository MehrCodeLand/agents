#!/usr/bin/env python
import sys
import warnings
import argparse
from datetime import datetime

from crew import BankAgent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the BankAgent crew with default workflow.
    """
    inputs = {
        'topic': 'Personal Banking',
        'current_year': str(datetime.now().year)
    }
    BankAgent().crew().kickoff(inputs=inputs)

def run_rag():
    """
    Run the RAG-based banking question answering system.
    """
    parser = argparse.ArgumentParser(description='Ask banking questions to the knowledge base')
    parser.add_argument('--question', '-q', type=str, help='Banking question to ask the knowledge base')
    parser.add_argument('--knowledge-dir', '-k', type=str, default='knowledge', 
                        help='Directory containing banking knowledge base text files')
    parser.add_argument('--topic', '-t', type=str, default='Banking Services',
                        help='Topic context for the banking question')
    
    args = parser.parse_args()
    
    if not args.question:
        # Interactive mode
        print("Banking Knowledge Base Question-Answering System")
        print(f"Using knowledge directory: {args.knowledge_dir}")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nWhat would you like to know about banking services? ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            agent = BankAgent(knowledge_dir=args.knowledge_dir)
            answer = agent.answer_question(question, args.topic)
            print("\nAnswer:")
            print(answer)
    else:
        # Single question mode
        agent = BankAgent(knowledge_dir=args.knowledge_dir)
        answer = agent.answer_question(args.question, args.topic)
        print(answer)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        # Remove the "rag" argument
        sys.argv.pop(1)
        run_rag()
    else:
        run()