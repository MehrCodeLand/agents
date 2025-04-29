# Path: mycdagent/src/mycdagent/main.py

#!/usr/bin/env python
import sys
import warnings
import argparse
from datetime import datetime

from crew import Mycdagent

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew with default workflow.
    """
    inputs = {
        'topic': 'DC Comics',
        'current_year': str(datetime.now().year)
    }
    Mycdagent().crew().kickoff(inputs=inputs)

def run_rag():
    """
    Run the RAG-based question answering system.
    """
    parser = argparse.ArgumentParser(description='Ask questions to the knowledge base')
    parser.add_argument('--question', '-q', type=str, help='Question to ask the knowledge base')
    parser.add_argument('--knowledge-dir', '-k', type=str, default='knowledge', 
                        help='Directory containing knowledge base text files')
    parser.add_argument('--topic', '-t', type=str, default='General Knowledge',
                        help='Topic context for the question')
    
    args = parser.parse_args()
    
    if not args.question:
        # Interactive mode
        print("Knowledge Base Question-Answering System")
        print(f"Using knowledge directory: {args.knowledge_dir}")
        print("Type 'exit' to quit")
        
        while True:
            question = input("\nYour question: ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            agent = Mycdagent(knowledge_dir=args.knowledge_dir)
            answer = agent.answer_question(question, args.topic)
            print("\nAnswer:")
            print(answer)
    else:
        # Single question mode
        agent = Mycdagent(knowledge_dir=args.knowledge_dir)
        answer = agent.answer_question(args.question, args.topic)
        print(answer)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rag":
        # Remove the "rag" argument
        sys.argv.pop(1)
        run_rag()
    else:
        run()