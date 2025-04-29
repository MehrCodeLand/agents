#!/usr/bin/env python
"""
This script demonstrates how to use the RAG system with the BankAgent.
It loads banking knowledge text files, builds a vector database,
and answers questions about banking services and financial products.

Usage:
    python bank_demo.py
"""

import os
import sys
# Make sure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from crew import BankAgent

def main():
    # Define the path to the knowledge directory
    knowledge_dir = "knowledge"
    
    # Ensure the knowledge directory exists
    if not os.path.exists(knowledge_dir):
        print(f"Knowledge directory '{knowledge_dir}' does not exist.")
        print("Creating directory...")
        os.makedirs(knowledge_dir)
        
        # Create a sample knowledge file if none exists
        sample_file = os.path.join(knowledge_dir, "banking_knowledge.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""# Banking Products and Services

## Checking Accounts

Checking accounts are the most basic banking accounts offered by financial institutions. They're designed for everyday transactions and provide easy access to your money. Features typically include:

- Debit cards for purchases and ATM withdrawals
- Online and mobile banking access
- Check writing capabilities
- Direct deposit options
- Overdraft protection services

Most basic checking accounts have low or no minimum balance requirements, though premium checking accounts may offer higher interest rates and additional benefits for maintaining a higher balance. Monthly maintenance fees vary by bank and account type, but many banks offer ways to waive these fees, such as setting up direct deposit or maintaining a minimum balance.

## Savings Accounts

Savings accounts are designed to help customers save money while earning interest. Key features include:

- Interest earning on deposited funds
- Limited monthly withdrawals (typically 6 per statement cycle)
- FDIC insurance up to $250,000
- Online and mobile account management
- Automatic transfer options for regular saving

Traditional savings accounts often offer lower interest rates, while high-yield savings accounts provide more competitive rates. Money market accounts are a type of savings account that typically offers higher interest rates but may require higher minimum balances.

## Certificates of Deposit (CDs)

CDs are time-deposit accounts that typically offer higher interest rates than regular savings accounts in exchange for leaving your money untouched for a specific term:

- Fixed terms ranging from 3 months to 5+ years
- Fixed interest rates for the entire term
- Higher interest rates for longer terms
- Early withdrawal penalties
- FDIC insurance up to $250,000

CD laddering is a strategy that involves opening multiple CDs with different maturity dates to provide both higher interest rates and periodic access to funds.

## Personal Loans

Banks offer personal loans for various needs, from debt consolidation to major purchases:

- Fixed interest rates and monthly payments
- Loan terms typically from 1-7 years
- Amounts typically from $1,000-$50,000
- No collateral required for unsecured loans
- Secured loan options (using collateral) for lower rates

Credit score, income, and debt-to-income ratio are major factors in loan approval and interest rate determination. Pre-qualification allows customers to check potential rates without affecting their credit score.

## Mortgages

Mortgage loans are used to purchase homes and real estate:

- Fixed-rate mortgages maintain the same interest rate for the entire loan term
- Adjustable-rate mortgages (ARMs) have rates that can change after an initial fixed period
- Conventional loans typically require a down payment of 3-20%
- FHA, VA, and USDA loans offer government-backed options with lower down payment requirements
- Jumbo loans exceed the conforming loan limits set by Fannie Mae and Freddie Mac

The mortgage process includes pre-approval, home shopping, application, underwriting, and closing. Mortgage interest may be tax-deductible for eligible borrowers.
            """)
        print(f"Created sample banking knowledge file: {sample_file}")
    
    # Initialize the agent with the knowledge directory
    agent = BankAgent(knowledge_dir=knowledge_dir)
    
    print("\n===== Banking Knowledge Assistant =====")
    print("This demo will answer your questions about banking services and financial products.")
    print("Type 'exit' to quit the demo.")
    
    # Interactive question answering loop
    while True:
        question = input("\nWhat would you like to know about banking services? ")
        
        if question.lower() in ["exit", "quit", "q"]:
            print("Exiting demo. Thank you for using the Banking Knowledge Assistant!")
            break
        
        print("\nSearching banking knowledge base...")
        answer = agent.answer_question(question, topic="Banking Services")
        print("\nAnswer:")
        print(answer)
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()