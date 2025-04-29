from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any
import os

# Import the RAG Tool
from tools.rag_tool import RAGTool

@CrewBase
class BankAgent():
    """BankAgent crew with RAG capability for banking information"""

    agents: List[BaseAgent]
    tasks: List[Task]
    knowledge_dir: str = "knowledge"
    
    def __init__(self, knowledge_dir: str = "knowledge", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_dir = knowledge_dir
        
        # Initialize LLMs
        self.ollama_llm = LLM(
            model='ollama/llama3.2:3b',
            base_url='http://127.0.0.1:11434'
        )
    
    @agent
    def financial_advisor(self) -> Agent:
        # Create the RAG tool
        rag_tool = RAGTool(knowledge_dir=self.knowledge_dir)
        
        return Agent(
            config=self.agents_config['financial_advisor'],  # type: ignore[index]
            verbose=True,
            llm=self.ollama_llm,
            tools=[rag_tool]  # Add the RAG tool to the financial advisor agent
        )

    @agent
    def customer_service(self) -> Agent:
        # Create the RAG tool
        rag_tool = RAGTool(knowledge_dir=self.knowledge_dir)
        
        return Agent(
            config=self.agents_config['customer_service'],  # type: ignore[index]
            verbose=True,
            llm=self.ollama_llm,
            tools=[rag_tool]  # Add the RAG tool to the customer service agent
        )
        
    @agent
    def banking_analyst(self) -> Agent:
        # Create the RAG tool
        rag_tool = RAGTool(knowledge_dir=self.knowledge_dir)
        
        return Agent(
            config=self.agents_config['banking_analyst'],  # type: ignore[index]
            verbose=True,
            llm=self.ollama_llm,
            tools=[rag_tool]  # Add the RAG tool to the banking analyst agent
        )

    @task
    def financial_advisory_task(self) -> Task:
        return Task(
            config=self.tasks_config['financial_advisory_task'],  # type: ignore[index]
        )

    @task
    def customer_service_task(self) -> Task:
        return Task(
            config=self.tasks_config['customer_service_task'],  # type: ignore[index]
        )
        
    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['analysis_task'],  # type: ignore[index]
            output_file='banking_report.md'
        )
        
    @task
    def knowledge_query_task(self) -> Task:
        """Create a task specifically for querying banking knowledge"""
        return Task(
            config=self.tasks_config['knowledge_query_task'],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BankAgent crew with RAG capabilities"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
        
    def answer_question(self, question: str, topic: str = "Banking Services") -> str:
        """
        Answer a specific banking question using the knowledge base
        
        Args:
            question: The user's banking-related question
            topic: The banking topic context
        
        Returns:
            The answer from the financial advisor agent
        """
        # Create a dynamic task for the question
        question_task = Task(
            description=f"Answer the following banking question using the Knowledge Base Query Tool:\n\n{question}",
            expected_output="A detailed and accurate answer based on the banking knowledge base",
            agent=self.financial_advisor
        )
        
        # Create a mini-crew with just this task
        mini_crew = Crew(
            agents=[self.financial_advisor()],
            tasks=[question_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew with the question
        result = mini_crew.kickoff(inputs={'topic': topic})
        return result