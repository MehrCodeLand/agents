# Path: mycdagent/src/mycdagent/crew.py

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any
import os

# Import the RAG Tool
from tools.rag_tool import RAGTool

@CrewBase
class Mycdagent():
    """Mycdagent crew with RAG capability"""

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
    def researcher(self) -> Agent:
        # Create the RAG tool
        rag_tool = RAGTool(knowledge_dir=self.knowledge_dir)
        
        return Agent(
            config=self.agents_config['researcher'],  # type: ignore[index]
            verbose=True,
            llm=self.ollama_llm,
            tools=[rag_tool]  # Add the RAG tool to the researcher agent
        )

    @agent
    def reporting_analyst(self) -> Agent:
        # Create the RAG tool
        rag_tool = RAGTool(knowledge_dir=self.knowledge_dir)
        
        return Agent(
            config=self.agents_config['reporting_analyst'],  # type: ignore[index]
            verbose=True,
            llm=self.ollama_llm,
            tools=[rag_tool]  # Add the RAG tool to the reporting agent
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],  # type: ignore[index]
            output_file='report.md'
        )
        
    @task
    def knowledge_query_task(self) -> Task:
        """Create a task specifically for querying knowledge"""
        return Task(
            description=(
                "Research the user's question using the Knowledge Base Query Tool. "
                "First, understand what information the user is looking for. "
                "Then, use the Knowledge Base Query Tool to find relevant information. "
                "Finally, synthesize a comprehensive answer based on the retrieved information."
            ),
            expected_output=(
                "A detailed answer to the user's question based on the knowledge base information. "
                "The answer should be well-structured and include all relevant details."
            ),
            agent=self.researcher,
            context=[]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Mycdagent crew with RAG capabilities"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
        
    def answer_question(self, question: str, topic: str = "General Knowledge") -> str:
        """
        Answer a specific question using the knowledge base
        
        Args:
            question: The user's question
            topic: The topic context
        
        Returns:
            The answer from the agent
        """
        # Create a dynamic task for the question
        question_task = Task(
            description=f"Answer the following question using the Knowledge Base Query Tool:\n\n{question}",
            expected_output="A detailed and accurate answer based on the knowledge base",
            agent=self.researcher
        )
        
        # Create a mini-crew with just this task
        mini_crew = Crew(
            agents=[self.researcher()],
            tasks=[question_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew with the question
        result = mini_crew.kickoff(inputs={'topic': topic})
        return result