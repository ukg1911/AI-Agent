import os
from crewai import LLM
from crewai import Agent, Crew, Task
from logging_config import get_logger

class ResearchCrew:
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        # Note: I removed the misplaced gemini_llm code from here
        self.crew = self.create_crew() 
        self.logger.info("ResearchCrew initialized")

    def create_crew(self):
        self.logger.info("Creating research crew with agents")
        
        # --- SWITCH TO GROQ (Free & Fast) ---
        groq_llm = LLM(
            model="openai/llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        researcher = Agent(
            role='Research Analyst',
            goal='Find and analyze key information',
            backstory='Expert at extracting information',
            llm=groq_llm,  # 2. Assign the brain (Required Change)
            verbose=self.verbose
        )

        writer = Agent(
            role='Content Summarizer',
            goal='Create clear summaries from research',
            backstory='Skilled at transforming complex information',
            llm=groq_llm,  # 2. Assign the brain (Required Change)
            verbose=self.verbose
        )

        self.logger.info("Created research and writer agents")

        crew = Crew(
            agents=[researcher, writer],
            tasks=[
                Task(
                    description='Research: {text}',
                    expected_output='Detailed research findings about the topic',
                    agent=researcher
                ),
                Task(
                    description='Write summary',
                    expected_output='Clear and concise summary of the research findings',
                    agent=writer
                )
            ]
        )
        self.logger.info("Crew setup completed")
        return crew