from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()

embedder_config={
        "provider": "custom",
        "config": {
            "embedder": default_ef
        }
    }

# Create a knowledge source for crew
content = "User name is Jérôme. He is 40 years old and lives in Caen."
string_source = StringKnowledgeSource(
    content=content,
    chunk_size=4000,      # Maximum size of each chunk (default: 4000)
    chunk_overlap=200     # Overlap between chunks (default: 200)
)


@CrewBase
class Ma4():


    @agent
    def about_user(self) -> Agent:
        # Define the manager agent
        about_user = Agent(
            role="About User",
            goal="You know everything about the user.",
            backstory="You are a master at understanding people and their preferences.",
            allow_delegation=False,
            verbose=True,
            llm = LLM(
                model="mistral/mistral-large-latest",
                temperature=0,
                top_p=1
            )
        )
        return about_user   




    @task
    def agent_writer_task(self) -> Task:
        # Define your task
        task = Task(
            description="""Answer the following questions about the user: {question}""",
            expected_output="""An answer to the question.""",
            agent=self.about_user()
        )
        return task
   

    @crew
    def crew(self) -> Crew:
        """Creates the Ma4 crew"""
    
        return Crew(
            agents=self.agents,         
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            memory=True, # by default storage is in ~/.local/share/<application_name>/ for linux
            embedder=embedder_config,
            knowledge_sources=[string_source]
        )

