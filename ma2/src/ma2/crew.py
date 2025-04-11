from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from crewai import LLM


@CrewBase
class Ma2():
    """Ma2 crew"""


    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    # agents_config = 'config/agents.yaml'
    # tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        # Define your agents
        researcher = Agent(
            role="Researcher",
            goal="Conduct thorough research and analysis on AI and AI agents",
            backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
            allow_delegation=False,
            #knowledge_sources=[self.string_source],
            llm = LLM(
                model="mistral/mistral-large-latest",
                temperature=0.7
            )
        )
        return researcher


    @agent
    def writer(self) -> Agent:
        writer = Agent(
            role="Senior Writer",
            goal="Create compelling content about AI and AI agents",
            backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
            allow_delegation=False,
            llm = LLM(
                model="mistral/mistral-large-latest",
                temperature=0.1
            )
        )
        return writer
    
    @agent
    def manager(self) -> Agent:
        # Define the manager agent
        manager = Agent(
            role="Project Manager",
            goal="Efficiently manage the crew and ensure high-quality task completion",
            backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
            allow_delegation=True,
        )
        return manager


    @task
    def research_task(self) -> Task:
        # Define your task
        task = Task(
            description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
            expected_output="5 bullet points, each with a paragraph and accompanying notes.",
        )
        return task

    @crew
    def crew(self) -> Crew:
        """Creates the Ma1 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=[self.researcher(), self.writer()], # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            #process=Process.sequential,
            manager_agent=self.manager(),
            verbose=True,
            process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
