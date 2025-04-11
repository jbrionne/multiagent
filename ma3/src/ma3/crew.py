from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from ma3.tools.custom_tool import MyCodeInterpreterTool

from crewai import LLM

code_interpreter = MyCodeInterpreterTool()

@CrewBase
class Ma3():
    """Ma3 crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    # agents_config = 'config/agents.yaml'
    # tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def coding_agent(self) -> Agent:
        # Define your agents
        coding_agent = Agent(
            role="Python Data Analyst",
            goal="Analyze data and provide insights using Python",
            backstory="You are an experienced data analyst with strong Python skills.",
            allow_delegation=False,
            allow_code_execution=True,
            tools=[code_interpreter],
            verbose=True,
            #knowledge_sources=[self.string_source],
            llm = LLM(
                model="mistral/codestral-latest",
                temperature=0.5
            )
        )
        return coding_agent

    
    @task
    def data_analysis_task(self) -> Task:
        # Define your task
        task = Task(
            description="Analyze the given dataset and calculate the average age of participants. Ages: {ages}",
            expected_output="The average age calculated from the dataset",
            agent=self.coding_agent(),
            human_input=True
        )
        return task

    @crew
    def crew(self) -> Crew:
        """Creates the Ma3 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            #manager_agent=self.manager(),
            verbose=True,
            #process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
