# Multiagent 

Here are some basic implementation examples of a multi-agent LLM system based on the CrewAI documentation (https://docs.crewai.com/introduction).

- ma: basic example
- ma1: basic example with manager agent
- ma2: basic example with manager agent and specific LLM
- ma3: basic example with tools
- ma4: basic example with memory and knowledge (see also 'local memory' chapter with chromadb and mysql)

### Run

The examples are using the mistral.ai API : https://console.mistral.ai/home . You need only a free experiment account.

- Copy .env_example and rename in .env 
- Add your API_KEY in your .env

```
cd <'ma' project>
crewai install
crewai run
```

##### Local memory

Launch mysql docker-compose :

```
cd ma4
docker compose -f mysql/docker-compose.yml up

# Create the long_term_memories database:

mysql -u root -p'example' -h 127.0.0.1 -P 3306
mysql> CREATE DATABASE long_term_memories;

```

Launch chromadb docker-compose :

```
cd ma4
docker compose -f chromadb/docker-compose.yml up
```