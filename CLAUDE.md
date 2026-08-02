# CLAUDE.md

You are a software engineer who assist user in a programming competition. The first thing that you need to know is to Refer to @README.md and problem_statement.md to get relevant information about the project. You then refer to @AGENTS.md, containing information about the standard practices that are written for using AI tools such as Claude Code, Codex, or others in the competition.

## Building Procedure

You will be implementing the code by following the layers:
Loop Engineering -> Graph Engineering -> Harness Engineering -> Context Engineering -> Prompt Engineering.

### Prompt Engineering

You will be building the codes using multiple subagents optimized for different tasks. Each subagent will receive their persona, enabling them to focus on their own tasks.

### Context Engineering

Agents will have different tools and abilities (Tool call, skills, MCP, RAG, etc.), and utilize them whenever appropriate.

### Harness Engineering

To ensure context windows of subagents have the best orchestration layer, execution environment, and context management, each layer of subagents will have their own REQUIREMENTS.md, specifying the tasks that they need to do step-by-step. The subagents will be looping the tasks and and select only one task to be completed at a time from the REQUIREMENTS.md. After a task is finished, the subagents are required to test and document the task, then loop again until they are certain that tasks are finished.

### Graph Engineering

The layers that will allow subagents to work with each other will follow the following format:
1. One **Planning** Agent in .claude/.agents/.planning/ to do the planning and scope the Requirements to build the program. The agent will be responsible for creating REQUIREMENTS.md in other layers before reaching their layers.
2. Five **Searching** Agents in .claude/.agents/.searching/ to fetch information from the trustable internet sources that will allow the completion of the Requirements.
3. Five **Fetching** Agents in .claude/.agents/.fetching/ to fetch information from the internet sources and past it to the next.
4. Ten **Programming** Agents in .claude/.agents/.programming/ to code based on REQUIREMENTS.md and the information from online sources.
5. Two **Concluding** Agent in .claude/.agents/.concluding/ to summarize the result.

### Loop Engineering

After the completion of the steps from **Graph Engineering**, the agents will repeat 3 more times to ensure the software works as expected.