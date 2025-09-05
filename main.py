from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from shemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "format_instructions"]
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm, tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(
        input={
            "input":"What is the weather right now in Thimphu, Bhutan?"
        }
    )
    print(result)

if __name__ == "__main__":
    main()