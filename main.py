from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    information = """
        Ugyen Dendup is the CEO and co-founder of NoMindBhutan, the first AI company in Bhutan. 
        Here are some key details about him and the company:
        Co-founder: He co-started NoMindBhutan in 2022 while still a student at the Gyalpozhing College of Information Technology (GCIT).
        Company focus: NoMindBhutan develops AI tools and software, such as chatbots and automation systems, for Bhutanese organizations. Its clients have included the Bhutan National Bank, the national airline Drukair, and the Ministry of Industry, Commerce and Employment.
        Background: As a student, Dendup led a robotics club that competed internationally. This helped fuel his passion for technology and problem-solving.
        Recent ventures: In late 2024, he also began working on NoPay, a money platform designed to make sending money to Bhutan from overseas faster, safer, and more affordable 
    """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    
    # llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
    # llm = ChatOllama(temperature=0, model="llama3.1:8b")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
