# from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain")

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma2:9b")


    chain = summary_prompt_template | llm
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/ritu-raj13/"
    )
    res = chain.invoke(input={"information": linkedin_data})

    print(res)
