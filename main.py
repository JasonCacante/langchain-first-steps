import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
Elon Reeve Musk (Pretoria, 28 de junio de 1971) es un empresario, inversor y magnate.[nota 1]​ Es el fundador, consejero delegado e «ingeniero» en jefe de la empresa SpaceX; inversor ángel, director general y arquitecto de productos de Tesla, Inc.; fundador de The Boring Company; y cofundador de Neuralink y OpenAI.[nota 2]​ Además, es el director de tecnología de X Corp.[3]​ Entre enero y mayo de 2025, ejerció como administrador de facto del Departamento de Eficiencia Gubernamental de la Casa Blanca bajo la segunda presidencia de Donald Trump.
    """
    summary_template = """
    given the information {information}, about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them
"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOllama(model="llama3", temperature=0)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
