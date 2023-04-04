from aleph_alpha_client import (
    Client,
    Prompt,
    CompletionRequest,
    CompletionResponse,
    SemanticEmbeddingRequest,
    SemanticEmbeddingResponse,
    SemanticRepresentation,
)

import streamlit as st
from streamlit_chat import message
from scipy.spatial.distance import cosine

# instantiate aleph alpha client
client = Client(token="")


def simple_completion(input: str):

    # A simple prompt for chatting
    prompt = f"""### Instruction: This is a simple chatbot. Answer the users question.

### Input: User:{input}


### Response: Bot:"""

    # define the parameters
    request = CompletionRequest(prompt=Prompt.from_text(prompt), maximum_tokens=64)

    # send the request to Aleph Alpha
    result = client.complete(request=request, model="luminous-base-control-beta")

    # get the result
    answer = result.completions[0].completion

    return answer


def simple_search(documents: list[str], query: str):

    # embed the documents
    embedded_documents = []

    for document in documents:
        document_params = {
            "prompt": Prompt.from_text(document),
            "representation": SemanticRepresentation.Document,
            "compress_to_size": 128,
        }
        document_request = SemanticEmbeddingRequest(**document_params)
        document_response = client.semantic_embed(
            request=document_request, model="luminous-base"
        )
        embedded_documents.append(document_response.embedding)

    # embed the query
    query_params = {
        "prompt": Prompt.from_text(query),
        "representation": SemanticRepresentation.Query,
        "compress_to_size": 128,
    }

    query_request = SemanticEmbeddingRequest(**query_params)
    query_response = client.semantic_embed(request=query_request, model="luminous-base")
    embedded_query = query_response.embedding

    # calculate the cosine similarity between the query and the documents
    search_scores = [
        1 - cosine(embedded_query, embedded_document)
        for embedded_document in embedded_documents
    ]

    # get the document with the highest score
    context = documents[search_scores.index(max(search_scores))]

    return context


##################################################################################

# streamlit chat interface
st.title("Hackathon")


# Storing the chat
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


# containe for the chat
cont = st.container()

user_input = get_text()

if user_input:
    with cont:
        # append the user's input and the generated output to the chat
        history = ""
        for user, bot in zip(st.session_state["past"], st.session_state["generated"]):
            history += f"User: {user}\nBot: {bot}\n"
            print(history)

        output = simple_completion(f"{history} + {user_input}")
        # store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

        if st.session_state["generated"]:

            # the chat messages are displayed in reverse order, so we reverse the list

            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))
