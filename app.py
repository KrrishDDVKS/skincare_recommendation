import streamlit as st
import re
import PIL.Image
from tqdm import tqdm
import os
from dotenv import dotenv_values
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import torch
import clip
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Optional,List
from typing import Annotated
import operator

#_________________________________State Variable_________________________
class State(BaseModel):
  image: Optional[str] = None
  eligible:bool=False
  age: int=0
  gender: Optional[str] = None
  bauman_type: Annotated[str, operator.add]
  skin_disease: Optional[str] = None
  meds: List[str] = []
  medimage: Optional[str] = None
  des: Optional[str] = None
  toxic: Optional[str] = None

#___________________________Vector Search________________________________
def RAG(image,index_name,api):
  image = preprocess(PIL.Image.open(image)).unsqueeze(0).to(device)
  pc = Pinecone(api_key=api)
  index = pc.Index(index_name)
  with torch.no_grad():
      image_features = model.encode_image(image)

  image_features /= image_features.norm(dim=-1, keepdim=True)
  query_vector = image_features.cpu().numpy().flatten().tolist()
  results = index.query(
      vector=query_vector,
      top_k=1,
      include_metadata=True
  )
  results = results["matches"]
  return results
#____________________________product Recommender using LLM________________________________________________________________________________________________
def chat(state:State):
  messages=[
          SystemMessage(content="Act as a loreal paris skin care product recommender."),
          HumanMessage(content=f"Please suggest a loreal skin care product for a {state.gender} of age {state.age} and skin type is {state.bauman_type}.")
      ]

  chat_response = llm.invoke(messages)
  return {'des':chat_response.content}
#____________________________Extraction of product names from LLM Response using RegEX__________________________________________________________________________________
def extract(state:State):
  matches = re.findall(r"\*\*(.*?)\*\*", state.des)
  return({'meds':matches})


# --- Page Config ---
st.set_page_config(
    page_title="MediDet-AI",
    page_icon="🕵️‍♂️",
    layout="wide",
)

global uploaded
uploaded=False

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# --- Main Panel: Title & Chat ---
st.markdown(
    "<h1 style='font-family: \"Special Elite\"; font-size: 36px; color: #f40000; text-align: center;'>Tech Beautician: A Skin Care Enhancer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 style='font-family: \"Special Elite\"; font-size: 24px;color: #f4d000; text-align: center;'>with a medical assistant for your skin</h2>",
    unsafe_allow_html=True
)

config = dotenv_values("keys.env")
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

index_name = "disease-symptoms-gpt-4"

embed = OpenAIEmbeddings(
model='text-embedding-ada-002',
openai_api_key=os.environ.get('OPEN_API_KEY')
)

llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],
                   model_name='gpt-4o',
                   temperature=0.0)


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please upload your face image through side bar"
}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Sidebar: Detective Bubble & Upload ---
with st.sidebar:
    st.markdown(
    "<h3 style='font-family: \"Special Elite\"; font-size: 20px; color: #00d000;text-align: left;'>💬 Lets Beautify you</h3>",
    unsafe_allow_html=True)
    bubble_html = """
    <div class="container">
        <div class="emoji">🕵️‍♂️</div>
        <div class="bubble">
            Got a mystery on your skin? Upload the clues. I'll investigate. 🔍
        </div>
    </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Upload or Capture Evidence", unsafe_allow_html=True)
    option = st.radio("Choose method:", ["Upload Image", "Open Camera"])
    st.markdown("</div>", unsafe_allow_html=True)

#______________________________________________________________________________________________________
    image_data = None
    if option == "Upload Image":
        uploaded = st.file_uploader("Drop the evidence (jpg/png)", type=['jpg', 'png'])
        if uploaded:
            image_data = PIL.Image.open(uploaded)
            st.image(image_data, caption="📁 Exhibit A", use_column_width=True)
            
            
            image = image_data.convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vector = image_features.cpu().numpy().flatten()
                st.success("✅ Image converted to CLIP vector!")
                st.write("Vector (first 10 values):", vector.shape)
                index_name = "skindisease-symptoms-gpt-4"
                pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
                index = pc.Index(index_name)
                rv=vector.reshape(1, -1)
                result= index.query(vector=rv.tolist(), top_k=1, include_metadata=True)
                prompt=result['matches'][0]['metadata']['Disease']
                st.write(prompt)
                messages=[SystemMessage(content="Accept the user’s skin condition as input and provide probable diagnoses and prescription for only that condition."),
                          HumanMessage(content=prompt)]
                chat_response = llm.invoke(messages)
                answer=chat_response.content
                st.session_state.messages.append({"role": "assistant", "content": answer})
#___________________________________________________________________________________________________________________________________________________________________
    elif option == "Open Camera":
        
        cam = st.camera_input("Live Surveillance")
        if cam:
            image = PIL.Image.open(cam)
            st.image(image, caption="📸 Snapshot captured!", use_column_width=True)
            image_data = image
            
            
            image = image_data.convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vector = image_features.cpu().numpy().flatten()
                st.success("✅ Image converted to CLIP vector!")
                st.write("Vector (first 10 values):", vector.shape)
                index_name = "skindisease-symptoms-gpt-4"
                pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
                index = pc.Index(index_name)
                rv=vector.reshape(1, -1)
                result= index.query(vector=rv.tolist(), top_k=1, include_metadata=True)
                prompt=result['matches'][0]['metadata']['Disease']
                st.write(prompt)
                messages=[SystemMessage(content="Accept the user’s skin condition as input and provide probable diagnoses and prescription for only that condition."),
                          HumanMessage(content=prompt)]
                chat_response = llm.invoke(messages)
                answer=chat_response.content
                st.session_state.messages.append({"role": "assistant", "content": answer})


    
if option == "Open Camera" and cam:
        st.chat_message("assistant").write(answer)
if uploaded:
        st.chat_message("assistant").write(answer)






