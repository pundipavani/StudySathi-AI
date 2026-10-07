import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# ============================================================
# StudySathi AI - Personal AI Learning Companion for B.Tech
# Requirements: only the packages listed in requirements.txt.
# ============================================================

st.set_page_config(page_title="StudySathi AI", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not API_KEY:
    try:
        API_KEY = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
    except Exception:
        API_KEY = ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

DATA_DIR = Path("studysathi_data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "chroma_db"
PROGRESS_FILE = DATA_DIR / "progress.json"

# ----------------------------- UI -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
*{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 5% 5%,#ebe7ff 0,transparent 28%),radial-gradient(circle at 95% 8%,#ffe7f2 0,transparent 26%),#f8f9fc}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#182033}
.hero{padding:40px 34px;border-radius:28px;background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(245,242,255,.94));border:1px solid rgba(80,70,160,.10);box-shadow:0 14px 45px rgba(24,32,51,.07);margin-bottom:24px}
.gradient-title{font-size:clamp(44px,7vw,80px);text-align:center;letter-spacing:-4px;background:linear-gradient(135deg,#5145df,#8b4fe9,#db4f9d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.home-center{text-align:center;padding:22px 0}.kicker{color:#6255ee;font-size:11px;font-weight:700;letter-spacing:1.5px}
.card{background:rgba(255,255,255,.92);border:1px solid rgba(70,78,110,.10);border-radius:20px;padding:20px;box-shadow:0 10px 35px rgba(24,32,51,.05)}
.feature-card{min-height:150px}.feature-card p{color:#6c7485;line-height:1.55}
.metric{background:rgba(255,255,255,.92);border:1px solid rgba(70,78,110,.10);border-radius:18px;padding:18px}.metric small{color:#70798d;display:block}.metric b{display:block;font:700 30px 'Space Grotesk';color:#182033;margin-top:5px}
.answer-box{background:white;border:1px solid #e5e7f0;border-radius:18px;padding:22px;box-shadow:0 8px 25px rgba(24,32,51,.05);line-height:1.75}
.source{border-left:4px solid #6758ef;background:#6758ef0c;border-radius:9px;padding:9px 12px;margin:6px 0;font-size:12px}
.interview-q{background:linear-gradient(135deg,#fff,#f4f2ff);border:1px solid #e6e2ff;border-radius:20px;padding:22px;margin:12px 0}
.badge{display:inline-block;padding:5px 10px;border-radius:999px;background:#eeeaff;color:#5b4ee5;font-size:12px;font-weight:700}
.stButton>button{border-radius:12px!important;font-weight:700!important}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#6255f5,#a156e7)!important;color:white!important;border:0!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#11172b,#211b43)}section[data-testid="stSidebar"] *{color:#eef1ff!important}
.small-note{color:#747d90;font-size:13px}.footer{text-align:center;color:#8c94a6;font-size:11px;padding:25px}
.code-card{background:#111827;border-radius:16px;padding:16px;color:#f8fafc}.lesson{background:#fff;border:1px solid #e6e8f0;border-radius:18px;padding:22px;line-height:1.7}
</style>
""", unsafe_allow_html=True)

# ----------------------------- STATE -----------------------------
DEFAULTS = {
    "page": "Home", "chat": [], "quiz": [], "quiz_done": False, "score": 0,
    "quiz_topic": "", "quiz_difficulty": "Medium", "quiz_module": "Technical",
    "interview_questions": [], "interview_index": 0, "interview_answers": [],
    "interview_score": 0, "interview_started": False, "interview_finished": False,
    "interview_module": "Technical", "interview_role": "Software Engineer",
    "interview_level": "Beginner",
}
for k, v in DEFAULTS.items(): st.session_state.setdefault(k, v)

# ----------------------------- PROGRESS -----------------------------
def default_progress():
    return {"notes":0,"chat_questions":0,"quiz_questions":0,"quizzes":0,"correct":0,"code_questions":0,"interviews":0,"interview_answers":0,"learn_sessions":0,"achievements":[]}

def get_progress():
    if not PROGRESS_FILE.exists(): return default_progress()
    try:
        d=json.loads(PROGRESS_FILE.read_text(encoding="utf-8")); r=default_progress(); r.update(d); return r
    except Exception: return default_progress()

def save_progress(d): PROGRESS_FILE.write_text(json.dumps(d,indent=2),encoding="utf-8")

def inc(**vals):
    d=get_progress()
    for k,v in vals.items(): d[k]=d.get(k,0)+v
    save_progress(d)

def achievement(name):
    d=get_progress(); d.setdefault("achievements",[])
    if name not in d["achievements"]: d["achievements"].append(name); save_progress(d)

# ----------------------------- LAZY RAG -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_collection():
    client=chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name="studysathi_documents",metadata={"hnsw:space":"cosine"})

def normalize(text): return re.sub(r"\s+"," ",(text or "").replace("\x00"," ")).strip()

def extract_chunks(uploaded_file):
    reader=PdfReader(uploaded_file); out=[]; size=900; overlap=120; step=size-overlap
    for page_no,page in enumerate(reader.pages,1):
        text=normalize(page.extract_text() or "")
        if not text: continue
        for start in range(0,len(text),step):
            piece=text[start:start+size]
            if len(piece.strip())<60: continue
            out.append({"text":piece,"page":page_no})
            if start+size>=len(text): break
    return out

def add_pdf(file):
    try:
        chunks=extract_chunks(file)
        if not chunks: return False,"No readable text was found in this PDF. It may be scanned/image-only.",0
        col=get_collection(); emb=get_embedder()
        name=file.name; doc_id=hashlib.sha1(name.encode()).hexdigest()[:16]
        existing=col.get(where={"document_id":doc_id},include=["metadatas"])
        if existing.get("ids"): return True,f"{name} is already in your notes.",0
        vectors=emb.encode([c["text"] for c in chunks],normalize_embeddings=True).tolist()
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        metas=[{"document_id":doc_id,"filename":name,"page":c["page"]} for c in chunks]
        col.add(ids=ids,documents=[c["text"] for c in chunks],embeddings=vectors,metadatas=metas)
        inc(notes=1); achievement("First Notes Upload")
        return True,f"Added {name} • {len(chunks)} chunks",len(chunks)
    except Exception as e: return False,f"Could not process the PDF: {e}",0

def retrieve(question,k=5):
    try:
        col=get_collection()
        if col.count()==0: return []
        vec=get_embedder().encode([question],normalize_embeddings=True).tolist()
        r=col.query(query_embeddings=vec,n_results=min(k,col.count()),include=["documents","metadatas","distances"])
        return [{"text":d,"meta":m,"distance":dist} for d,m,dist in zip(r["documents"][0],r["metadatas"][0],r["distances"][0])]
    except Exception: return []

# ----------------------------- GEMINI -----------------------------
def gemini_text(prompt):
    if not API_KEY: return None
    try:
        client=genai.Client(api_key=API_KEY)
        response=client.models.generate_content(model=GEMINI_MODEL,contents=prompt)
        text=getattr(response,"text",None)
        return text.strip() if text else None
    except Exception as e:
        st.session_state["gemini_error"]=str(e)
        return None

def clean_answer(text):
    if not text:
        return ""
    text=re.sub(r"^\s*(Answer|Response)\s*:\s*", "", text, flags=re.I)
    text=re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
    text=text.replace("```", "")
    text=re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text=re.sub(r"\*{1,3}", "", text)
    text=re.sub(r"_{1,3}", "", text)
    text=re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text=re.sub(r"^\s*>\s?", "", text, flags=re.M)
    text=re.sub(r"\s+", " ", text)
    return text.strip()

# ----------------------------- HOME -----------------------------
def home():
    st.markdown('<div class="home-center"><div class="kicker">YOUR B.TECH LEARNING COMPANION</div><h1 class="gradient-title">StudySathi AI</h1><p>Learn from notes, practise coding, prepare for exams and build interview confidence in one place.</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="hero"><h2>About this Project</h2><p><b>StudySathi AI</b> is an academic learning platform designed for B.Tech students. It combines document-based learning, exam preparation, PDF-based quizzes and interview preparation in one simple workspace.</p><p>The system can read uploaded academic PDFs, retrieve relevant passages using semantic search, and use an AI model when available to produce complete, context-aware answers. It also provides deterministic learning material and practice content so the interface remains useful even when an external AI quota is unavailable.</p><p><b>Goal:</b> help students learn concepts, practise technical skills and prepare for placements without switching between many tools.</p></div>',unsafe_allow_html=True)
    features=[("📚","My Notes","Upload academic PDFs and build your personal study library."),("💬","AI Study Chat","Ask questions and receive complete explanations from your notes."),("📝","Exam Preparation","Generate medium-length, exam-ready answers from uploaded notes."),("🎯","Quiz Zone","Create questions and answers directly from your uploaded PDFs."),("🎤","AI Interview","Practise Technical, Communication, HR, Aptitude and AI/ML interview modules."),("📊","My Progress","Track your study questions, quizzes and interview practice.")]
    cols=st.columns(4)
    for i,(icon,title,desc) in enumerate(features):
        with cols[i%4]: st.markdown(f'<div class="card feature-card"><h3>{icon} {title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
        if i%4==3: st.write("")

# ----------------------------- NOTES -----------------------------
def notes_page():
    st.title("📚 My Notes")
    st.write("Upload academic PDFs. StudySathi indexes them for document-grounded learning.")
    files=st.file_uploader("Upload PDF notes",type=["pdf"],accept_multiple_files=True)
    if files and st.button("✨ Process Notes",type="primary"):
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                ok,msg,_=add_pdf(f)
            if ok: st.success(msg)
            else: st.error(msg)
    try:
        c=get_collection(); count=c.count()
        st.info(f"Indexed chunks: {count}")
        if count:
            data=c.get(limit=min(1000,count),include=["metadatas"])
            names=sorted(set(m.get("filename","") for m in data.get("metadatas",[]) if m))
            if names:
                st.subheader("Your indexed PDFs")
                for n in names: st.write("•",n)
    except Exception as e: st.warning(f"Notes database is not ready yet: {e}")

# ----------------------------- CHAT -----------------------------
def chat_page():
    st.title("💬 AI Study Chat")
    if not st.session_state.chat: st.caption("Ask a question from your uploaded notes.")
    for item in st.session_state.chat:
        with st.chat_message(item["role"]): st.markdown(item["content"])
    q=st.chat_input("Ask a question about your notes...")
    if q:
        st.session_state.chat.append({"role":"user","content":q}); inc(chat_questions=1); achievement("First Study Question")
        hits=retrieve(q,5)
        if not hits:
            answer="I could not find relevant content in your uploaded notes. Please upload the correct PDF or ask a question related to the indexed material."
            sources=[]
        else:
            context="\n\n".join(f"[Page {h['meta'].get('page','?')}] {h['text']}" for h in hits)
            prompt=f"""You are StudySathi AI, an exact academic tutor. Answer the student's question completely and directly using ONLY the supplied notes. Do not invent facts. Do not repeat the question. Do not stop after a definition. Give a medium-length complete explanation. Use full words instead of contractions or unexplained abbreviations. Use plain text only. Do not use Markdown symbols such as #, *, _, backticks, bullets, or decorative symbols. Write a clean medium-length answer using short paragraphs. Include a definition, explanation, important points, steps, and a suitable example only when supported by the notes. Do not stop after one or two sentences. If the notes do not contain enough information, explicitly say what is missing.\n\nSTUDENT QUESTION:\n{q}\n\nNOTES:\n{context}"""
            answer=clean_answer(gemini_text(prompt))
            if not answer:
                answer="Answer. "+" ".join(h["text"] for h in hits)
            sources=[h["meta"] for h in hits]
        st.session_state.chat.append({"role":"assistant","content":answer})
        st.rerun()

# ----------------------------- EXAM -----------------------------
def exam_page():
    st.title("📝 Exam Preparation")
    q=st.text_area("Enter the exact exam question",height=110,placeholder="Example: Explain normalization in DBMS and describe 1NF, 2NF and 3NF.")
    marks=st.selectbox("Answer format",["2 Marks","5 Marks","10 Marks"])
    if st.button("✨ Generate Complete Answer",type="primary",disabled=not q.strip()):
        hits=retrieve(q,7)
        if not hits: st.error("No relevant notes found. Upload the subject PDF first."); return
        context="\n\n".join(f"[Page {h['meta'].get('page','?')}] {h['text']}" for h in hits)
        prompt=f"""You are a strict B.Tech university exam answer writer. Answer the EXACT question below using ONLY the notes. The response must be complete, accurate and exam-ready. Never give a half answer. Never repeat the question or the same point. Use full words instead of contractions or unexplained abbreviations. For {marks}, write a medium-length answer appropriate to the marks: 2 marks = concise definition plus essential explanation; 5 marks = definition, clear explanation, important points or steps, and a suitable example if present; 10 marks = introduction or definition, detailed explanation with headings, components or steps, example or diagram description if supported, advantages or limitations or applications when supported, and conclusion. Use complete sentences and clear academic language. Do not add facts absent from the notes.\n\nQUESTION:\n{q}\n\nNOTES:\n{context}"""
        answer=clean_answer(gemini_text(prompt))
        if not answer:
            answer="Complete Answer. "+" ".join(h["text"] for h in hits)
        st.markdown('<div class="answer-box">'+answer.replace("\n","<br>")+"</div>",unsafe_allow_html=True)
        with st.expander("🔎 Sources used"):
            for h in hits: st.markdown(f"<div class='source'><b>{h['meta'].get('filename')}</b> — Page {h['meta'].get('page')}</div>",unsafe_allow_html=True)

# ----------------------------- QUIZ -----------------------------
def generate_note_quiz(topic, n=5):
    topic=topic.strip()
    if not topic:
        return [], "Please enter a topic name first."
    hits=retrieve(topic, max(10,n*3))
    if not hits:
        return [], f"No PDF content related to '{topic}' was found. Upload and process the relevant PDF first."
    context="\n\n".join(f"[Page {h['meta'].get('page','?')}] {h['text']}" for h in hits)
    prompt=f"""Create exactly {n} high-quality multiple-choice questions about the topic "{topic}".
Use ONLY the supplied PDF content. Every question must be directly supported by the PDF and specifically related to the requested topic.
Do not use general knowledge that is not present in the PDF. Do not repeat questions or make generic questions.
Each question must have exactly four options and exactly one correct answer.
Return ONLY a JSON array. Each object must contain: question, options, answer, explanation.
The explanation must be short and supported by the PDF.
Do not include markdown, headings, or code fences.

REQUESTED TOPIC:
{topic}

PDF CONTENT:
{context}"""
    raw=gemini_text(prompt)
    if raw:
        try:
            raw=re.sub(r"```json|```", "", raw).strip()
            data=json.loads(raw)
            valid=[]; seen=set()
            for x in data:
                q=str(x.get("question","")).strip(); opts=x.get("options",[])
                ans=str(x.get("answer","")).strip(); exp=str(x.get("explanation","")).strip()
                if q and q.lower() not in seen and isinstance(opts,list) and len(opts)==4 and all(str(o).strip() for o in opts) and ans in opts and exp:
                    valid.append({"question":q,"options":[str(o) for o in opts],"answer":ans,"explanation":exp}); seen.add(q.lower())
                if len(valid)>=n: break
            if len(valid)==n: return valid, ""
        except Exception:
            pass
    return [], "The AI service could not create a reliable quiz for this topic. Please check your Gemini API quota and try again."

def quiz_pdf(quiz, title):
    sections=["QUESTIONS AND CORRECT ANSWERS",""]
    for i,x in enumerate(quiz,1):
        sections += [f"Q{i}. {x['question']}", *(f"   {chr(65+j)}. {o}" for j,o in enumerate(x['options'])), f"Answer: {x['answer']}", f"Explanation: {x['explanation']}", ""]
    return make_simple_pdf(title,sections)

def quiz_page():
    st.title("🎯 Quiz Zone")
    st.write("Enter a topic from your uploaded PDF and generate questions specifically from that topic.")
    topic=st.text_input("Enter topic name",placeholder="Example: Normalization in DBMS")
    n=st.selectbox("Number of questions",[5,10],index=0)
    if st.button("Generate Quiz",type="primary",disabled=not topic.strip()):
        with st.spinner(f"Reading your PDF notes and creating questions about {topic}..."):
            quiz,msg=generate_note_quiz(topic,n)
        if quiz:
            st.session_state.quiz=quiz; st.session_state.quiz_done=False; st.session_state.score=0; st.session_state.quiz_topic="Uploaded PDFs"; st.rerun()
        else: st.error(msg)
    if st.session_state.quiz:
        answers=[]
        for i,x in enumerate(st.session_state.quiz,1):
            st.markdown(f'<div class="interview-q"><span class="badge">Question {i}</span><h3>{x["question"]}</h3></div>',unsafe_allow_html=True)
            answers.append(st.radio("Choose one answer",x["options"],key=f"quiz_note_{i}",index=None))
        if st.button("Submit Quiz",type="primary"):
            if any(a is None for a in answers): st.warning("Please answer every question before submitting.")
            else:
                score=sum(a==x["answer"] for a,x in zip(answers,st.session_state.quiz)); st.session_state.score=score; st.session_state.quiz_done=True
                inc(quiz_questions=len(st.session_state.quiz),quizzes=1,correct=score); achievement("PDF Quiz Completed"); st.rerun()
    if st.session_state.quiz_done:
        st.success(f"Score: {st.session_state.score}/{len(st.session_state.quiz)}")
        st.subheader("Questions and Correct Answers")
        for i,x in enumerate(st.session_state.quiz,1):
            st.markdown(f"**Q{i}. {x['question']}**")
            st.write(f"**Correct answer:** {x['answer']}")
            st.write(f"**Explanation:** {x['explanation']}")
        pdf=quiz_pdf(st.session_state.quiz,"StudySathi AI - Quiz Questions and Answers From Uploaded Notes")
        st.download_button("📄 Download Questions and Answers PDF",data=pdf,file_name="StudySathi_PDF_Quiz_Questions_Answers.pdf",mime="application/pdf")

# ----------------------------- INTERVIEW -----------------------------
# Built-in interview MCQs. This replaces the removed FALLBACK_QUIZZES dependency.
INTERVIEW_BANK={
"Technical":[
 {"q":"Which data structure follows the Last In, First Out principle?","options":["Queue","Stack","Linked List","Tree"],"answer":"Stack","why":"A stack follows Last In, First Out, meaning the most recently added item is removed first."},
 {"q":"What is the main purpose of a database management system?","options":["To edit images","To store and manage data","To compile Python code","To design web pages"],"answer":"To store and manage data","why":"A database management system is used to store, organize, retrieve and manage data efficiently."},
 {"q":"Which language is commonly used for general-purpose programming and artificial intelligence?","options":["Python","HTML","CSS","SQL"],"answer":"Python","why":"Python is a general-purpose programming language widely used for software development, data science and artificial intelligence."},
 {"q":"What does SQL primarily work with?","options":["Databases","Images","Operating system drivers","Animations"],"answer":"Databases","why":"SQL is used to create, read, update and manage data in relational databases."},
 {"q":"Which concept allows a class to acquire properties and methods from another class?","options":["Encapsulation","Inheritance","Compilation","Iteration"],"answer":"Inheritance","why":"Inheritance allows a derived class to reuse and extend properties and methods of a base class."},
],
"Communication":[
 {"q":"Which is the best approach when answering an interview question you do not fully know?","options":["Invent an answer","Stay silent","Be honest and explain what you know","Change the topic"],"answer":"Be honest and explain what you know","why":"A professional response should be honest and should explain the relevant knowledge you do have."},
 {"q":"Which skill is important for effective communication?","options":["Active listening","Ignoring questions","Speaking without pauses","Avoiding eye contact"],"answer":"Active listening","why":"Active listening helps you understand the speaker and respond clearly and appropriately."},
 {"q":"What should a good interview introduction usually include?","options":["Only hobbies","Name, education, skills and career goal","Only family details","Only marks"],"answer":"Name, education, skills and career goal","why":"A concise introduction should present relevant academic background, skills and career direction."},
 {"q":"Which response is most professional when receiving feedback?","options":["Ignore it","Argue immediately","Listen and identify improvements","Leave the interview"],"answer":"Listen and identify improvements","why":"Professional candidates listen to feedback and use it to improve their performance."},
 {"q":"What helps make an interview answer clear?","options":["Long unrelated stories","A structured and direct response","Very low voice","Changing the subject"],"answer":"A structured and direct response","why":"A structured answer helps the interviewer understand the main point and supporting details."},
],
"HR & Behavioral":[
 {"q":"Why do interviewers ask about strengths?","options":["To test typing speed","To understand relevant abilities","To check internet speed","To avoid technical questions"],"answer":"To understand relevant abilities","why":"Strength questions help interviewers understand abilities that may support success in the role."},
 {"q":"What is a good way to answer a weakness question?","options":["Say you have no weaknesses","Mention a real weakness and how you are improving","Blame others","Avoid answering"],"answer":"Mention a real weakness and how you are improving","why":"A thoughtful answer shows self-awareness and a willingness to improve."},
 {"q":"How should you handle a disagreement in a team?","options":["Listen and discuss the issue professionally","Ignore the team","Insult the teammate","Stop working"],"answer":"Listen and discuss the issue professionally","why":"Professional discussion and listening help teams resolve disagreements constructively."},
 {"q":"What should you do before a company interview?","options":["Research the company and role","Avoid preparation","Memorize unrelated facts","Skip the job description"],"answer":"Research the company and role","why":"Research helps you understand the organization, role and expectations before the interview."},
 {"q":"What does teamwork involve?","options":["Working without communication","Collaborating toward a common goal","Avoiding responsibility","Competing with every teammate"],"answer":"Collaborating toward a common goal","why":"Teamwork means cooperating, communicating and contributing toward a shared objective."},
],
"Aptitude":[
 {"q":"If a number is increased from 100 to 120, what is the percentage increase?","options":["10%","15%","20%","25%"],"answer":"20%","why":"The increase is 20, and 20 divided by 100 multiplied by 100 gives 20%."},
 {"q":"What is the average of 10, 20 and 30?","options":["15","20","25","30"],"answer":"20","why":"The sum is 60 and 60 divided by 3 equals 20."},
 {"q":"If a car travels 60 kilometers in 2 hours, what is its average speed?","options":["20 km/h","30 km/h","40 km/h","120 km/h"],"answer":"30 km/h","why":"Average speed equals distance divided by time, so 60 divided by 2 equals 30 km/h."},
 {"q":"What is 15% of 200?","options":["20","25","30","35"],"answer":"30","why":"Fifteen percent of 200 is 0.15 multiplied by 200, which equals 30."},
 {"q":"If 5 workers complete a task in 10 days at the same rate, how many worker-days are required?","options":["15","25","50","100"],"answer":"50","why":"Worker-days equal number of workers multiplied by number of days: 5 multiplied by 10 equals 50."},
],
"Coding":[
 {"q":"Which keyword defines a function in Python?","options":["func","def","function","define"],"answer":"def","why":"Python uses the def keyword to define a function."},
 {"q":"Which symbol starts a comment in Python?","options":["//","#","<!--","/*"],"answer":"#","why":"A hash symbol starts a single-line comment in Python."},
 {"q":"Which Python collection stores key-value pairs?","options":["List","Tuple","Dictionary","Set"],"answer":"Dictionary","why":"A Python dictionary stores data as key-value pairs."},
 {"q":"What does len() return for a Python list?","options":["The first element","The last element","The number of elements","The data type"],"answer":"The number of elements","why":"The len() function returns the number of items in a collection such as a list."},
 {"q":"Which loop is commonly used to iterate over items in a Python collection?","options":["for","switch","repeat-until","foreach-only"],"answer":"for","why":"Python's for loop is commonly used to iterate through items in sequences and other iterables."},
],
"AI / ML":[
 {"q":"What is machine learning?","options":["A method where systems learn patterns from data","Only manual programming","A database language","A computer network"],"answer":"A method where systems learn patterns from data","why":"Machine learning enables systems to learn patterns from data and use them for predictions or decisions."},
 {"q":"Which type of learning uses labelled training data?","options":["Supervised learning","Unsupervised learning","Reinforcement learning only","Random learning"],"answer":"Supervised learning","why":"Supervised learning trains a model using examples that include input data and known target labels."},
 {"q":"What is classification used for?","options":["Predicting categories","Only sorting files","Increasing storage","Drawing diagrams"],"answer":"Predicting categories","why":"Classification predicts discrete categories such as spam or not spam."},
 {"q":"What is a feature in machine learning?","options":["An input variable used by a model","Only the final prediction","A programming language","A database table"],"answer":"An input variable used by a model","why":"A feature is an input variable or measurable property used by a machine learning model."},
 {"q":"Why is a test dataset used?","options":["To evaluate model performance on unseen data","To train the model twice","To delete the training data","To increase file size"],"answer":"To evaluate model performance on unseen data","why":"A test dataset helps measure how well a trained model generalizes to data it has not seen during training."},
],
}

def interview_page():
    st.title("🎤 B.Tech AI Interview")
    st.write("Choose the interview area you want to practise. The MCQ mode works even when the external AI quota is unavailable.")
    c1,c2,c3=st.columns(3)
    with c1: module=st.selectbox("Interview module",list(INTERVIEW_BANK.keys()),key="int_module_select")
    with c2: role=st.selectbox("Target role",["Software Engineer","Python Developer","AI/ML Engineer","Data Analyst","Fresher / Campus Placement"])
    with c3: level=st.selectbox("Level",["Beginner","Intermediate","Advanced"])
    mode=st.radio("Interview mode",["MCQ Practice","AI Mock Interview"],horizontal=True)
    if mode=="MCQ Practice":
        st.session_state.interview_module=module
        bank=INTERVIEW_BANK[module]
        if st.button("Start MCQ Module",type="primary"):
            st.session_state.interview_questions=bank[:min(5,len(bank))]; st.session_state.interview_answers=[]; st.session_state.interview_index=0; st.session_state.interview_finished=False; st.rerun()
        if st.session_state.interview_questions:
            answers=[]
            for i,x in enumerate(st.session_state.interview_questions,1):
                st.markdown(f'<div class="interview-q"><span class="badge">{module} • Q{i}</span><h3>{x["q"]}</h3></div>',unsafe_allow_html=True)
                answers.append(st.radio("Select your answer",x["options"],key=f"intmcq_{module}_{i}",index=None))
            if st.button("Submit Interview Module",type="primary"):
                if any(a is None for a in answers): st.warning("Answer every question before submitting.")
                else:
                    score=sum(a==x["answer"] for a,x in zip(answers,st.session_state.interview_questions)); inc(interviews=1,interview_answers=len(answers)); st.session_state.interview_score=score; st.session_state.interview_finished=True; achievement("Interview Module Completed"); st.rerun()
            if st.session_state.interview_finished:
                st.success(f"Module score: {st.session_state.interview_score}/{len(st.session_state.interview_questions)}")
                for i,x in enumerate(st.session_state.interview_questions,1): st.write(f"**Q{i}:** Correct answer — **{x['answer']}**. {x['why']}")
    else:
        q=st.text_area("Ask the AI interviewer a question or start a mock interview",height=110,placeholder="Example: Start my Software Engineer interview with an introduction question.")
        if st.button("Ask AI Interviewer",type="primary",disabled=not q.strip()):
            prompt=f"You are a professional B.Tech campus-placement interviewer. Target role: {role}. Level: {level}. Module: {module}. User request: {q}. Ask one focused interview question or evaluate the user's answer if they provided one. For evaluation, give score out of 10, strengths, exact improvements, communication feedback and a model answer. Do not give half answers."
            ans=gemini_text(prompt)
            if not ans: ans="AI Interviewer is temporarily unavailable because the configured AI service has no available quota. Use MCQ Practice above to continue practising." 
            st.markdown(f'<div class="answer-box">{ans.replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)

# ----------------------------- PROGRESS -----------------------------
def progress_page():
    st.title("📊 My Progress")
    d=get_progress(); cols=st.columns(6)
    items=[("Notes",d["notes"]),("Study Questions",d["chat_questions"]),("Quiz Questions",d["quiz_questions"]),("Correct",d["correct"]),("Interviews",d["interviews"])]
    for col,(label,val) in zip(cols,items):
        with col: st.markdown(f'<div class="metric"><small>{label}</small><b>{val}</b></div>',unsafe_allow_html=True)
    st.subheader("🏆 Achievements")
    if d["achievements"]:
        for a in d["achievements"]: st.success(a)
    else: st.info("Complete your first study, quiz, coding or interview activity to unlock achievements.")

# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown("# 📚 StudySathi AI")
    st.caption("Your B.Tech learning companion")
    pages=["🏠 Home","📚 My Notes","💬 AI Study Chat","📝 Exam Preparation","🎯 Quiz Zone","🎤 AI Interview","📊 My Progress"]
    choice=st.radio("Navigate",pages,index=pages.index(st.session_state.page if st.session_state.page in pages else "🏠 Home"),label_visibility="collapsed")
    st.session_state.page=choice
    st.divider()
    st.caption("Learn • Practise • Prepare • Grow")

# ----------------------------- ROUTER -----------------------------
if st.session_state.page=="🏠 Home": home()
elif st.session_state.page=="📚 My Notes": notes_page()
elif st.session_state.page=="💬 AI Study Chat": chat_page()
elif st.session_state.page=="📝 Exam Preparation": exam_page()
elif st.session_state.page=="🎯 Quiz Zone": quiz_page()
elif st.session_state.page=="🎤 AI Interview": interview_page()
elif st.session_state.page=="📊 My Progress": progress_page()

st.markdown('<div class="footer">StudySathi AI • Built for B.Tech students</div>',unsafe_allow_html=True)
