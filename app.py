# app.py
import streamlit as st
import pickle
import os
from rag_engine import query_trusted_facts

# 1. Page Configuration
st.set_page_config(
    page_title="TrustGuard AI | News Verification Guardrail",
    page_icon="🛡️",
    layout="wide"
)

# 2. Advanced Styling (CSS)
st.markdown("""
    <style>
    /* Startup Page Styling */
    .startup-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        height: 60vh;
    }
    .startup-title {
        font-size: 4rem;
        font-weight: 900;
        color: #1E3A8A;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    .startup-subtitle {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    
    /* Main App Styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Management for Page Routing
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# --- SCREEN 1: STARTUP / SPLASH PAGE ---
if not st.session_state.initialized:
    st.markdown("""
        <div class="startup-container">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🛡️</div>
            <h1 class="startup-title">FAKE NEWS GUARDRAIL</h1>
            <p class="startup-subtitle">Hybrid Machine Learning & GenAI Verification Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Center the enter button horizontally
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        if st.button("🚀 Enter Dashboard", use_container_width=True, type="primary"):
            st.session_state.initialized = True
            st.rerun()

# --- SCREEN 2: MAIN APPLICATION ---
else:
    # Sidebar Setup
    with st.sidebar:
        st.markdown("## 🛡️ **System Status**")
        st.success("Core Engine Active")
        st.info("💡 Combining data-driven statistical patterns (Naive Bayes) with contextual knowledge retrieval (RAG) to build deterministic guardrails.")
        
        st.markdown("---")
        st.markdown("### **Preset Sample Claims**")
        if st.button("Sample 1: Verified Financial News"):
            st.session_state.input_text = "The president signed a new economic bill into law today after congressional approval focusing on infrastructure."
        if st.button("Sample 2: Medical Clickbait"):
            st.session_state.input_text = "Scientists discover that eating dark chocolate completely cures all chronic heart diseases overnight secret trick."
        if st.button("Sample 3: Wild Alien Hoax"):
            st.session_state.input_text = "Aliens landed in Washington DC yesterday and spoke directly to world leaders secret leaked video."
        
        st.markdown("---")
        if st.button("🔄 Reset to Startup Page"):
            st.session_state.initialized = False
            st.rerun()

    # Main Layout
    st.markdown('<h1 class="main-title">🛡️ TrustGuard AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; margin-bottom:2rem;'>Analyze and cross-reference statements against trusted linguistic baselines and live semantic databases.</p>", unsafe_allow_html=True)

    user_text = st.text_area(
        "Enter the news article headline, social media claim, or text statement to analyze:",
        value=st.session_state.input_text,
        height=130,
        placeholder="Type or paste a claim here..."
    )

    # Load Model trained using the Kaggle dataset
    @st.cache_resource
    def load_ml_model():
        with open('naive_bayes_model.pkl', 'rb') as f:
            return pickle.load(f)

    model_pipeline = load_ml_model()

    if st.button("🚀 Verify Authenticity", type="primary"):
        if not user_text.strip():
            st.error("❌ Please provide a valid text input claim first.")
        else:
            with st.status("Running defensive verification workflows...", expanded=True) as status:
                st.write("🧠 Querying Classic ML Linguistic Model...")
                probabilities = model_pipeline.predict_proba([user_text])[0]
                fake_probability_score = probabilities[1] * 100
                
                st.write("🔍 Extracting context signals from Verified Vector Store...")
                retrieved_context = query_trusted_facts(user_text, k=2)
                context_str = "\n".join([f"- {fact}" for fact in retrieved_context])
                
                st.write("🤖 Compiling Aggregated Trust Verdict via LLM Engine...")
                
                # Dynamic Environment Configuration for Dual Mode
                system_prompt = "You are an expert trust-guardrail AI analyst. Evaluate the claim against the facts."
                user_prompt = f"Claim: {user_text}\nML Score: {fake_probability_score:.2f}%\nContext:\n{context_str}"
                
                # Check for Groq API Key (looks inside Streamlit Cloud Secrets or local environment variables)
                groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
                
                if groq_api_key:
                    # MODE A: Cloud Execution via Groq
                    try:
                        from groq import Groq
                        client = Groq(api_key=groq_api_key)
                        completion = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.2
                        )
                        llm_reasoning = completion.choices[0].message.content
                    except Exception as cloud_err:
                        llm_reasoning = f"⚠️ Cloud API Error: {str(cloud_err)}"
                else:
                    # MODE B: Local Fallback Execution via Ollama
                    try:
                        import ollama
                        response = ollama.generate(model='llama3', system=system_prompt, prompt=user_prompt)
                        llm_reasoning = response['response']
                    except Exception:
                        # MODE C: Failsafe Static Report Guardrail if no LLM runs
                        llm_reasoning = f"### 📊 System Evaluation Summary\n\n" \
                                        f"• The input text displays a linguistic abnormality index of **{fake_probability_score:.2f}%**.\n" \
                                        f"• **Linguistic Classification**: {'🚨 UNRELIABLE PATTERNS DETECTED' if fake_probability_score > 50 else '✅ STANDARD JOURNALISTIC SYNTAX'}\n\n" \
                                        f"**Vector Store Verification References:**\n{context_str if context_str else 'No exact matching factual verification documents indexed.'}\n\n" \
                                        f"*Configuration Note: Live cloud features are currently running in deterministic static-guardrail mode.*"
                    
                status.update(label="Analysis Completed Successfully!", state="complete", expanded=False)

            # Results Display Layout
            st.markdown("---")
            st.markdown("### 📊 Analysis Reports")
            
            layout_col1, layout_col2 = st.columns([1, 2], gap="large")
            
            with layout_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    label="Linguistic Risk Score", 
                    value=f"{fake_probability_score:.1f}%",
                    delta=f"{fake_probability_score - 50:.1f}% vs threshold",
                    delta_color="inverse"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                if fake_probability_score > 50:
                    st.error("🚨 **High Risk Indicator:** The structural framing patterns heavily match standard historical disinformation layouts.")
                else:
                    st.success("✨ **Low Risk Indicator:** The structural syntax aligns cleanly with normal verified distributions.")
                    
                with st.expander("📍 View Retrieved Source Knowledge Base Context", expanded=True):
                    if retrieved_context:
                        for fact in retrieved_context:
                            st.markdown(f"• `{fact}`")
                    else:
                        st.caption("No explicit background reference records found matching this context window.")
                        
            with layout_col2:
                st.markdown("### 🤖 Hybrid Intelligence Reasoning Summary")
                st.info(llm_reasoning)