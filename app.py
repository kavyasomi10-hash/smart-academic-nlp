# app.py - Streamlit UI for Smart Academic Text Processing
# Run with: streamlit run app.py
import streamlit as st
import sys, json, os, re
sys.path.append("modules")

st.set_page_config(
    page_title="Smart Academic Text Processor",
    page_icon="📚", layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:bold;color:#1F4E79;
    text-align:center;padding:1rem 0 0.2rem 0;}
.sub-header{font-size:0.9rem;color:#595959;text-align:center;
    padding-bottom:1rem;border-bottom:2px solid #C9A84C;
    margin-bottom:1.5rem;}
.section-title{font-size:0.95rem;font-weight:bold;
    color:#1F4E79;margin:0.8rem 0 0.4rem 0;}
.original-box{background:#F0F4F8;border-left:4px solid #1F4E79;
    border-radius:0 8px 8px 0;padding:1rem;font-size:0.9rem;
    line-height:1.7;min-height:160px;}
.simplified-box{background:#F0F8F4;border-left:4px solid #1E6B3C;
    border-radius:0 8px 8px 0;padding:1rem;font-size:0.9rem;
    line-height:1.7;min-height:160px;}
.summary-box{background:#FFF8F0;border-left:4px solid #C9A84C;
    border-radius:0 8px 8px 0;padding:1rem;font-size:0.9rem;
    line-height:1.7;min-height:160px;}
.metric-card{background:white;border:1px solid #E0E0E0;
    border-radius:8px;padding:1rem;text-align:center;
    box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:0.5rem;}
.metric-value{font-size:1.6rem;font-weight:bold;color:#1F4E79;}
.metric-label{font-size:0.78rem;color:#595959;margin-top:0.2rem;}
.metric-delta{font-size:0.82rem;color:#1E6B3C;font-weight:bold;
    margin-top:0.3rem;}
.word-badge{background:#E8F0F8;color:#1F4E79;padding:2px 8px;
    border-radius:12px;font-size:0.75rem;font-weight:bold;}
.sample-box{background:#FFFDE7;border:1px solid #C9A84C;
    border-radius:6px;padding:0.8rem;font-size:0.85rem;
    color:#444;margin-bottom:0.5rem;}
</style>""", unsafe_allow_html=True)


# ── Metrics (YOUR OWN CODE) ────────────────────────────────────
def count_syllables(word):
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3: return 1
    vowels = "aeiouy"
    count, prev = 0, False
    for c in word:
        v = c in vowels
        if v and not prev: count += 1
        prev = v
    if word.endswith('e'): count -= 1
    return max(1, count)

def flesch_kincaid_grade(text):
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    ns = max(1, len(sents))
    words = text.split()
    nw = max(1, len(words))
    nsyl = sum(count_syllables(w) for w in words)
    return round(0.39*(nw/ns) + 11.8*(nsyl/nw) - 15.59, 2)

def rouge1_score(ref, hyp):
    r = set(ref.lower().split())
    h = set(hyp.lower().split())
    if not r or not h: return 0.0
    ov = r & h
    p = len(ov)/len(h)
    rc = len(ov)/len(r)
    if p+rc == 0: return 0.0
    return round(2*p*rc/(p+rc), 4)

def compression_ratio(orig, comp):
    return round(
        (1 - len(comp.split())/max(1,len(orig.split())))*100, 1)

def readability_label(s):
    if s<=6:  return "Grade 6 — Easy"
    if s<=8:  return "Grade 8 — Moderate"
    if s<=10: return "Grade 10 — Difficult"
    return "Undergraduate — Very Difficult"

def readability_color(s):
    if s<=6:  return "#1E6B3C"
    if s<=8:  return "#2E75B6"
    if s<=10: return "#C9A84C"
    return "#B84A2E"


# ── Load Models ────────────────────────────────────────────────
@st.cache_resource
def load_simplifier():
    from simplification import TextSimplifier
    return TextSimplifier()

@st.cache_resource
def load_summarizer():
    from summarization import TextSummarizer
    return TextSummarizer()

@st.cache_resource
def load_preprocessor():
    from preprocessing import TextPreprocessor
    return TextPreprocessor()

@st.cache_data
def load_dataset():
    p = "data/ncert_dataset.json"
    if os.path.exists(p):
        with open(p,'r',encoding='utf-8') as f:
            return json.load(f)
    return None


# ── Sample Texts ───────────────────────────────────────────────
SAMPLES = {
    "NLP / Transformers (Complex)": (
        "Transformer-based language models have demonstrated "
        "remarkable capabilities across a wide range of natural "
        "language processing tasks. These models leverage "
        "self-attention mechanisms to capture long-range "
        "dependencies in text, enabling them to generate coherent "
        "and contextually appropriate responses. Pre-training on "
        "large-scale corpora followed by task-specific fine-tuning "
        "has emerged as the dominant paradigm for developing "
        "high-performance NLP systems. However, the computational "
        "requirements of these models present significant challenges "
        "for deployment on resource-constrained hardware. Knowledge "
        "distillation techniques have been proposed to compress "
        "large models into smaller, more efficient variants while "
        "preserving most of their performance characteristics. "
        "The development of lightweight transformer architectures "
        "represents a critical step towards democratizing access "
        "to advanced NLP capabilities for researchers and "
        "practitioners operating under computational constraints."
    ),
    "Microorganisms (NCERT Ch.2)": (
        "Microorganisms are living organisms that are too small "
        "to be seen with the naked eye. They can only be observed "
        "under a microscope. Microorganisms include bacteria, "
        "fungi, protozoa and some algae. They live in all kinds "
        "of environments including soil, water, air, and inside "
        "the bodies of animals and plants. Some microorganisms "
        "are beneficial to us while others are harmful. Beneficial "
        "microorganisms help in the preparation of curd, bread and "
        "cake. They are also used in the production of alcohol and "
        "medicines called antibiotics which kill disease-causing "
        "bacteria. Harmful microorganisms cause diseases in human "
        "beings, plants and animals. Diseases like cholera, typhoid "
        "and tuberculosis are caused by bacteria. Microorganisms "
        "also help in increasing soil fertility by fixing "
        "atmospheric nitrogen and decomposing dead organic matter "
        "into simpler substances that enrich the soil."
    ),
    "Force and Pressure (NCERT Ch.5)": (
        "A force is essentially a push or a pull. A bullock "
        "pulling a cart, a footballer kicking a ball, a carpenter "
        "using a saw on a plank of wood are some examples of "
        "forces being applied. Forces applied on an object in "
        "the same direction add to one another. If the two forces "
        "act in the opposite directions on an object, the net "
        "force acting on it is the difference between the two "
        "forces. The strength of a force is usually expressed by "
        "its magnitude. The pressure exerted by air around us is "
        "known as atmospheric pressure. The atmospheric pressure "
        "at sea level is approximately equal to the weight of a "
        "column of air of 10 metres high over each square "
        "centimetre of our body. Pressure is defined as force "
        "per unit area. Liquids also exert pressure on the walls "
        "of the container in which they are stored."
    ),
}


# ── Header ─────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">📚 Smart Academic Text Processor</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">NCERT Class 8 Science — '
    'Simplification &amp; Summarization using Lightweight '
    'Transformer Models &nbsp;|&nbsp; MCA Final Year Project'
    '</div>',
    unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Mode", [
        "📝 Paste Your Own Text",
        "🧪 Use Sample Text",
        "📚 Select NCERT Chapter"],
        label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📖 Reading Level")
    reading_level = st.selectbox("Level",
        ["Grade 6","Grade 8","Grade 10","Undergraduate"],
        index=1)

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("""
| Property | Value |
|----------|-------|
| Source | NCERT Cl.8 Sci |
| Chapters | 13 |
| Paragraphs | 235 |
| Words | 69,289 |
""")
    st.markdown("---")
    st.markdown("### 🤖 Models Used")
    st.markdown("""
**Simplification:**
T5-base (250M params)
Hybrid: Vocab + Neural

**Summarization:**
BART-large-CNN (400M)
CNN/DailyMail trained

**Hardware:**
CPU only | 8GB RAM
No GPU required
""")


# ── Input Section ──────────────────────────────────────────────
input_text = ""

if "Paste" in mode:
    st.markdown(
        '<div class="section-title">📝 Enter Text</div>',
        unsafe_allow_html=True)
    input_text = st.text_area("Text", height=160,
        placeholder="Paste any academic paragraph here "
                    "(100+ words recommended for best results)...",
        label_visibility="collapsed")
    if input_text.strip():
        wc = len(input_text.split())
        col1, col2 = st.columns([1,5])
        with col1:
            st.markdown(
                f'<span class="word-badge">{wc} words</span>',
                unsafe_allow_html=True)
        if wc < 30:
            st.warning("Please add more text (min 30 words)")
        elif wc < 80:
            st.info("Tip: 100+ words gives better summarization")

elif "Sample" in mode:
    st.markdown(
        '<div class="section-title">🧪 Choose Sample Text</div>',
        unsafe_allow_html=True)
    sample_name = st.selectbox("Sample",
        list(SAMPLES.keys()),
        label_visibility="collapsed")
    input_text = SAMPLES[sample_name]
    fk = round(0.39*(len(input_text.split())/
        max(1,len([s for s in re.split(r'[.!?]+',input_text)
                   if s.strip()]))) +
        11.8*(sum(max(1,len([c for c in w.lower()
                             if c in 'aeiouy']))
                  for w in input_text.split()) /
              max(1,len(input_text.split()))) - 15.59, 1)
    st.markdown(
        f'<div class="sample-box">'
        f'<b>FK Grade:</b> {fk} | '
        f'<b>Words:</b> {len(input_text.split())} | '
        f'Good for demonstrating simplification</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="original-box">'
        f'{input_text[:400]}{"..." if len(input_text)>400 else ""}'
        f'</div>',
        unsafe_allow_html=True)

else:
    ds = load_dataset()
    if ds:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="section-title">📚 Chapter</div>',
                unsafe_allow_html=True)
            cnames = [
                f"Ch{c['chapter_number']}: {c['chapter_title']}"
                for c in ds["chapters"]]
            sel_ch = st.selectbox("Ch", cnames,
                label_visibility="collapsed")
            chi = cnames.index(sel_ch)
            chdata = ds["chapters"][chi]
        with col2:
            st.markdown(
                '<div class="section-title">📄 Paragraph</div>',
                unsafe_allow_html=True)
            if chdata["paragraphs"]:
                popts = [
                    f"Para {i+1} ({len(p.split())} words)"
                    for i,p in enumerate(chdata["paragraphs"])]
                sel_p = st.selectbox("Para", popts,
                    label_visibility="collapsed")
                pi = popts.index(sel_p)
                input_text = chdata["paragraphs"][pi]
        if input_text:
            st.markdown(
                f'<div class="original-box">'
                f'{input_text[:500]}'
                f'{"..." if len(input_text)>500 else ""}'
                f'</div>',
                unsafe_allow_html=True)
            st.caption(f"{len(input_text.split())} words")
    else:
        st.error(
            "Dataset not found. "
            "Run modules/dataset_loader.py first.")


# ── Process ────────────────────────────────────────────────────
st.markdown("")
_, c2, _ = st.columns([1,2,1])
with c2:
    go = st.button("🚀  Process Text",
        use_container_width=True, type="primary")

if go:
    if not input_text or len(input_text.strip()) < 10:
        st.error("Please enter or select some text first.")
    elif len(input_text.split()) < 20:
        st.warning("Please provide at least 20 words.")
    else:
        with st.spinner("Loading models (first time: ~1 min)..."):
            try:
                pre  = load_preprocessor()
                simp = load_simplifier()
                summ = load_summarizer()
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

        prog = st.progress(0, text="Starting pipeline...")
        prog.progress(20, text="Step 1/4: Preprocessing...")
        try:
            clean = pre.preprocess(input_text)
        except Exception:
            clean = input_text

        prog.progress(45, text="Step 2/4: Simplifying with T5-base...")
        try:
            simplified = simp.simplify(clean, level=reading_level)
        except Exception as e:
            simplified = clean
            st.warning(f"Simplification issue: {e}")

        prog.progress(75,
            text="Step 3/4: Summarizing with BART-large-CNN...")
        try:
            summary = summ.summarize(clean)
        except Exception as e:
            summary = "Could not generate summary."
            st.warning(f"Summarization issue: {e}")

        prog.progress(95, text="Step 4/4: Calculating metrics...")
        orig_fk   = flesch_kincaid_grade(input_text)
        simp_fk   = flesch_kincaid_grade(simplified)
        fk_diff   = round(orig_fk - simp_fk, 2)
        simp_comp = compression_ratio(input_text, simplified)
        summ_comp = compression_ratio(input_text, summary)
        rouge     = rouge1_score(input_text, summary)

        prog.progress(100, text="Complete!")
        prog.empty()
        st.success("✅ Processing complete!")
        st.markdown("---")

        # Results
        st.markdown("## 📊 Results")
        c1, c2, c3 = st.columns(3)

        with c1:
            wc = len(input_text.split())
            st.markdown(
                f'<div class="section-title">📄 Original Text '
                f'<span class="word-badge">{wc} words</span>'
                f'</div>', unsafe_allow_html=True)
            txt = input_text[:600]
            if len(input_text) > 600: txt += "..."
            st.markdown(
                f'<div class="original-box">{txt}</div>',
                unsafe_allow_html=True)

        with c2:
            wc = len(simplified.split())
            st.markdown(
                f'<div class="section-title">✏️ Simplified '
                f'<span class="word-badge">{wc} words</span>'
                f'</div>', unsafe_allow_html=True)
            txt = simplified[:600]
            if len(simplified) > 600: txt += "..."
            st.markdown(
                f'<div class="simplified-box">{txt}</div>',
                unsafe_allow_html=True)

        with c3:
            wc = len(summary.split())
            st.markdown(
                f'<div class="section-title">📝 Summary '
                f'<span class="word-badge">{wc} words</span>'
                f'</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="summary-box">{summary}</div>',
                unsafe_allow_html=True)

        # Metrics
        st.markdown("---")
        st.markdown("## 📈 Evaluation Metrics")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            col = readability_color(simp_fk)
            arrow = "↓" if fk_diff > 0 else "↑"
            st.markdown(f"""
<div class="metric-card">
<div class="metric-value" style="color:{col}">{simp_fk}</div>
<div class="metric-label">FK Grade After</div>
<div class="metric-delta">{arrow} {abs(fk_diff)} grades easier</div>
<div style="font-size:0.72rem;color:#888;margin-top:0.3rem">
Before: {orig_fk} ({readability_label(orig_fk)})</div>
</div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-value" style="color:#1E6B3C">{summ_comp}%</div>
<div class="metric-label">Compression</div>
<div class="metric-delta">shorter than original</div>
<div style="font-size:0.72rem;color:#888;margin-top:0.3rem">
{len(input_text.split())} → {len(summary.split())} words</div>
</div>""", unsafe_allow_html=True)

        with m3:
            rouge_color = "#1E6B3C" if rouge >= 0.65 else "#C9A84C"
            st.markdown(f"""
<div class="metric-card">
<div class="metric-value" style="color:{rouge_color}">{rouge}</div>
<div class="metric-label">ROUGE-1 Score</div>
<div class="metric-delta">{round(rouge*100,1)}% info retained</div>
<div style="font-size:0.72rem;color:#888;margin-top:0.3rem">
0.65+ = good retention</div>
</div>""", unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
<div class="metric-card">
<div class="metric-value" style="color:#1F4E79">
{len(input_text.split())}</div>
<div class="metric-label">Original Words</div>
<div class="metric-delta">→ {len(simplified.split())} simplified</div>
<div style="font-size:0.72rem;color:#888;margin-top:0.3rem">
→ {len(summary.split())} in summary</div>
</div>""", unsafe_allow_html=True)

        # Download
        st.markdown("---")
        dl = "\n".join([
            "SMART ACADEMIC TEXT PROCESSOR - RESULTS",
            "="*55, "",
            "ORIGINAL TEXT:", input_text, "",
            "SIMPLIFIED TEXT:", simplified, "",
            "SUMMARY:", summary, "",
            "EVALUATION METRICS:",
            f"  FK Grade Before         : {orig_fk} "
            f"({readability_label(orig_fk)})",
            f"  FK Grade After          : {simp_fk} "
            f"({readability_label(simp_fk)})",
            f"  FK Grade Improvement    : {fk_diff} grades easier",
            f"  Simplification Change   : {simp_comp}%",
            f"  Summarization Compression: {summ_comp}%",
            f"  ROUGE-1 Score           : {rouge}",
            f"  Original Words          : {len(input_text.split())}",
            f"  Simplified Words        : {len(simplified.split())}",
            f"  Summary Words           : {len(summary.split())}",
        ])
        _, c2, _ = st.columns([1,2,1])
        with c2:
            st.download_button(
                "⬇️  Download Results as Text File",
                data=dl,
                file_name="smart_nlp_results.txt",
                mime="text/plain",
                use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>"
    "Smart Academic Text Processing &nbsp;|&nbsp; "
    "MCA Final Year Project &nbsp;|&nbsp; "
    "T5-base + BART-large-CNN &nbsp;|&nbsp; "
    "NCERT Class 8 Science &nbsp;|&nbsp; "
    "CPU Only · 8GB RAM"
    "</small></center>",
    unsafe_allow_html=True)