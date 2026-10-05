"""
Automated Optimization Strategy Selection System
Streamlit Dashboard for Faculty Demonstration.

This UI is connected to the actual compiler optimization backend.
All results are computed from real TAC, not fabricated.
"""

import streamlit as st
from copy import deepcopy

# Project backend imports - already working
from backend.tac.parser import parse_tac, constant_folding_example, dead_code_example, cse_example, algebraic_simplification_example, mixed_example
from backend.tac.instruction import Instruction
from backend.features.extractor import extract_features
from backend.scoring.rule_based import rule_based_strategy_scores, select_strategy, strategy_reason
from backend.optimizer.engine import run_iterative_optimization

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Automated Optimization Strategy Selection System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium CSS – dark compiler dashboard
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---- Dark base theme ---- */
html, html * { color: #f0f4f8; background:#0b0e17; }
#MainMenu {visibility: hidden;} /* hide streamlit menu */
footer {visibility: hidden;}

/* ---- Glassmorphism ---- */
.glass { 
    background: rgba(255,255,255,0.05); 
    border: 1px solid rgba(255,255,255,0.12); 
    backdrop-filter: blur(8px); 
    -webkit-backdrop-filter: blur(8px); 
    border-radius: 10px; 
    padding: 0.75rem 1rem; 
}

/* ---- Aurora/small radial glow ---- */
.glow-aurora { position: relative; }
.glow-aurora::after {
    content: ""; position: absolute; inset: 0; border-radius: 50%;
    background: radial-gradient(ellipse at 30% 20%, rgba(59,130,246,0.18), transparent 60%);
    opacity: 0; transition: opacity 0.4s ease; }
.glow-aurora:hover::after { opacity: 0.4; }

/* ---- Neo-brutalist primary button ---- */
.btn-neo {
    background: linear-gradient(135deg, #ff4b4b, #c44569);
    color: white; border: 1px solid #ff4b4b;
    padding: 0.6rem 1rem; border-radius: 8px;
    font-weight: 600; font-size: 0.95rem; letter-spacing: 0.02em;
    transition: all 0.2s ease; text-shadow: none; }
.btn-neo:hover {
    background: linear-gradient(135deg, #e64343, #a63a5d);
    border-color: #e64343;
    box-shadow: 0 2px 6px rgba(255,75,75,0.25); }

/* ---- Metric card ---- */
.metric-card { background: var(--glass-bg, rgba(255,255,255,0.05)); 
    border: 1px solid var(--glass-border, rgba(255,255,255,0.12)); 
    border-radius: 10px; padding: 1rem 1.2rem; }

/* ---- Progress bar custom ---- */
.stProgress > div > div > div { background: linear-gradient(90deg, #ff4b4b, #c44569); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state init
# ---------------------------------------------------------------------------
ss_keys = [
    "original_tac", "current_tac", "features", "scores", "selected_strategy",
    "iteration_history", "run_simulation", "example_type", "tac_input"
]
for k in ss_keys:
    if k not in st.session_state:
        st.session_state[k] = None


# ---------------------------------------------------------------------------
# Helper: parse TAC
# ---------------------------------------------------------------------------
def _parse_tac(source: str):
    """Parse TAC source string -> TAC object using the real parser."""
    return parse_tac(source)


# ---------------------------------------------------------------------------
# 1. HERO / WELCOME
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding:1rem; background:var(--glass-bg, rgba(255,255,255,0.05)); border:1px solid var(--glass-border, rgba(255,255,255,0.12)); border-radius:8px; margin-bottom:1rem;">
<div style="font-size:1.2rem; font-weight:600; color:#ff4b4b;">COMPILER INTELLIGENCE SYSTEM</div>
<div style="font-size:0.95rem; color:#8892b0; margin-top:0.25rem;">Automated Optimization Strategy Selection System</div>
<div style="font-size:0.85rem; color:#666; margin-top:0.25rem; margin-bottom:0.25rem;">"Analyze TAC → identify opportunities → select the best optimization → measure improvement → repeat until convergence."</div>
<div style="font-size:1rem; color:#28a745;">● SYSTEM READY</div>
</div>

<h2>Start a Simulation</h2>
<p style="font-size:0.9rem; color:#666;">Choose a demonstration program or provide your own TAC.</p>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. START HERE / EXAMPLE BUTTONS
# ---------------------------------------------------------------------------
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("Constant Folding", use_container_width=True):
        st.session_state.example_type = "constant_folding"; st.rerun()

with example_col2:
    if st.button("CSE", use_container_width=True):
        st.session_state.example_type = "cse"; st.rerun()

with example_col3:
    if st.button("Dead Code", use_container_width=True):
        st.session_state.example_type = "dead_code"; st.rerun()

st.markdown("")  # spacer
with example_col1:
    if st.button("Algebraic", use_container_width=True):
        st.session_state.example_type = "algebraic"; st.rerun()

with example_col2:
    if st.button("Mixed Example", use_container_width=True):
        st.session_state.example_type = "mixed"; st.rerun()

st.markdown("</div></div>", unsafe_allow_html=True)


# 3. BENTO GRID – ROW 2: TAC INPUT + HOW IT WORKS
# ---------------------------------------------------------------------------
st.markdown("""<div class="glass-card-large" style="grid-column: span 2;;">
<h3 style="margin-top:0; margin-bottom:0.5rem;">TAC INPUT</h3>
<span style="font-size:0.75rem; color:#666; vertical-align:middle;">THREE ADDRESS CODE</span>
<br/>
<br/>
<div style="background:var(--glass-bg, rgba(30,35,45,0.8)); border:1px solid var(--glass-border, rgba(255,255,255,0.1)); border-radius:8px; padding:0.75rem; min-height:140px; max-height:180px; overflow:auto; font-family: ui-monospace, Menlo, Monaco, 'Cascadia Mono', sans-serif; font-size:0.85rem; color:#e8eaed;">
<div id="tac-input-area" style="white-space:pre-wrap; line-height:1.4;"></div>
</div>
<script>document.getElementById('tac-input-area').textContent=`t1 = 5 + 3
t2 = a + b
t3 = a + b
t4 = t1 * 1
t5 = 100
x = t4`</script>
<div style="margin-top:0.5rem;">
<form style="display:flex;gap:0.5rem;">
  <button type="button" class="btn-neo" style="align-self:flex-end;" onclick="st.session_state.update({'example_type':'constant_folding'})">Constant Folding</button>
  <button type="button" class="btn-neo" style="align-self:flex-end;" onclick="st.session_state.update({'example_type':'cse'})">CSE</button>
  <button type="button" class="btn-neo" style="align-self:flex-end;" onclick="st.session_state.update({'example_type':'dead_code'})">Dead Code</button>
  <button type="button" class="btn-neo" style="align-self:flex-end;" onclick="st.session_state.update({'example_type':'algebraic'})">Algebraic</button>
  <button type="button" class="btn-neo" style="align-self:flex-end;" onclick="st.session_state.update({'example_type':'mixed'})">Mixed</button>
  <button type="button" style="align-self:flex-end; background:#666; padding:0.6rem 1rem;">Reset</button>
</form>
</div>
</div>""" + "</div>", unsafe_allow_html=True)

st.markdown("""<div class="glass-card-large" style="grid-column: span 2; padding-top:1rem;">
<h3 style="margin-top:0; margin-bottom:0.5rem;">HOW IT WORKS</h3>
<ol style="font-size:0.9rem; color:#666; margin-bottom:0;">
<li>TAC analyzed → features extracted</li>
<li>6 strategies scored</li>
<li>Best strategy selected</li>
<li>Optimization applied</li>
<li>Instruction count measured</li>
<li>Re-analyzed → repeat until convergence</li>
</ol>
</div>""" + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. PROCESS ANALYZE (only when state changes)
# ---------------------------------------------------------------------------
if st.session_state.original_tac is not None:
    # Feature Analysis
    features = st.session_state.features
    st.markdown("### Feature Analysis")
    cols = st.columns(8)
    card_data = [
        ("Statements", features["num_statements"], "📝"),
        ("Constants", features["num_constants"], "🔢"),
        ("Repeated Expr", features["num_repeated_expr"], "🔁"),
        ("Dead Stmts", features["num_dead_stmts"], "💀"),
        ("Algebraic", features["num_algebraic_ops"], "📐"),
        ("Loop Ops", features["num_loop_ops"], "🔄"),
        ("Tmp Vars", features["num_tmp_vars"], "🌐"),
        ("Const Ratio", f"{features['const_ratio']*100:.1f}%", "📊"),
    ]
    for i, (label, value, emoji) in enumerate(card_data):
        with cols[i]:
            st.metric(label=label, value=value)

    # Strategy Scores
    st.markdown("### Strategy Scores")
    strategy_names = {
        "constant_folding": "Constant Folding",
        "constant_propagation": "Constant Propagation",
        "common_subexpression": "Common Subexpression Elimination",
        "dead_code_elimination": "Dead Code Elimination",
        "algebraic_simplification": "Algebraic Simplification",
        "strength_reduction": "Strength Reduction",
    }
    scores = st.session_state.scores
    selected = st.session_state.selected_strategy
    
    # 2x3 Bento grid for strategies
    strategy_cfg = [
        ("constant_folding", "Constant Folding", "🔢"),
        ("constant_propagation", "Constant Propagation", "🔢"),
        ("common_subexpression", "Common Subexpression", "🔁"),
        ("dead_code_elimination", "Dead Code", "💀"),
        ("algebraic_simplification", "Algebraic", "📐"),
        ("strength_reduction", "Strength Red.", "🔄"),
    ]
    
    for r in range(2):
        st.markdown("<div style='display:grid grid-template-columns:repeat(3, 1fr); gap:0.5rem;'>", unsafe_allow_html=True)
        for c in range(3):
            idx = r * 3 + c
            if idx >= 6:
                break
            key, name, icon = strategy_cfg[idx]
            score = scores.get(key, 0) * 100
            is_selected = (key == selected)
            
            sel_badge = ''
            if is_selected:
                sel_badge = '<span style="color:#ff4b4b; font-weight:bold;">SELECTED</span>'
            
            st.markdown(f"""
            <div style="background:#f8f9fa; border:1px solid #e3e6e8; border-radius:8px; padding:0.5rem 0.25rem; text-align:center;">
            <div style="font-size:0.85rem; margin-bottom:0.1rem;">{icon} {name}</div>
            <div style="font-size:1.1rem; color:#ff4b4b; margin-bottom:0.1rem;">{score:.0f}%</div>
            {sel_badge}
            </div>
            """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Selected strategy highlight
    if selected:
        st.markdown(f"""
        <div style="background:#f8f9fa; border:2px solid #ff4b4b; border-radius:8px; padding:0.5rem; margin-top:0.5rem;">
        <div style="font-weight:bold; color:#ff4b4b;">SELECTED: {selected}</div>
        <div style="font-size:0.85rem; color:#666;">Rule-Based Score: {scores.get(selected, 0)*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Decision panel
    reason = strategy_reason(st.session_state.original_tac)
    st.markdown(f"""
    **CURRENT DECISION**
    Selected Strategy: **{selected}**
    Rule-Based Score: **{scores.get(selected, 0)*100:.0f}%**
    Reason: {reason}
    """)

    # Before/After
    before_count = len(st.session_state.original_tac)
    opt_fn_map = {
        "constant_folding": ("constant_folding", __import__('backend.optimizations.constant_folding', fromlist=['constant_folding']).constant_folding),
        "constant_propagation": ("constant_propagation", __import__('backend.optimizations.constant_propagation', fromlist=['constant_propagation']).constant_propagation),
        "common_subexpression": ("common_subexpression", __import__('backend.optimizations.common_subexpression', fromlist=['common_subexpression_elimination']).common_subexpression_elimination),
        "dead_code_elimination": ("dead_code_elimination", __import__('backend.optimizations.dead_code', fromlist=['dead_code_elimination']).dead_code_elimination),
        "algebraic_simplification": ("algebraic_simplification", __import__('backend.optimizations.algebraic_simplification', fromlist=['algebraic_simplification']).algebraic_simplification),
        "strength_reduction": ("strength_reduction", __import__('backend.optimizations.strength_reduction', fromlist=['strength_reduction']).strength_reduction),
    }
    
    opt_key, opt_fn = opt_fn_map.get(selected, (None, None))
    if opt_fn:
        after_tac = opt_fn(st.session_state.original_tac)
        after_count = len(after_tac)
    else:
        after_count = before_count
    
    reduction = before_count - after_count
    improvement_pct = (reduction / before_count * 100) if before_count > 0 else 0
    
    st.markdown(f"""
    **Instruction Count**
    Before: **{before_count}** instructions
    After: **{after_count}** instructions
    Reduction: **{reduction}** instructions
    Improvement: **{improvement_pct:.1f}%**
    """)

    # Run Optimization
    if st.button("Run Optimization Simulation", type="primary", use_container_width=True):
        result = run_iterative_optimization(st.session_state.original_tac, max_iterations=20)
        st.session_state.iteration_history = result["history"]
        st.session_state.current_tac = result["final_tac"]
        st.session_state.run_simulation = False
        st.rerun()

    # Iteration history
    if st.session_state.iteration_history:
        st.markdown("### Optimization Simulation")
        for iter_data in st.session_state.iteration_history:
            st.markdown(f"""
            <div style="background:#f8f9fa; border:1px solid #e3e6e8; border-radius:8px; padding:0.5rem; margin-bottom:0.5rem;">
            <div style="font-weight:bold; color:#ff4b4b;">Iteration {iter_data['iteration']}</div>
            <div style="font-size:0.85rem; color:#8892b0;">{iter_data['selected_strategy']}</div>
            <div style="font-size:0.85rem; color:#ff4b4b;">{iter_data['before_count']} → {iter_data['after_count']} instructions</div>
            <div style="font-size:0.75rem; color:#666;">Reduction: {iter_data['improvement']} ({iter_data['improvement_percentage']:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Final summary
        last = st.session_state.iteration_history[-1]
        total_red = len(st.session_state.original_tac) - last.get("after_count", len(st.session_state.original_tac))
        total_iters = len(st.session_state.iteration_history)
        pct = (total_red / len(st.session_state.original_tac) * 100) if len(st.session_state.original_tac) > 0 else 0
        
        st.markdown(f"""
        **Final Results**
        Original: {len(st.session_state.original_tac)} instructions
        Final: {last.get('after_count', len(st.session_state.original_tac))} instructions
        Reduction: {total_red} ({pct:.1f}%)
        Iterations: {total_iters}
        Status: {'CONVERGED' if last['improvement'] == 0 else 'STOPPED'}
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pass