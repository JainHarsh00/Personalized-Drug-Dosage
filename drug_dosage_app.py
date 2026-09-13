import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import io
import sys

# Page configuration
st.set_page_config(
    page_title="MedDose AI - Personalized Drug Dosage System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 100%;
    }
    .metric-card {
        background: #8a54d2;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #ff3446;
        border-left: 4px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #0dcaf0;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2 {
        color: #34495e;
        font-weight: 600;
    }
    h3 {
        color: #546e7a;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Stochastic Patient Environment
class PatientEnvStochastic(gym.Env):
    def __init__(self, drug_config):
        super().__init__()
        self.drug_config = drug_config
        
        # Pharmacokinetics
        self.base_decay_rate = drug_config['decay_rate']
        self.decay_noise_level = 0.02
        
        # Therapeutic Window
        self.target_min = drug_config['therapeutic_min']
        self.target_max = drug_config['therapeutic_max']
        
        # Toxic levels
        self.toxic_level = drug_config['toxic_level']
        
        # Action and Observation Spaces
        self.action_space = spaces.Box(
            low=0.0, 
            high=drug_config['max_dose'], 
            shape=(1,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, 
            high=100.0, 
            shape=(5,),  # concentration, heart_rate, bp_sys, bp_dias, recovery_level
            dtype=np.float32
        )
        
        self.max_steps = 100
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_concentration = 0.0
        self.current_step = 0
        
        # Initialize stochastic patient parameters
        self.heart_rate = np.random.normal(75, 5)  # Normal: 60-100 bpm
        self.bp_systolic = np.random.normal(120, 10)  # Normal: 90-120 mmHg
        self.bp_diastolic = np.random.normal(80, 8)   # Normal: 60-80 mmHg
        self.recovery_level = 0.0  # 0 to 100%
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        return np.array([
            self.current_concentration,
            self.heart_rate,
            self.bp_systolic,
            self.bp_diastolic,
            self.recovery_level
        ]).astype(np.float32)
    
    def _update_vitals(self, concentration):
        """Update patient vitals based on drug concentration"""
        # Heart rate variation based on concentration
        if concentration > self.target_max:
            self.heart_rate += np.random.normal(2, 1)
        elif concentration < self.target_min:
            self.heart_rate += np.random.normal(-1, 0.5)
        else:
            self.heart_rate += np.random.normal(0, 0.5)
        
        # Blood pressure variation
        if concentration > self.toxic_level:
            self.bp_systolic += np.random.normal(5, 2)
            self.bp_diastolic += np.random.normal(3, 1)
        else:
            self.bp_systolic += np.random.normal(0, 1)
            self.bp_diastolic += np.random.normal(0, 0.8)
        
        # Clamp vitals to realistic ranges
        self.heart_rate = np.clip(self.heart_rate, 50, 150)
        self.bp_systolic = np.clip(self.bp_systolic, 80, 200)
        self.bp_diastolic = np.clip(self.bp_diastolic, 50, 120)
        
        # Recovery level increases when in therapeutic window
        if self.target_min <= concentration <= self.target_max:
            self.recovery_level = min(100, self.recovery_level + 2)
        else:
            self.recovery_level = max(0, self.recovery_level - 0.5)
    
    def step(self, action):
        dosage_administered = np.asarray(action).item()
        
        # Stochastic decay rate
        random_noise = np.random.normal(loc=0.0, scale=self.decay_noise_level)
        current_decay_rate = self.base_decay_rate + random_noise
        current_decay_rate = max(0.01, current_decay_rate)
        
        # Update concentration
        drug_cleared = self.current_concentration * current_decay_rate
        self.current_concentration += dosage_administered - drug_cleared
        self.current_concentration = max(0, self.current_concentration)
        
        # Update vitals
        self._update_vitals(self.current_concentration)
        
        # Calculate reward
        if self.target_min <= self.current_concentration <= self.target_max:
            reward = 1
        elif self.current_concentration > self.toxic_level:
            reward = -10  # Severe penalty for toxicity
        else:
            reward = -1
        
        self.current_step += 1
        
        terminated = self.current_concentration > self.toxic_level * 1.5  # Critical toxicity
        truncated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, truncated, {}

# Drug database
DRUG_DATABASE = {
    "Warfarin": {
        "description": "Anticoagulant for blood clot prevention",
        "disease": "Atrial Fibrillation / DVT",
        "therapeutic_min": 2.0,
        "therapeutic_max": 3.0,
        "toxic_level": 5.0,
        "max_dose": 15.0,
        "decay_rate": 0.15,
        "unit": "mg",
        "half_life": "40 hours",
        "onset": "24-72 hours",
        "monitoring": "INR levels",
        "category": "Anticoagulant"
    },
    "Digoxin": {
        "description": "Cardiac glycoside for heart failure",
        "disease": "Congestive Heart Failure",
        "therapeutic_min": 0.5,
        "therapeutic_max": 2.0,
        "toxic_level": 3.0,
        "max_dose": 5.0,
        "decay_rate": 0.08,
        "unit": "ng/mL",
        "half_life": "36-48 hours",
        "onset": "5-30 minutes (IV)",
        "monitoring": "Serum digoxin levels",
        "category": "Cardiac Glycoside"
    },
    "Phenytoin": {
        "description": "Anticonvulsant for seizure control",
        "disease": "Epilepsy",
        "therapeutic_min": 10.0,
        "therapeutic_max": 20.0,
        "toxic_level": 30.0,
        "max_dose": 40.0,
        "decay_rate": 0.12,
        "unit": "μg/mL",
        "half_life": "22 hours",
        "onset": "30-60 minutes",
        "monitoring": "Serum phenytoin levels",
        "category": "Anticonvulsant"
    },
    "Lithium": {
        "description": "Mood stabilizer for bipolar disorder",
        "disease": "Bipolar Disorder",
        "therapeutic_min": 0.6,
        "therapeutic_max": 1.2,
        "toxic_level": 2.0,
        "max_dose": 3.0,
        "decay_rate": 0.10,
        "unit": "mEq/L",
        "half_life": "18-24 hours",
        "onset": "1-2 weeks",
        "monitoring": "Serum lithium levels",
        "category": "Mood Stabilizer"
    },
    "Vancomycin": {
        "description": "Antibiotic for serious bacterial infections",
        "disease": "MRSA Infection",
        "therapeutic_min": 10.0,
        "therapeutic_max": 20.0,
        "toxic_level": 40.0,
        "max_dose": 50.0,
        "decay_rate": 0.18,
        "unit": "μg/mL",
        "half_life": "4-6 hours",
        "onset": "Immediate (IV)",
        "monitoring": "Trough levels",
        "category": "Antibiotic"
    },
    "Theophylline": {
        "description": "Bronchodilator for respiratory conditions",
        "disease": "Asthma / COPD",
        "therapeutic_min": 10.0,
        "therapeutic_max": 20.0,
        "toxic_level": 30.0,
        "max_dose": 40.0,
        "decay_rate": 0.14,
        "unit": "μg/mL",
        "half_life": "8-9 hours",
        "onset": "30 minutes",
        "monitoring": "Serum theophylline",
        "category": "Bronchodilator"
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'selected_drug' not in st.session_state:
        st.session_state.selected_drug = None
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False

def create_concentration_chart(data, drug_config):
    """Create interactive concentration time-series chart"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Drug Concentration Over Time', 'Patient Vitals'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    # Concentration plot
    fig.add_trace(
        go.Scatter(
            x=data['time'],
            y=data['concentration'],
            mode='lines',
            name='Drug Concentration',
            line=dict(color='#3498db', width=3),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.1)'
        ),
        row=1, col=1
    )
    
    # Therapeutic window
    fig.add_hrect(
        y0=drug_config['therapeutic_min'],
        y1=drug_config['therapeutic_max'],
        fillcolor="rgba(40, 167, 69, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Therapeutic Window",
        annotation_position="right",
        row=1, col=1
    )
    
    # Toxic level line
    fig.add_hline(
        y=drug_config['toxic_level'],
        line_dash="dash",
        line_color="red",
        annotation_text="Toxic Level",
        annotation_position="right",
        row=1, col=1
    )
    
    # Vitals plot - Heart Rate
    fig.add_trace(
        go.Scatter(
            x=data['time'],
            y=data['heart_rate'],
            mode='lines',
            name='Heart Rate (bpm)',
            line=dict(color='#e74c3c', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text=f"Concentration ({drug_config['unit']})", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_vitals_chart(data):
    """Create blood pressure chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['bp_systolic'],
        mode='lines',
        name='Systolic BP',
        line=dict(color='#9b59b6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['time'],
        y=data['bp_diastolic'],
        mode='lines',
        name='Diastolic BP',
        line=dict(color='#1abc9c', width=2)
    ))
    
    # Normal BP range
    fig.add_hrect(
        y0=90, y1=120,
        fillcolor="rgba(40, 167, 69, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Normal BP Range (Systolic)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Blood Pressure Monitoring",
        xaxis_title="Time (hours)",
        yaxis_title="Blood Pressure (mmHg)",
        height=350,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def pk_pd_controller(concentration, drug_config):
    """
    PK/PD-informed dosing controller.

    Strategy:
      - Loading phase: if below the therapeutic window, administer an
        aggressive loading dose to rapidly bring concentration into range.
      - Maintenance phase: once in range, administer exactly the amount
        cleared each step (steady-state maintenance = C_target * ke),
        plus a small proportional correction for drift.
      - Safety clamp: dose is capped at max_dose and zeroed if toxic.
    """
    target_min = drug_config['therapeutic_min']
    target_max = drug_config['therapeutic_max']
    target_mid = (target_min + target_max) / 2.0
    ke = drug_config['decay_rate']          # elimination rate constant
    max_dose = drug_config['max_dose']
    toxic_level = drug_config['toxic_level']

    # --- Safety first: stop dosing if we are already at/above toxic ---
    if concentration >= toxic_level * 0.9:
        return 0.0

    # --- Below therapeutic window: loading dose ---
    if concentration < target_min:
        deficit = target_mid - concentration
        # Loading dose = deficit + maintenance to compensate for this step's clearance
        loading_dose = deficit + target_mid * ke
        # Proportional boost scaled by how far we are below target
        proportional_gain = 1.5
        dose = min(loading_dose * proportional_gain, max_dose)

    # --- In therapeutic window: maintenance dose only ---
    elif target_min <= concentration <= target_max:
        # Maintenance = concentration * ke (replaces what's cleared)
        maintenance = concentration * ke
        # Small proportional correction to steer toward midpoint
        error = target_mid - concentration
        correction = 0.3 * ke * error
        dose = max(0.0, maintenance + correction)

    # --- Above therapeutic window but not toxic: withhold dose ---
    else:
        dose = 0.0

    return float(np.clip(dose, 0.0, max_dose))


def run_simulation(drug_name, num_steps=100):
    """Run the dosage simulation using a PK/PD-informed dosing controller."""
    drug_config = DRUG_DATABASE[drug_name]
    env = PatientEnvStochastic(drug_config)

    obs, _ = env.reset()

    data = {
        'time': [],
        'concentration': [],
        'dosage': [],
        'heart_rate': [],
        'bp_systolic': [],
        'bp_diastolic': [],
        'recovery_level': [],
        'in_therapeutic_window': []
    }

    for step in range(num_steps):
        current_concentration = obs[0]

        # Compute dose from controller
        dose = pk_pd_controller(current_concentration, drug_config)
        action = np.array([dose], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        data['time'].append(step)
        data['concentration'].append(obs[0])
        data['dosage'].append(action[0])
        data['heart_rate'].append(obs[1])
        data['bp_systolic'].append(obs[2])
        data['bp_diastolic'].append(obs[3])
        data['recovery_level'].append(obs[4])
        data['in_therapeutic_window'].append(
            1 if drug_config['therapeutic_min'] <= obs[0] <= drug_config['therapeutic_max'] else 0
        )

        if terminated or truncated:
            break

    return pd.DataFrame(data), env

def main():
    initialize_session_state()
    
    # Header
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.title("💊 MedDose AI")
        st.markdown("### Personalized Drug Dosage Management System")
        st.markdown("*Powered by Deep Reinforcement Learning*")
    
    # Disclaimer
    st.markdown("""
    <div class="danger-box">
        <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
        This is a research prototype and simulation tool. It is <strong>NOT intended for clinical use</strong> 
        and should <strong>NOT</strong> be used for actual medical decision-making. Always consult qualified 
        healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/cotton/128/000000/medical-heart.png", width=100)
        st.header("Drug Selection")
        
        selected_drug = st.selectbox(
            "Select Medication",
            options=list(DRUG_DATABASE.keys()),
            format_func=lambda x: f"{x} - {DRUG_DATABASE[x]['category']}"
        )
        
        st.session_state.selected_drug = selected_drug
        
        st.markdown("---")
        
        # Drug information
        drug_info = DRUG_DATABASE[selected_drug]
        st.markdown(f"**{selected_drug}**")
        st.caption(drug_info['description'])
        
        st.markdown("#### 📋 Drug Details")
        st.info(f"""
        **Indication:** {drug_info['disease']}  
        **Category:** {drug_info['category']}  
        **Half-life:** {drug_info['half_life']}  
        **Onset:** {drug_info['onset']}  
        **Monitoring:** {drug_info['monitoring']}
        """)
        
        st.markdown("#### 🎯 Therapeutic Range")
        st.success(f"""
        **Minimum:** {drug_info['therapeutic_min']} {drug_info['unit']}  
        **Maximum:** {drug_info['therapeutic_max']} {drug_info['unit']}  
        **Toxic Level:** {drug_info['toxic_level']} {drug_info['unit']}
        """)
        
        st.markdown("---")
        
        simulation_steps = st.slider(
            "Simulation Duration (hours)",
            min_value=24,
            max_value=168,
            value=72,
            step=12
        )
        
        if st.button("🚀 Start Simulation", use_container_width=True):
            with st.spinner("Running AI-powered dosage optimization..."):
                data, env = run_simulation(selected_drug, simulation_steps)
                st.session_state.simulation_data = data
                st.session_state.env = env
            st.success("✅ Simulation Complete!")
    
    # Main content area
    if st.session_state.simulation_data is not None:
        data = st.session_state.simulation_data
        drug_config = DRUG_DATABASE[selected_drug]
        
        # Key metrics
        st.markdown("## 📊 Treatment Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_conc = data['concentration'].iloc[-1]
            delta_conc = current_conc - data['concentration'].iloc[0]
            st.metric(
                "Current Concentration",
                f"{current_conc:.2f} {drug_config['unit']}",
                f"{delta_conc:+.2f} {drug_config['unit']}"
            )
        
        with col2:
            time_in_window = (data['in_therapeutic_window'].sum() / len(data)) * 100
            st.metric(
                "Time in Therapeutic Window",
                f"{time_in_window:.1f}%",
                "Target: >80%"
            )
        
        with col3:
            recovery = data['recovery_level'].iloc[-1]
            st.metric(
                "Recovery Level",
                f"{recovery:.1f}%",
                f"+{recovery:.1f}%"
            )
        
        with col4:
            avg_hr = data['heart_rate'].mean()
            st.metric(
                "Avg Heart Rate",
                f"{avg_hr:.0f} bpm",
                "Normal Range"
            )
        
        # Status indicator
        current_conc = data['concentration'].iloc[-1]
        if drug_config['therapeutic_min'] <= current_conc <= drug_config['therapeutic_max']:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>OPTIMAL</strong> - Drug concentration is within therapeutic window
            </div>
            """, unsafe_allow_html=True)
        elif current_conc > drug_config['toxic_level']:
            st.markdown("""
            <div class="danger-box">
                ⛔ <strong>TOXIC</strong> - Drug concentration exceeds safe levels
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>SUBTHERAPEUTIC</strong> - Drug concentration below effective range
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("## 📈 Visualization")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Concentration Profile",
            "❤️ Vital Signs",
            "💊 Dosage History",
            "📋 Data Table"
        ])
        
        with tab1:
            fig = create_concentration_chart(data, drug_config)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_bp = create_vitals_chart(data)
                st.plotly_chart(fig_bp, use_container_width=True)
            
            with col2:
                # Recovery level
                fig_recovery = go.Figure()
                fig_recovery.add_trace(go.Scatter(
                    x=data['time'],
                    y=data['recovery_level'],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#27ae60', width=3),
                    fillcolor='rgba(39, 174, 96, 0.2)'
                ))
                fig_recovery.update_layout(
                    title="Patient Recovery Progress",
                    xaxis_title="Time (hours)",
                    yaxis_title="Recovery Level (%)",
                    height=350,
                    template='plotly_white'
                )
                st.plotly_chart(fig_recovery, use_container_width=True)
        
        with tab3:
            fig_dosage = go.Figure()
            fig_dosage.add_trace(go.Bar(
                x=data['time'],
                y=data['dosage'],
                marker_color='#3498db',
                name='Dosage Administered'
            ))
            fig_dosage.update_layout(
                title=f"Dosage Administration History",
                xaxis_title="Time (hours)",
                yaxis_title=f"Dosage ({drug_config['unit']})",
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_dosage, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Dosage", f"{data['dosage'].mean():.2f} {drug_config['unit']}")
            with col2:
                st.metric("Maximum Dosage", f"{data['dosage'].max():.2f} {drug_config['unit']}")
            with col3:
                st.metric("Total Administered", f"{data['dosage'].sum():.2f} {drug_config['unit']}")
        
        with tab4:
            st.dataframe(
                data.style.format({
                    'concentration': '{:.2f}',
                    'dosage': '{:.2f}',
                    'heart_rate': '{:.1f}',
                    'bp_systolic': '{:.1f}',
                    'bp_diastolic': '{:.1f}',
                    'recovery_level': '{:.1f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{selected_drug}_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Clinical insights
        st.markdown("## 🔬 AI-Generated Clinical Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dosage Optimization")
            insights = []
            
            if time_in_window >= 80:
                insights.append("✅ Excellent maintenance of therapeutic levels")
            elif time_in_window >= 60:
                insights.append("⚠️ Moderate therapeutic coverage - consider adjustment")
            else:
                insights.append("❌ Poor therapeutic coverage - protocol review needed")
            
            avg_dosage = data['dosage'].mean()
            if avg_dosage < drug_config['max_dose'] * 0.3:
                insights.append("💡 Conservative dosing strategy employed")
            elif avg_dosage > drug_config['max_dose'] * 0.7:
                insights.append("⚠️ Aggressive dosing - monitor for toxicity")
            
            for insight in insights:
                st.info(insight)
        
        with col2:
            st.markdown("### Patient Response")
            responses = []
            
            hr_variability = data['heart_rate'].std()
            if hr_variability < 5:
                responses.append("❤️ Stable cardiovascular response")
            else:
                responses.append("⚠️ Increased heart rate variability detected")
            
            bp_sys_avg = data['bp_systolic'].mean()
            if 90 <= bp_sys_avg <= 120:
                responses.append("✅ Blood pressure within normal limits")
            else:
                responses.append("⚠️ Blood pressure outside normal range")
            
            if recovery > 70:
                responses.append("📈 Excellent therapeutic response")
            elif recovery > 40:
                responses.append("📊 Moderate therapeutic response")
            else:
                responses.append("📉 Limited therapeutic response")
            
            for response in responses:
                st.info(response)
    
    else:
        # Welcome screen
        st.markdown("## 👋 Welcome to MedDose AI")
        
        st.markdown("""
        MedDose AI is an advanced drug dosage optimization system powered by **Deep Reinforcement Learning**. 
        Our AI agent learns to maintain drug concentrations within the therapeutic window while accounting for 
        patient-specific physiological variations.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Precision Dosing</h3>
                <p>AI-optimized dosage recommendations based on real-time patient parameters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Live Monitoring</h3>
                <p>Continuous tracking of drug levels and patient vitals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Adaptive Learning</h3>
                <p>System adapts to stochastic physiological variations</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔬 How It Works")
        
        st.markdown("""
        1. **Select a medication** from the sidebar based on the patient's condition
        2. **Configure simulation parameters** including duration and monitoring frequency
        3. **Run the AI simulation** to see how the RL agent optimizes dosing
        4. **Analyze results** through interactive visualizations and clinical insights
        5. **Export data** for further analysis or documentation
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Supported Medications")
        
        cols = st.columns(3)
        for idx, (drug, info) in enumerate(DRUG_DATABASE.items()):
            with cols[idx % 3]:
                st.markdown(f"""
                **{drug}**  
                {info['category']}  
                *{info['disease']}*
                """)
        
        st.markdown("---")
        
        st.info("👈 **Get started** by selecting a medication from the sidebar and clicking 'Start Simulation'")

if __name__ == "__main__":
    main()
