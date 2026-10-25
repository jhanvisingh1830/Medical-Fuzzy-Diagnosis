
import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Fuzzy Medical Case Study",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Diagnosis Using Fuzzy Logic")
st.write(
    "Educational demonstration of a Mamdani fuzzy inference system."
)

st.warning(
    "This is not a real medical diagnostic tool. "
    "The score is illustrative and must not be used for medical decisions."
)

# Define input and output variables
fever = ctrl.Antecedent(np.arange(35, 41, 0.1), 'fever')
cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# Fever membership functions
fever['normal'] = fuzz.trimf(fever.universe, [35, 36.5, 37.5])
fever['mild'] = fuzz.trimf(fever.universe, [37, 37.8, 38.5])
fever['high'] = fuzz.trimf(fever.universe, [38, 39, 40.5])

# Cough membership functions
cough['low'] = fuzz.trimf(cough.universe, [0, 0, 4])
cough['medium'] = fuzz.trimf(cough.universe, [2, 5, 8])
cough['high'] = fuzz.trimf(cough.universe, [6, 10, 10])

# Risk membership functions
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

# Define fuzzy rules
# Define a complete fuzzy rule base

rule1 = ctrl.Rule(
    fever['normal'] & cough['low'],
    risk['low']
)

rule2 = ctrl.Rule(
    fever['normal'] & cough['medium'],
    risk['low']
)

rule3 = ctrl.Rule(
    fever['normal'] & cough['high'],
    risk['medium']
)

rule4 = ctrl.Rule(
    fever['mild'] & cough['low'],
    risk['low']
)

rule5 = ctrl.Rule(
    fever['mild'] & cough['medium'],
    risk['medium']
)

rule6 = ctrl.Rule(
    fever['mild'] & cough['high'],
    risk['high']
)

rule7 = ctrl.Rule(
    fever['high'] & cough['low'],
    risk['medium']
)

rule8 = ctrl.Rule(
    fever['high'] & cough['medium'],
    risk['high']
)

rule9 = ctrl.Rule(
    fever['high'] & cough['high'],
    risk['high']
)

# Create fuzzy control system
diagnosis_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])

# User inputs
st.subheader("Enter Symptoms")

temperature = st.number_input(
    "Body temperature (°C)",
    min_value=35.0,
    max_value=40.9,
    value=39.0,
    step=0.1
)

cough_severity = st.slider(
    "Cough severity",
    min_value=0,
    max_value=10,
    value=9
)

# Calculate result
if st.button("Calculate Illustrative Risk Score"):

    diagnosis = ctrl.ControlSystemSimulation(diagnosis_ctrl)

    diagnosis.input['fever'] = temperature
    diagnosis.input['cough'] = cough_severity

    try:
        diagnosis.compute()

        result = diagnosis.output['risk']

        st.subheader("Result")

        st.metric(
            label="Illustrative Risk Score",
            value=f"{result:.2f} / 100"
        )

        st.write(
            "This result is generated from the fuzzy rules "
            "and membership functions defined in the project."
        )

        # Display fever membership graph
        st.subheader("Fever Membership Functions")

        fig1, ax1 = plt.subplots()
        for label in fever.terms:
            ax1.plot(
                fever.universe,
                fever[label].mf,
                label=label
            )

        ax1.set_xlabel("Temperature (°C)")
        ax1.set_ylabel("Membership Degree")
        ax1.set_ylim(0, 1.05)
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        # Display cough membership graph
        st.subheader("Cough Membership Functions")

        fig2, ax2 = plt.subplots()
        for label in cough.terms:
            ax2.plot(
                cough.universe,
                cough[label].mf,
                label=label
            )

        ax2.set_xlabel("Cough Severity")
        ax2.set_ylabel("Membership Degree")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

        # Display risk membership graph
        st.subheader("Risk Membership Functions")

        fig3, ax3 = plt.subplots()
        for label in risk.terms:
            ax3.plot(
                risk.universe,
                risk[label].mf,
                label=label
            )

        ax3.set_xlabel("Illustrative Risk Score")
        ax3.set_ylabel("Membership Degree")
        ax3.set_ylim(0, 1.05)
        ax3.legend()
        st.pyplot(fig3)
        plt.close(fig3)

    except Exception as e:
        st.error(
            "No fuzzy rule was activated for these inputs. "
            "The rule base needs to be expanded."
        )
        st.code(str(e))
