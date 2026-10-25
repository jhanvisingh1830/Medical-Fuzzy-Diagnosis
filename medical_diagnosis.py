
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step 1: Define input and output variables
fever = ctrl.Antecedent(np.arange(35, 41, 0.1), 'fever')
cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# Step 2: Define membership functions
fever['normal'] = fuzz.trimf(fever.universe, [35, 36.5, 37.5])
fever['mild'] = fuzz.trimf(fever.universe, [37, 37.8, 38.5])
fever['high'] = fuzz.trimf(fever.universe, [38, 39, 40.5])

cough['low'] = fuzz.trimf(cough.universe, [0, 0, 4])
cough['medium'] = fuzz.trimf(cough.universe, [2, 5, 8])
cough['high'] = fuzz.trimf(cough.universe, [6, 10, 10])

risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

# Step 3: Define fuzzy rules
rule1 = ctrl.Rule(fever['normal'] & cough['low'], risk['low'])
rule2 = ctrl.Rule(fever['mild'] & cough['medium'], risk['medium'])
rule3 = ctrl.Rule(fever['high'] & cough['high'], risk['high'])
rule4 = ctrl.Rule(fever['high'] & cough['low'], risk['medium'])
rule5 = ctrl.Rule(fever['normal'] & cough['high'], risk['medium'])

# Step 4: Create control system
diagnosis_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5
])

diagnosis = ctrl.ControlSystemSimulation(diagnosis_ctrl)

# Step 5: Take user input
temperature = float(input("Enter body temperature in Celsius: "))
cough_severity = float(input("Enter cough severity (0-10): "))

diagnosis.input['fever'] = temperature
diagnosis.input['cough'] = cough_severity

# Step 6: Compute result
diagnosis.compute()

result = diagnosis.output['risk']

print("Illustrative Risk Score:", round(result, 2))
import matplotlib.pyplot as plt

# Display fever membership functions
fever.view()
plt.title("Fever Membership Functions")
plt.show()

# Display cough membership functions
cough.view()
plt.title("Cough Severity Membership Functions")
plt.show()

# Display risk membership functions
risk.view()
plt.title("Illustrative Risk Membership Functions")
plt.show()