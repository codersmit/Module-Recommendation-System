import matplotlib.pyplot as plt

# --- Constants ---
LIFELINE_SPACING = 2
ARROW_LENGTH = 1.5
DECISION_SIZE = 0.3

# --- Component Lifelines ---
plt.hlines(7, 0, 6, label='Sensor')
plt.hlines(5, 0, 6, label='AI Hub')
plt.hlines(3, 0, 6, label='Env. Modules')
plt.hlines(1, 0, 6, label='Cloud')

# --- Sensor Data and Initial Analysis---
plt.arrow(1, 7, ARROW_LENGTH, -1.5, head_width=0.2, head_length=0.3, label='Sensor Data')
plt.scatter(2.5, 5, s=DECISION_SIZE**2 * 100, color='orange') # Decision Point 1
plt.arrow(2.5, 4.5, ARROW_LENGTH, -1.5, head_width=0.2, head_length=0.3, label='Analyze + Response')

# --- Optional: Cloud Interaction ---
plt.arrow(4.5, 3, ARROW_LENGTH, 1, head_width=0.2, head_length=0.3, label='Store Data')

# --- Aesthetics ---
plt.xlim(-1, 8)
plt.ylim(0, 8)
plt.xlabel('Time / Sequence')
plt.legend(loc='upper left')
plt.title('PetHaven View 2: Simplified Sequence')
plt.show()
