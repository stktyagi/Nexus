import matplotlib.pyplot as plt
import numpy as np

# Model names and their accuracy scores from Static Analysis results
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracy_scores = [0.6925, 0.7491, 0.8692] # Using the accuracy values from the report

# Create a bar chart
plt.figure(figsize=(10, 6)) # Set the figure size for better readability
bars = plt.bar(models, accuracy_scores, color=['skyblue', 'lightgreen', 'salmon'])

# Add the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

# Set the title and labels
plt.title('Static Analysis Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Machine Learning Model')
plt.ylim(0, 1) # Set y-axis limit from 0 to 1 for accuracy

# Improve layout and display the plot
plt.tight_layout()

# To save the plot to a file instead of displaying it directly
# plt.savefig('static_analysis_accuracy_comparison.png')

# To display the plot (will not work directly in this environment, saving is preferred)
plt.show()
