import matplotlib.pyplot as plt
import numpy as np

# Model names and their Mean Cross-Validation accuracy scores from Dynamic Analysis results
# Using Mean Cross-Validation Accuracy as per the report for a more robust comparison
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracy_scores = [0.6480, 0.7520, 0.7560] # Using the Mean CV accuracy values from the report

# Create a bar chart
plt.figure(figsize=(10, 6)) # Set the figure size for better readability
bars = plt.bar(models, accuracy_scores, color=['skyblue', 'lightgreen', 'salmon'])

# Add the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    # Format the text to show 4 decimal places as in the report's CV scores
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

# Set the title and labels
plt.title('Dynamic Analysis Model Mean CV Accuracy Comparison')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.xlabel('Machine Learning Model')
plt.ylim(0, 1) # Set y-axis limit from 0 to 1 for accuracy

# Improve layout and display the plot
plt.tight_layout()

# To save the plot to a file instead of displaying it directly
# plt.savefig('dynamic_analysis_accuracy_comparison.png')

# To display the plot (will not work directly in this environment, saving is preferred)
plt.show()

