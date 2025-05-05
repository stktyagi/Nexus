import matplotlib.pyplot as plt
import numpy as np

# Model names and their Mean Cross-Validation accuracy scores from Network Analysis results
# Including both models for LR and RF as provided in the report
models = ['LR Model 1', 'LR Model 2', 'RF Model 1', 'RF Model 2']
accuracy_scores = [0.92, 0.93, 0.96, 0.98] # Using the Mean CV accuracy values from the report

# Create a bar chart
plt.figure(figsize=(10, 6)) # Set the figure size for better readability
bars = plt.bar(models, accuracy_scores, color=['skyblue', 'skyblue', 'lightgreen', 'lightgreen'])

# Add the accuracy values on top of the bars
for bar in bars:
    yval = bar.get_height()
    # Format the text to show 2 decimal places as in the report's mean CV scores for Network Analysis
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom')

# Set the title and labels
plt.title('Network Analysis Model Mean CV Accuracy Comparison')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.xlabel('Machine Learning Model Configuration')
plt.ylim(0.9, 1) # Set y-axis limit to focus on the range of scores

# Improve layout and display the plot
plt.tight_layout()

# To save the plot to a file instead of displaying it directly
# plt.savefig('network_analysis_accuracy_comparison.png')

# To display the plot (will not work directly in this environment, saving is preferred)
plt.show()

