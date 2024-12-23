import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
biomarker_values = []
true_labels = []
mean_value = np.mean(biomarker_values)
predicted_labels = [1 if value >mean_value else 0 for value in biomarker_values]
fpr, tpr, thresholds = roc_curve(true_labels, biomarker_values)
roc_auc = auc(fpr, tpr)
print(f'AUC: {roc_auc:.3f}')
plt.figure(figsize=(6, 6))
plt.step(fpr, tpr, where='post', color='darkblue', lw=2, label=f'AUC: {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Reference-560',fontweight='bold')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.text(0.5, 0.5, f'AUC: {roc_auc:.3f}', color='darkblue', fontsize=16, ha='center', va='center', alpha=0.8)
plt.grid(alpha=0.3)
plt.show()
