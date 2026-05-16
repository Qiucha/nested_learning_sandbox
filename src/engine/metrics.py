import numpy as np

class CLEvaluator:
    def __init__(self, num_tasks=5):
        self.num_tasks = num_tasks
        # Initialize the T x T Accuracy Matrix (R)
        self.R = np.zeros((num_tasks, num_tasks))
        
    def update_matrix(self, train_task_id, eval_task_id, accuracy):
        """Records accuracy of model trained up to 'train_task_id' evaluated on 'eval_task_id'."""
        self.R[train_task_id, eval_task_id] = accuracy
        
    def compute_metrics(self):
        """Calculates final Average ACC, BWT, and Average Forgetting."""
        T = self.num_tasks
        
        # Average Accuracy (Class-IL standard)
        acc = np.mean(self.R[T-1, :])
        
        bwt = 0.0
        forgetting = 0.0
        
        if T > 1:
            # Backward Transfer (BWT)
            bwt_sum = sum([self.R[T-1, i] - self.R[i, i] for i in range(T-1)])
            bwt = bwt_sum / (T - 1)
            
            # Average Forgetting (F)
            f_j = [np.max(self.R[:T-1, j]) - self.R[T-1, j] for j in range(T-1)]
            forgetting = np.mean(f_j)
            
        return {
            "Accuracy_Matrix": self.R,
            "Average_ACC": acc,
            "BWT": bwt,
            "Forgetting": forgetting
        }