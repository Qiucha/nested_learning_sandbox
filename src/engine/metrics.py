import numpy as np
import os

class CLEvaluator:
    # NEW: Added total_epochs parameter (default 25 = 5 tasks * 5 epochs)
    def __init__(self, num_tasks=5, total_epochs=25):
        self.num_tasks = num_tasks
        self.total_epochs = total_epochs
        
        # Initialize the T x T Accuracy Matrix (R) for standard task boundaries
        self.R = np.zeros((num_tasks, num_tasks))
        
        # Initialize the E x T History Matrix for high-resolution tracking
        # We fill it with NaN (Not a Number) so un-trained tasks don't plot as 0s
        self.history = np.full((total_epochs, num_tasks), np.nan)
        
    def update_matrix(self, train_task_id, eval_task_id, accuracy):
        """Records accuracy at strict task boundaries."""
        self.R[train_task_id, eval_task_id] = accuracy
        
    def update_history(self, global_epoch, eval_task_id, accuracy):
        """Records accuracy at the end of every single epoch."""
        self.history[global_epoch, eval_task_id] = accuracy
        
    def compute_metrics(self):
        """Calculates final Average ACC, BWT, and Average Forgetting."""
        T = self.num_tasks
        
        acc = np.mean(self.R[T-1, :])
        
        bwt = 0.0
        forgetting = 0.0
        
        if T > 1:
            bwt_sum = sum([self.R[T-1, i] - self.R[i, i] for i in range(T-1)])
            bwt = bwt_sum / (T - 1)
            f_j = [np.max(self.R[:T-1, j]) - self.R[T-1, j] for j in range(T-1)]
            forgetting = np.mean(f_j)
            
        return {
            "Accuracy_Matrix": self.R,
            "Average_ACC": acc,
            "BWT": bwt,
            "Forgetting": forgetting
        }

    def export_matrix_to_csv(self, filepath):
        """Exports the Accuracy Matrix (R)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, self.R, delimiter=",", fmt="%.4f")

    # History Export Method
    def export_history_to_csv(self, filepath):
        """Exports the high-resolution E x T history matrix."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savetxt(filepath, self.history, delimiter=",", fmt="%.4f")

    def export_summary_dict(self):
        """Returns scalar metrics ready for JSON serialization."""
        metrics = self.compute_metrics()
        return {
            "Average_ACC": float(metrics["Average_ACC"]),
            "BWT": float(metrics["BWT"]),
            "Forgetting": float(metrics["Forgetting"])
        }