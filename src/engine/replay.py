import torch

class MemoryBuffer:
    def __init__(self, samples_per_class, device):
        self.samples_per_class = samples_per_class
        self.device = device
        self.x = None
        self.y = None

    def is_empty(self):
        return self.x is None or self.x.shape[0] == 0

    def sample(self, batch_size):
        if self.is_empty():
            return None, None
            
        num_samples = self.x.shape[0]
        bs = min(batch_size, num_samples)
        indices = torch.randperm(num_samples)[:bs]
        
        return self.x[indices].to(self.device), self.y[indices].to(self.device)

    def update_memory(self, dataloader, classes_in_task):
        if self.samples_per_class <= 0:
            return
            
        collected_x = {c: [] for c in classes_in_task}
        collected_counts = {c: 0 for c in classes_in_task}
        
        for data, target in dataloader:
            for i in range(len(target)):
                lbl = target[i].item()
                if lbl in classes_in_task and collected_counts[lbl] < self.samples_per_class:
                    collected_x[lbl].append(data[i].unsqueeze(0)) # keep batch dim
                    collected_counts[lbl] += 1
                    
            if all(count >= self.samples_per_class for count in collected_counts.values()):
                break
                
        # Aggregate
        new_x = []
        new_y = []
        for c in classes_in_task:
            if len(collected_x[c]) > 0:
                new_x.append(torch.cat(collected_x[c]))
                new_y.append(torch.full((len(collected_x[c]),), c, dtype=torch.long))
                
        if len(new_x) > 0:
            new_x = torch.cat(new_x)
            new_y = torch.cat(new_y)
            
            # Store in CPU memory to save GPU memory, send to device during sample()
            if self.x is None:
                self.x = new_x.cpu()
                self.y = new_y.cpu()
            else:
                self.x = torch.cat([self.x, new_x.cpu()])
                self.y = torch.cat([self.y, new_y.cpu()])
