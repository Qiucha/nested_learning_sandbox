import torch
import torch.nn as nn
from .metrics import CLEvaluator
from src.optimizers.factory import get_optimizer
from src.engine.replay import MemoryBuffer
from torch.utils.data import TensorDataset, DataLoader

def train_cl_scenario(model, tasks_train, tasks_test, task_classes, device, opt_name='SGD', epochs=5, lr=1e-3, f=20, alpha=0.5, beta3=0.9, stab=True, samples_per_class=0, replay_batch_size=32, ft_epochs=1, ft_lr=1e-4, replay_mode='bft'):
    """Executes the continual learning loop across all tasks, evaluating both CIL and TIL."""
    model = model.to(device)
    optimizer = get_optimizer(model, opt_name, lr=lr, f=f, stabilize=stab, alpha=alpha, beta3=beta3)
    criterion = nn.CrossEntropyLoss()
    
    num_tasks = len(tasks_train)
    do_bft = (samples_per_class > 0 and ft_epochs > 0 and replay_mode == 'bft')
    total_epochs_per_task = epochs + (ft_epochs if do_bft else 0)
    total_epochs = num_tasks * total_epochs_per_task
    
    # Instantiate dual evaluators with the new total_epochs parameter
    evaluator_cil = CLEvaluator(num_tasks=num_tasks, total_epochs=total_epochs)
    evaluator_til = CLEvaluator(num_tasks=num_tasks, total_epochs=total_epochs)
    
    global_epoch = 0 # Tracks absolute time across all task transitions

    steps_per_epoch = 0
    
    # Initialize MemoryBuffer
    memory = MemoryBuffer(samples_per_class, device)
    
    for task_id in range(num_tasks):
        train_loader = tasks_train[task_id]
        
        if replay_mode == 'blend' and not memory.is_empty():
            class IntTargetDataset(torch.utils.data.Dataset):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                def __len__(self):
                    return len(self.x)
                def __getitem__(self, idx):
                    return self.x[idx], self.y[idx].item()
                    
            combined_dataset = torch.utils.data.ConcatDataset([
                train_loader.dataset, 
                IntTargetDataset(memory.x, memory.y)
            ])
            train_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True)
            
        steps_per_epoch = len(train_loader)

        classes_str = ", ".join(map(str, task_classes[task_id]))
        print(f"\n[ Task {task_id + 1}/{num_tasks} ({classes_str}) | Optimizer: {opt_name} | Steps/Epoch: {steps_per_epoch} ]")
        
        def evaluate_and_log(is_final_epoch):
            model.eval()
            with torch.no_grad():
                for eval_id in range(task_id + 1):
                    test_loader = tasks_test[eval_id]
                    correct_cil, correct_til, total = 0, 0, 0
                    
                    valid_classes = task_classes[eval_id]
                    
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # 1. Class-IL Prediction
                        pred_cil = output.argmax(dim=1, keepdim=True)
                        correct_cil += pred_cil.eq(target.view_as(pred_cil)).sum().item()
                        
                        # 2. Task-IL Prediction
                        mask = torch.full_like(output, float('-inf'))
                        mask[:, valid_classes] = output[:, valid_classes]
                        pred_til = mask.argmax(dim=1, keepdim=True)
                        correct_til += pred_til.eq(target.view_as(pred_til)).sum().item()
                        
                        total += target.size(0)
                    
                    acc_cil = correct_cil / total
                    acc_til = correct_til / total
                    
                    # Log high-resolution data EVERY epoch
                    evaluator_cil.update_history(global_epoch, eval_id, acc_cil)
                    evaluator_til.update_history(global_epoch, eval_id, acc_til)
                    
                    # Log standard matrix data ONLY on the final epoch of the task
                    if is_final_epoch:
                        evaluator_cil.update_matrix(task_id, eval_id, acc_cil)
                        evaluator_til.update_matrix(task_id, eval_id, acc_til)
                        print(f"  -> [Task Boundary] Eval on Task {eval_id + 1} | CIL: {acc_cil:.4f} | TIL: {acc_til:.4f}")

        model.zero_grad()
        # --- Training Phase ---
        for epoch in range(epochs):
            model.train() # Make sure to set train mode inside the epoch loop
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                if replay_mode == 'blend_resample' and not memory.is_empty():
                    mem_data, mem_target = memory.sample(replay_batch_size)
                    if mem_data is not None:
                        data = torch.cat([data, mem_data])
                        target = torch.cat([target, mem_target])

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            evaluate_and_log(is_final_epoch=(epoch == epochs - 1 and not do_bft))
            global_epoch += 1

        # Update memory (of past tasks) at the end of the main task epochs
        memory.update_memory(tasks_train[task_id], task_classes[task_id])

        # --- Balanced Fine-Tuning (BFT) Phase ---
        if do_bft and not memory.is_empty():
            print(f"--- Running Balanced Fine-Tuning for {ft_epochs} epoch(s) ---")
            
            # Freeze internal model
            for param in model.parameters():
                param.requires_grad = False
                
            # Unfreeze head and collect its parameters
            head_params = []
            if hasattr(model, 'head'):
                if isinstance(model.head, nn.ModuleDict):
                    for head_name in model.head:
                        for param in model.head[head_name].parameters():
                            param.requires_grad = True
                            head_params.append(param)
                else:
                    for param in model.head.parameters():
                        param.requires_grad = True
                        head_params.append(param)
            else:
                for param in model.parameters():
                    param.requires_grad = True
                    head_params.append(param)
            
            # Use standard SGD for Linear Probing
            ft_optimizer = torch.optim.SGD(head_params, lr=ft_lr, momentum=0.9)
            
            mem_dataset = TensorDataset(memory.x, memory.y)
            mem_loader = DataLoader(mem_dataset, batch_size=replay_batch_size, shuffle=True)
            
            for ft_epoch in range(ft_epochs):
                model.train()
                for data, target in mem_loader:
                    data, target = data.to(device), target.to(device)
                    ft_optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    ft_optimizer.step()
                    
                evaluate_and_log(is_final_epoch=(ft_epoch == ft_epochs - 1))
                global_epoch += 1

            # Unfreeze the whole model for the next task
            for param in model.parameters():
                param.requires_grad = True

    return {
        'CIL': evaluator_cil.compute_metrics(),
        'TIL': evaluator_til.compute_metrics(),
        'evaluator_cil': evaluator_cil,
        'evaluator_til': evaluator_til,
        'steps_per_epoch': steps_per_epoch
    }