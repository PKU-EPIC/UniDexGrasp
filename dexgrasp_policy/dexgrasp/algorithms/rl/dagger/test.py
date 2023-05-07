import torch
goal_dist = torch.tensor([0,0,2,2,0])
successes = torch.zeros(5)
successes = torch.where(goal_dist<=0.05,torch.ones_like(successes),successes)
print(successes)