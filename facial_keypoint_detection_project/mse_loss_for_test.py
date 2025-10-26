import torch.nn as nn

def net_test_mse():
    criterion = nn.MSELoss(reduction='sum')  # sum over all elements
    net.eval()  # evaluation mode
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  # no gradients needed
        for sample in test_loader:
            images = sample['image'].to(device).float()
            key_pts = sample['keypoints'].to(device).float()
            
            # forward pass
            output_pts = net(images)
            output_pts = output_pts.view(output_pts.size(0), 68, -1)
            key_pts = key_pts.view(key_pts.size(0), 68, -1)
            
            # accumulate loss
            loss = criterion(output_pts, key_pts)
            total_loss += loss.item()
            total_samples += images.size(0)
    
    # average per-sample MSE
    avg_loss = total_loss / total_samples
    print(f"Average MSE loss on test dataset: {avg_loss:.4f}")
    return avg_loss