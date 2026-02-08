import torch.nn as nn

class TandemModel(nn.Module):
    def __init__(self, forward_model, inverse_model):
        super(TandemModel, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        
        # freeze forward model
        self.forward_model.eval()
        for param in self.forward_model.parameters():
            param.requires_grad = False

    def forward(self, forward_y):
        # forward_y is the target of the forward model (R)
        # predicted_inverse_y is the target\
        #   predicted by the inverse model (geo) using the forward model's\
        #   prediction as input
        predicted_inverse_y = self.inverse_model(forward_y)
        
        # reconstruct the target of the forward model (R)\
        #   by passing the predicted target of the inverse model (geo\
        #   in the frozen forward model
        reconstructed_forward_y = self.forward_model(predicted_inverse_y)
        
        return predicted_inverse_y, reconstructed_forward_y
