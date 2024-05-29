import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from src.utils import onehot_to_label, label_to_onehot


## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.mlp = nn.Sequential(
            # Linear block 1
            nn.Linear(input_size, 128      , bias=True),
            nn.ReLU(),
            # Linear block 2
            nn.Linear(128       , 32      , bias=True),
            nn.ReLU(),
            #Linear block 3
            nn.Linear(32       , 16      , bias=True),
            nn.ReLU(),
            ## linear block 4
            nn.Linear(16      , n_classes, bias=True)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        
        preds = self.mlp(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = self.cnn(x)

        return preds

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patch_size = h // n_patches

    # Unfold the image into patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n, n_patches**2, -1)

    return patches
def get_positional_embeddings(sequence_length, d):
    position = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2).float() * (-torch.log(torch.tensor(10000.0)) / d))
    
    pos_emb = torch.zeros(sequence_length, d)
    pos_emb[:, 0::2] = torch.sin(position * div_term)
    pos_emb[:, 1::2] = torch.cos(position * div_term)
    
    return pos_emb
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        self.d_head = d // n_heads

        # Linear layers for Q, K, V transformations
        self.q_mappings = nn.Linear(d, d)
        self.k_mappings = nn.Linear(d, d)
        self.v_mappings = nn.Linear(d, d)

        # Output linear layer
        self.output_linear = nn.Linear(d, d)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences shape: (N, seq_length, token_dim)
        N, seq_length, token_dim = sequences.shape

        # Linear projections
        Q = self.q_mappings(sequences)
        K = self.k_mappings(sequences)
        V = self.v_mappings(sequences)

        # Reshape for multi-head attention (N, seq_length, n_heads, d_head)
        Q = Q.view(N, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(N, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(N, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(N, seq_length, self.d)
        output = self.output_linear(attention_output)

        return output

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super().__init__()
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] // n_patches, chw[2] // n_patches) ### patch size in terms of height and width

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings =  get_positional_embeddings(n_patches **2 +1 ,hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, c, h, w = x.shape

        # Divide images into patches.
        patches = patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        preds = tokens + self.positional_embeddings.repeat(n,1,1)

        # Transformer Blocks
        for block in self.blocks:
            preds = block(preds)

        # Get the classification token only.
        preds = preds[:, 0]

        # Map to the output distribution.
        preds = self.mlp(preds)

        return preds

class CustomWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(CustomWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step < self.warmup_steps:
            return [base_lr * (current_step / self.warmup_steps) for base_lr in self.base_lrs]
        return [base_lr * (self.total_steps - current_step) / (self.total_steps - self.warmup_steps) for base_lr in self.base_lrs]


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, average_loss_list):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.average_loss_list = average_loss_list

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) 

        num_training_steps = epochs * 60000 // batch_size
        self.num_warmup_steps = int(0.1 * num_training_steps)
        self.scheduler = CustomWarmupScheduler(self.optimizer, warmup_steps=self.num_warmup_steps, total_steps=num_training_steps)


    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        running_loss = 0
        # Used to add average of one epoch to the list for plotting
        number_of_samples = 0
        running_loss_on_epoch = 0
        train_loss = 0.0
        N_EPOCHS = self.epochs
        for it, batch in enumerate(dataloader):
            images ,targets = batch

            #fwd + bwd + optimize
            logits = self.model(images)
            loss = self.criterion(logits, targets.long())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.detach().cpu().item() / len(dataloader)

            # For plotting
            
            running_loss_on_epoch += loss.item()
            number_of_samples += 1

            #zero gradients
            self.optimizer.zero_grad()

        print(f"Epoch {ep + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        self.average_loss_list.append(running_loss_on_epoch / number_of_samples)


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for batch in dataloader:
                output = self.model(batch[0])
                pred_labels.append(output)
            pred_labels = torch.cat(pred_labels)
        return onehot_to_label(F.softmax(pred_labels, dim = 1))
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(transforms.Normalize((0.5,), (0.5,))(torch.from_numpy( training_data).float()), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch  
        test_dataset = TensorDataset(transforms.Normalize((0.5,), (0.5,))(torch.from_numpy(test_data).float()))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()