import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy

class LeNet5Lightning(pl.LightningModule):
    def __init__(self, learning_rate=0.1):
        super(LeNet5Lightning, self).__init__()
        self.save_hyperparameters()  # Salva os hiperparâmetros (como a taxa de aprendizado)
        # Definindo as camadas convolucionais
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Definindo as camadas totalmente conectadas (fully connected)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Função de perda
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Definindo a passagem para frente (forward pass)
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        # Correção: adicionar o argumento 'task' para especificar o tipo de problema de classificação
        acc = accuracy(y_pred.softmax(dim=-1), y, task='multiclass', num_classes=10)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# Testando o modelo para verificar se não há erros de sintaxe
if __name__ == "__main__":
    model = LeNet5Lightning()
    print(model)