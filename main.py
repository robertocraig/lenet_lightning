import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import LeNet5Lightning
from data_prep import MNISTDataModule
import argparse

def main():
    # Configuração do ArgumentParser
    parser = argparse.ArgumentParser(description='Treinamento do modelo LeNet-5 no dataset MNIST')

    # Definindo argumentos de linha de comando
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas para treinar (default: 6)')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamanho do batch para o DataLoader (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Taxa de aprendizado para o otimizador (default: 0.1)')
    
    # Argumentos para acelerador e dispositivos
    if torch.cuda.is_available():
        parser.add_argument('--accelerator', type=str, default='gpu', help='Tipo de acelerador para usar (default: gpu se disponível)')
        parser.add_argument('--devices', type=int, default=1, help='Número de GPUs para usar (default: 1)')
    else:
        parser.add_argument('--accelerator', type=str, default='cpu', help='Tipo de acelerador para usar (default: cpu)')
        parser.add_argument('--devices', type=int, default=1, help='Número de CPUs para usar (default: 1)')
    
    args = parser.parse_args()

    # Inicializando o datamodule e o modelo
    mnist_dm = MNISTDataModule(batch_size=args.batch_size)
    model = LeNet5Lightning(learning_rate=args.learning_rate)

    # Inicializando o trainer com os argumentos atualizados
    trainer = Trainer(
        max_epochs=args.epochs, 
        accelerator=args.accelerator, 
        devices=args.devices
    )

    # Treinamento do modelo
    trainer.fit(model, datamodule=mnist_dm)

    # Salvando o modelo treinado
    trainer.save_checkpoint("lenet5_mnist.ckpt")
    print("Modelo treinado salvo como lenet5_mnist.ckpt")

if __name__ == "__main__":
    main()