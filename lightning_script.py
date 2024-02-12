import pytorch_lightning as pl
from torch.optim import Adam
import torch
from torchvision.models import resnet152, resnet50, resnet34, mobilenet_v3_large, MobileNet_V3_Large_Weights, ResNet34_Weights
from metrics import Accuracy, precision, recall, f1_score
from torchmetrics import ConfusionMatrix
class LPRCharacterClassification(pl.LightningModule):
    def __init__(self,
                 lr: float = 1e-7,
                 num_classes: int = 48) -> None:
        super().__init__()
        self.lr = lr
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(512, num_classes)
        
        self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()
        self.training_step_outputs = []
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        return opt
    
    def training_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        self.training_step_outputs.append({'loss': loss, 'x': x, 'y_hat': y_hat, 'y': y})
        return {'loss': loss, 'x': x, 'y_hat': y_hat, 'y': y}
    
    # def on_train_epoch_end(self) -> None:
    #     batch = torch.stack(self.training_step_outputs)
    #     y = batch['y']
    #     y_hat = batch['y_hat']
    #     acc = Accuracy()(y_hat, y)
    #     prec = precision(y_hat, y)
    #     rcl = recall(y_hat, y)
    #     f1 = f1_score(y_hat, y)
        
    #     self.log('train_epoch_acc', acc, prog_bar=True, logger=True)
    #     self.log('train_epoch_prec', prec, prog_bar=True, logger=True)
    #     self.log('train_epoch_rcl', rcl, prog_bar=True, logger=True)
    #     self.log('train_epoch_f1', f1, prog_bar=True, logger=True)
        
    def validation_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = Accuracy()(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_prec', precision(y_hat, y), sync_dist=True)
        self.log('val_rcl', recall(y_hat, y), sync_dist=True)
        self.log('val_f1', f1_score(y_hat, y), sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_id):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        acc = Accuracy()(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss, y, y_hat
    
    # def test_epoch_end(self, outputs) -> None:
    #     _, y, y_hat = outputs[0]
    #     conf_mtx = ConfusionMatrix(num_classes=46, task="multiclass").to(self.device)
    #     conf_mtx = conf_mtx(y_hat, y)
    #     self.log(conf_mtx)
    #     return conf_mtx