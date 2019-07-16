import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../"))
import utils
from compression_utils import ActivationUniformQuantizerBwOnly
import logging
from subprocess import check_output

# Hyper-parameters 
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
hidden_size = 128
# we quantize the second linear layer of the 2layer MLP
sample_act_shape = [batch_size * 100, hidden_size]
# learning_rate = 0.1

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

def setup_bw_comp_act(model, nbit, sample_act_shape, do_stoc, do_auto_clip):
  model.fc2.quantizer = ActivationUniformQuantizerBwOnly(model.fc2, nbit, sample_act_shape,
    do_stoc=do_stoc, do_auto_clip=do_auto_clip)

def collect_sample_to_estimate_clip_threshold(model, train_loader):
  for i, (images, labels) in enumerate(train_loader):
    # Reshape images to (batch_size, input_size)
    images = images.reshape(-1, 28*28)
    if torch.cuda.is_available():
      images = images.cuda()
      labels = labels.cuda()
    # Forward pass
    outputs = model(images)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp_name", type=str, help="The name of the experiment.")
  parser.add_argument("--result_dir", type=str, help="The top level of experiment result directory.")
  parser.add_argument("--use_quant", action="store_true", help="Use activation quantization for backward.")
  parser.add_argument("--quant_nbit", type=int, help="# of bits for quantized activation.")
  parser.add_argument("--quant_clip", action="store_true", help="Do auto clipping for activations.")
  parser.add_argument("--quant_stoc", action="store_true", help="Use stochastic quantization for activation.")
  parser.add_argument("--lr", type=float, default=0.1, help="The learning rate for training")
  parser.add_argument("--seed", type=int, help="Random seed for the run.")
  parser.add_argument("--debug", action="store_true", help="If the job is in debuging mode, git diff must be empty if not specified.")
  args = parser.parse_args()

  init_results = utils.experiment_setup(args)

  run_folder = utils.get_output_folder(args)
  writer = SummaryWriter(log_dir=run_folder)

  # MNIST dataset (images and labels)
  train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

  test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

  # Data loader (input pipeline)
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

  # Logistic regression model
  # model = nn.Linear(input_size, num_classes)
  model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
  setup_bw_comp_act(model, args.quant_nbit, sample_act_shape=sample_act_shape,
    do_stoc=args.quant_stoc, do_auto_clip=args.quant_clip)
  if torch.cuda.is_available():
    model = model.cuda()

  # Loss and optimizer
  # nn.CrossEntropyLoss() computes softmax internally
  criterion = nn.CrossEntropyLoss()
  if torch.cuda.is_available():
    criterion = criterion.cuda()  
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  

  # Train the model
  train_loss_list = []
  test_acc_list = []
  total_step = len(train_loader)
  for epoch in range(num_epochs):
      if args.use_quant:
        # determine clipping threshold if necessary
        model.eval()
        model.fc2.quantizer.start_per_epoch_setup()
        if args.quant_clip:
          collect_sample_to_estimate_clip_threshold(model, train_loader)
          logging.info("Collected sample for clip threshold estimation for epoch {}".format(epoch) )
        model.fc2.quantizer.end_per_epoch_setup()
        # run the training steps
        model.train()
        logging.info("Pre training procedures done for epoch {}".format(epoch) )
      
      for i, (images, labels) in enumerate(train_loader):
          # start quantizer for new epoch
          model.fc2.quantizer.start_epoch()

          # Reshape images to (batch_size, input_size)
          images = images.reshape(-1, 28*28)
          if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
          
          # Forward pass
          outputs = model(images)
          loss = criterion(outputs, labels)

          writer.add_scalar(tag="train_loss", scalar_value=loss.item(), global_step=epoch * total_step + i)
          # logging.info("Train loss step {}: {}".format(epoch * total_step + i, loss.item()))
          train_loss_list.append(loss.item())

          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          if (i+1) % 100 == 0:
              logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                     .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
          # turn off quantizer at the end of each epoch
          model.fc2.quantizer.end_epoch()


      # Test the model
      # In test phase, we don't need to compute gradients (for memory efficiency)
      with torch.no_grad():
          model.eval()
          model.fc2.quantizer.start_epoch()
          correct = 0
          total = 0
          for images, labels in test_loader:
              images = images.reshape(-1, 28*28)
              if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

              outputs = model(images)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum()
          model.fc2.quantizer.end_epoch()
          model.train()
          writer.add_scalar(tag="test_acc", scalar_value=100 * float(correct) / float(total), global_step=total_step * (epoch + 1))
          test_acc_list.append(100 * float(correct) / float(total))
          logging.info('Accuracy of the model on the 10000 test images: {} %'.format(100 * float(correct) / float(total)))

  writer.close()

  results = {"train_loss": train_loss_list, 
    "test_acc": test_acc_list}
  results.update(init_results)
  utils.save_result_json(args, results, run_folder)
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

if __name__ == "__main__":
  main()

