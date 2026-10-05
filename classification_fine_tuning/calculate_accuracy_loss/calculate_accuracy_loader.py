"""This file contains the methodw which calculates accuracy on classification prediction"""
import torch
class CalcAccuracy:
    def __init__(self):
        pass


    #accuracy loader method
    @staticmethod
    def cacl_accuracy_loader(data_loader,model,device, num_batches=None):
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None: #if batches has not been delegated we use data_loader's length instead
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                with torch.no_grad():
                    logits = model(input_batch)[:, -1, :]
                predicted_labels = torch.argmax(logits, dim=-1)


                num_examples += predicted_labels.shape[0] #compute the number of the predictions
                correct_predictions += ((predicted_labels == target_batch).sum().item()) #calculate correct predictions
            else:
                break
        return correct_predictions / num_examples #divide correct predictions with the number of the examples to computer accuracy precentage