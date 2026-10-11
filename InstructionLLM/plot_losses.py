import matplotlib.pyplot as plt
def plot_loss(num_epochs, tokens_seen, train_losses, val_losses, name="losses.png"):
    # axes-x : epoches
    epochs_seen = [i * num_epochs / len(train_losses) for i in range(len(train_losses))]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # axe with tokens seen 
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # use this to calibrate the axes
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(name)
    plt.show()