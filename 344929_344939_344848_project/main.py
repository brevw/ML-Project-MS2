import argparse

import numpy as np
from torchinfo import summary
import time
from matplotlib import pyplot as plt

from src.data import load_data
from src.methods.pca import PCA
from src.methods.dummy_methods import DummyClassifier
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    if args.plotMLP:
        args.test = False

    # Make a validation set
    if not args.test:
        N = xtrain.shape[0]
        validation_size = int(N * 0.2)
        rand_idx = np.random.permutation(N)
        val_idx = rand_idx[:validation_size]
        train_idx = rand_idx[validation_size:]
        xtest = xtrain[val_idx,:]
        ytest = ytrain[val_idx]
        xtrain = xtrain[train_idx,:]
        ytrain = ytrain[train_idx]
        pass
    
        

    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        print(f'The total variance explained by the first {args.pca_d} principal components is {pca_obj.find_principal_components(xtrain):.3f} %')
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)



        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data

    # plotting functions
    if args.plotMLP:
        n_classes = get_n_classes(ytrain)

        # without pca
        model = MLP(xtrain.shape[1], n_classes)
        average_loss_epoch_list_without_pca = []
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, average_loss_list=average_loss_epoch_list_without_pca)
        train_without_pca_start = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        train_without_pca_stop = time.time()
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set without: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        acc_without_pca = accuracy_fn(preds, ytest)
        macrof1_without_pca = macrof1_fn(preds, ytest)
        print(f"Validation set without pca:  accuracy = {acc_without_pca:.3f}% - F1-score = {macrof1_without_pca:.6f}")

        # with pca
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        print(f'The total variance explained by the first {args.pca_d} principal components is {pca_obj.find_principal_components(xtrain):.3f} %')
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

        model = MLP(xtrain.shape[1], n_classes)
        average_loss_epoch_list_with_pca = []
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, average_loss_list=average_loss_epoch_list_with_pca)
        train_with_pca_start = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        train_with_pca_stop = time.time()
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set with: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        acc_with_pca = accuracy_fn(preds, ytest)
        macrof1_with_pca = macrof1_fn(preds, ytest)
        print(f"Validation set with pca:  accuracy = {acc_with_pca:.3f}% - F1-score = {macrof1_with_pca:.6f}")

        # plotting
        nbr_epoc = np.arange(1, args.max_iters + 1)
        plt.figure()
        plt.title("Performance analysis considering PCA")
        skip_factor = 5
        plt.plot(nbr_epoc[::skip_factor], average_loss_epoch_list_without_pca[::skip_factor], 'ro-' , label = f"w/o PCA: Time running {train_without_pca_stop - train_without_pca_start:.2f} - acc: {acc_without_pca:.2f} - f1: {macrof1_without_pca:.2f}")
        plt.plot(nbr_epoc[::skip_factor], average_loss_epoch_list_with_pca[::skip_factor], 'bo-', label = f"w/ PCA: Time running {train_with_pca_stop - train_with_pca_start:.2f} - acc: {acc_with_pca:.2f} - f1: {macrof1_with_pca:.2f}")
        plt.ylabel("average loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()
        exit(0)

    if args.plotCNN:
        n_classes = get_n_classes(ytrain)

        # lr = 1e-5, batch_size = 64
        args.lr = 1e-5
        
        model = CNN(1, n_classes)
        average_loss_epoch_list_1 = []
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, average_loss_list=average_loss_epoch_list_1)
        train_start_1 = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        train_stop_1 = time.time()
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set without: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        acc_1 = accuracy_fn(preds, ytest)
        macrof1_1 = macrof1_fn(preds, ytest)
        print(f"Validation set without pca:  accuracy = {acc_1:.3f}% - F1-score = {macrof1_1:.6f}")

        # lr = 1e-3 
        args.lr = 1e-3

        model = CNN(1, n_classes)
        average_loss_epoch_list_2 = []
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, average_loss_list=average_loss_epoch_list_2)
        train_start_2 = time.time()
        preds_train = method_obj.fit(xtrain, ytrain)
        train_stop_2 = time.time()
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set with: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        acc_2 = accuracy_fn(preds, ytest)
        macrof1_2 = macrof1_fn(preds, ytest)
        print(f"Validation set with pca:  accuracy = {acc_2:.3f}% - F1-score = {macrof1_2:.6f}")

        # plotting
        nbr_epoc = np.arange(1, args.max_iters + 1)
        plt.figure()
        plt.title("Performance analysis considering PCA")
        skip_factor = 5
        plt.plot(nbr_epoc[::skip_factor], average_loss_epoch_list_1[::skip_factor], 'ro-' , label = f"w/ lr -> 1e-5: Time running {train_start_1 - train_stop_1:.2f} - acc: {acc_1:.2f} - f1: {macrof1_1:.2f}")
        plt.plot(nbr_epoc[::skip_factor], average_loss_epoch_list_2[::skip_factor], 'bo-', label = f"w/ lr -> 1e-3: Time running {train_start_2 - train_stop_2:.2f} - acc: {acc_2:.2f} - f1: {macrof1_2:.2f}")
        plt.ylabel("average loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()
        exit(0)



    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes)
    elif args.nn_type == "cnn":
        model = CNN(1, n_classes)
    elif args.nn_type == "transformer":
        model = MyViT((1, 28, 28), 7, 2, 8, 2, n_classes)
    else :
        model = DummyClassifier(0) 
    summary(model)

    # Trainer object
    average_loss_epoch_list = []
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, average_loss_list=average_loss_epoch_list)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    if not args.test:
        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")




    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!
    parser.add_argument('--method', default="dummy-classifier", type=str, help="method to run")
    parser.add_argument('--plotMLP', action="store_true",
                        help="show performance graph (kinda like cross-validation) on a MLP model")
    parser.add_argument('--plotCNN', action="store_true",
                        help="show performance graph (kinda like cross-validation) on a MLP model")
    

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)