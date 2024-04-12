---
layout: post
title: "Finetuning models for alternative uses"
date: 2024-4-10 11:58:00 +0000
categories: research notes pytorch

---

The standard use of model finetuning is fairly easy to get started with. If we just want to adjust the parameters of the entire model to change the type of classification task we're performing then we can create a new model to wrap the existing one, then change the output dimension and retrain. We can achive this with this code:

```python
class NewNet(torch.nn.Module):
    def __init__(self, new_output_shape):
        super(NewNet, self).__init__()
        self.googlenet = torchvision.models.googlenet(weights='IMAGENET1K_V1')
        output = torch.nn.Linear(self.googlenet.fc.in_features, new_output_shape)
        self.googlenet.fc = output

    def forward(self, x):
        logits = self.googlenet(x)
        return logits
```

Here we're taking a [GoogLeNet](https://pytorch.org/vision/stable/models/googlenet.html) model from [torchvision](https://pytorch.org/vision/stable/index.html), loading the pre-trained weights and adding a custom `Linear` layer at the top to manipulate the output shape to whatever we need it to be for the classifcation task we're trying to solve. Then in the `forward` function we just pass it to the GoogLeNet and get the output logits. Once this has been set up, we'd just train the network as normal and (provided the classification task isn't too far from the original task of the pre-trained network), it should converge to a good solution quite quickly since we're leveraging the features the network has already learned.

Things get a bit more complicated if we want to freeze a part of the network before retraining! In the previous example we were allowing the entire network to be retrained, so parameters from the very early or very late layers could be adjusted to help us with the current classification task. If we freeze layers, we're specifically telling the parameters in those layers not to change. The early layers of a network are usually frozen since they extract the more fundamental aspects of the data which does not change massively between similar datasets (e.g. vision based networks usually extract [Gabor filters](https://en.wikipedia.org/wiki/Gabor_filter) in the first few layers). Freezing these layers can speed up network training which can be critical depending on the amount of compute you have available to you. 

The layer freezing is in the code below:

```python
def freeze_weights(self, threshold_layer, verbose=False, invert=False):
    """
    threshold_layer (str): This and subsequent layers will be re-initialised
    invert (bool): Invert the freezing/non-freezing (early layers not frozen, later layers frozen)
    verbose (bool): Tell the user which layers are frozen/not frozen
    """
    freeze_flag = False | invert
    flag_changed = False
    # Layer types to ignore when not freezing/freezing
    ignore_layer_set = (torchvision.models.GoogLeNet, torch.nn.Sequential, BasicConv2d, Inception, # Should be ignored since they're not executable
                        torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.Dropout) # These don't have trainable parameters

    layers = self.googlenet.named_modules()
    
    for idx, layer_data in enumerate(layers): 
        name, layer = layer_data

        if not flag_changed and name == threshold_layer:
            freeze_flag = not freeze_flag
            flag_changed = True
        
        if not isinstance(layer, ignore_layer_set):
            if not freeze_flag:
                for p in layer.parameters():
                    p.requires_grad = False
                
                if verbose:
                    print("Layer {} <- Frozen".format(name))
            else:
                if verbose:
                    print("Layer {} <- Not frozen".format(name))
```

The above code allows us to specify a layer in the network by name and freeze all layers before it, and not freeze all layers after it. This is achieved by setting the `requires_grad` property of the parameters of the layer to `False`. We also have some extra code here (`invert`) which allows us to invert the freezing (i.e. un-freeze all layers before the named layer and freeze the named layer and all others after).

To get the names of the layers from the model we can use the following helper function:

```python
def view_available_layers(model):
    """
    Print all available layers

    Parameters:
    - model (torch.nn.Module): The model to visualise the layer of
    """
    for name, layer in model.named_modules():
        print("{}".format(name))
```

Finally, we can do something more interesting! When layers are kept trainable they still start from their pre-trained state. However, it's interesting to think about what the networks will learn if we keep some pre-trained layers frozen and then reset the other trainable layer parameters to a random initialisation. The effect that this reset has on the model is the focus of our next stage of research (so watch this space!). We can take the `freeze_weights` code from above and make some small adjustments to reset the parameters.    

```python
def freeze_and_init_weights(self, threshold_layer, verbose=False, invert=False):
    """
    threshold_layer (str): This and subsequent layers will be re-initialised
    invert (bool): Invert the freezing/reinitialisation (early layers reinitialised, later layers frozen)
    verbose (bool): Tell the user which layers are frozen/not frozen
    """
    reinit_flag = False | invert
    flag_changed = False
    # Layer types to ignore when resetting/freezing
    ignore_layer_set = (torchvision.models.GoogLeNet, torch.nn.Sequential, BasicConv2d, Inception, # Should be ignored since they're not executable
                        torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.Dropout) # These don't have trainable parameters

    layers = self.googlenet.named_modules()

    for idx, layer_data in enumerate(layers):
        name, layer = layer_data

        if not flag_changed and name == threshold_layer:
            reinit_flag = not reinit_flag
            flag_changed = True

        if not isinstance(layer, ignore_layer_set):
            if not reinit_flag:
                for p in layer.parameters():
                    p.requires_grad = False

                if verbose:
                    print("Layer {} <- Frozen".format(name))
            else:
                if isinstance(layer, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
                    layer.reset_running_stats()
                elif isinstance(layer, ignore_layer_set):
                    pass
                else:
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias != None:
                        torch.nn.init.zeros_(layer.bias)
                if verbose:
                    print("Layer {} <- Reset".format(name))
```

This code has the reinitialisation of the unfrozen layers included. We set the weights to a random initialisation and the biases are reset to 0. The thing that took me a while to recognise and fix, is that if we don't reset the paramaters (weights and biases) correctly, the model will consistently fail to learn anything! The one that caught me out is that the weights of the `BatchNorm2d` layer need to be set to 1 when re-initialising! This took me a while to fix, but I got there in the end!
