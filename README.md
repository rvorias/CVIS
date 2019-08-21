# Computer Vision Project

*Diana Triantafyllidou & Raphael Vorias*

This project is divided into four main sections:

1. A Convolutional Auto-Encoder
2. A Multi-label Classifier
3. A Dual AE/Classifier
4. An Image Segmentator

Each section consists of build files that constuct a model and save it locally.
Then, pipeline modules use the built models in order to train.
Images and masks are preproccesed and then handled via `flow_from_dataframe`.

Some sections, such as the AE section, contain visualization files.

While these models are not getting state of the art results, the 

## AE

Tested models:

|                    Model | Accuracy | Params |
|-------------------------:|----------|--------|
|         U-Net - unfrozen |   0.71   |   2 M  |
| Squeeze U-Net - unfrozen |   0.78   |  726 K |
|        Baseline - frozen |   0.65   |  116 K |
|      Baseline - unfrozen |   0.43   |  170 K |
|         Baseline - blank |   0.68   |  170 K |
|        Dual architecture |   0.66   |  440 K |

Example reconstruction:

![AE prediction][AE_combined]

## Classifier
Three variations were tested:

1. Baseline Scratch: baseline architecture with reinitialized weights.
2. Baseline Finetune: trained encoder part of the AE to finetune these pre-trained weights.
3. Baseline Frozen: trained encoder part of the AE which are frozen, only the last dense layers are trained.

Next to self-made models, U-Net architectures were used.

|                    Model | Accuracy | Params |
|-------------------------:|----------|--------|
|         U-Net - unfrozen |   0.71   |   2 M  |
| Squeeze U-Net - unfrozen |   0.78   |  726 K |
|        Baseline - frozen |   0.65   |  116 K |
|      Baseline - unfrozen |   0.43   |  170 K |
|         Baseline - blank |   0.68   |  170 K |
|        Dual architecture |   0.66   |  440 K |

![CL plots][CL_plots]

### Dual network
This network has both an AutoEncoder and a Classifier and are trained simultaneously.

Weight plots of the first convolutional layer:

| Untrained | Trained|
| ------------- |:-------------:|
| ![DU untrained][DU_untrained] | ![DU trained][DU_trained] |

TSNE-plot of the final dense layer:

![DU tsne][DU_tsne]

Confusion plot after 250 epochs:

![DU conf][DU_conf]

## Segmentor
Trained using Dice Loss.
Examples:

![SE ex1][SE_ex1]![SE ex2][SE_ex2]![SE ex3][SE_ex3]![SE ex4][SE_ex4]



[AE_combined]: Plots/AE/prediction_combined.png ""
[CL_plots]: Plots/CL/CL_plots_combined.png ""
[DU_untrained]: Plots/DU/DU_weights_untrained.png ""
[DU_trained]: Plots/DU/DU_weights_trained.png ""
[DU_tsne]: Plots/DU/DU_deep_tsne.png ""
[DU_conf]: Plots/DU/DU_confusion.png ""

[SE_ex1]: Plots/SE/ex1.png ""
[SE_ex2]: Plots/SE/ex2.png ""
[SE_ex3]: Plots/SE/ex3.png ""
[SE_ex4]: Plots/SE/ex4.png ""