# Lithology Identification with Federated Learning

This repository contains the implementation of a Federated Learning approach for lithology identification using the [FORCE 2020 Lithology Competition Dataset](https://www.sodir.no/globalassets/2-force/2020/seminars/contest-machine-learning/blog-post-litho-facies-competition-geological-summary.pdf). The project leverages the Flower framework for Federated Learning and XGBoost for multi-class classification.

### `Important Warning`
The federated learning code with Flower in this repository is deprecated. The framework has completely reworked the code base. Viewers should follow the new tutorial available at [Flower Quickstart Tutorial with XGBoost](https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html).

## Introduction

In the realm of machine learning, Federated Learning (FL) has emerged as a transformative approach, especially in scenarios where data privacy is paramount. Unlike the conventional centralized model training method, FL offers a decentralized way of training models across devices or servers while ensuring data remains localized and private.

This project focuses on transitioning from Centralized Learning (CL) to Federated Learning (FL) in the context of multi-class classification using authentic geological data.

## Flower Framework

Flower is an open-source platform designed to simplify Federated Learning experiments. It provides a user-friendly codebase, built-in communication protocols, and compatibility with popular machine learning frameworks such as PyTorch and TensorFlow. In comparison to other Federated Learning frameworks like TensorFlow Federated (TFF), Flower stands out due to its consistent maintenance, documentation, and community support.

## Dataset

The dataset used for this project is well-log data from the FORCE 2020 competition in lithology identification. It comprises 118 wells in the South and North Viking graben (Norway), with 98 wells for training and 10 wells for testing.

## Centralized Learning

The centralized learning experiment was originally conducted by Ibrahim Olawale, the winner of the competition. This repository builds upon his work, refactoring and cleaning the code for improved readability. The centralized learning approach yielded promising results, but faced challenges with class imbalance and similarities in log properties for certain lithology classes.

## Federated Learning

Federated Learning can potentially address the challenges faced in centralized learning and enhance the model's capability to distinguish between similar rock types. The training process in this project employs federated XGBoost in a horizontal setting.

## Navigate through the repository

- **XGBoost**: Centralized learning with XGBoost.
- **fl**: Implementation of federated learning.
- **cl**: Attempt of achieving lithology classification with Image Segmentation (SegLog).
- **data_transform**: Split data between clients, perform val-train split, and convert well log data into images for image segmentation.


## Further Reading

For a detailed walkthrough of the project, its motivations, and the intricacies of the implementation, refer to the accompanying [Medium article](https://medium.com/@hyonbokan/federated-learning-for-multi-class-xgboost-with-flower-fl-force-2020-lithology-competition-d38b1177db64).
