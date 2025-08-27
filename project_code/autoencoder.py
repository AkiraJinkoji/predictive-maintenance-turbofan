# Autoencoders Definition
#Imports 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses
from keras.models import Model

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        '''
        Initialization of constructor 
        Inputs: 
                latend_dim: dimension of the latent space (bottle neck)
                     shape: shape of input data
        '''
        # Call constructor of parent class
        super(Autoencoder, self).__init__()   
        self.latent_dim = latent_dim 
        self.shape = shape

        # Encoder: LSTM Network for temporal dependencies
        self.encoder = tf.keras.Sequential([
            layers.LSTM(64, activation='relu', return_sequences=True, input_shape=shape),
            layers.LSTM(latent_dim, activation='relu')      # bottleneck
        ])        
        # Decoder: Reconstruct original sequence
        self.decoder = tf.keras.Sequential([
            layers.RepeatVector(shape[0]),                  # repeats latent vector for timesteps
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.TimeDistributed(layers.Dense(shape[1]))  # reconstructs all 24 features
        ])


    def call(self, x):
        '''
        Defines forward pass to the model 
        Input:         x: tensor
        Output:  decoded: reconstructed output
        '''
        # Pass x through encoder to get latent representation
        encoded = self.encoder(x)  
        # Pass latent representation through the decoder to reconstruct the input
        decoded = self.decoder(encoded)
        return decoded
    
