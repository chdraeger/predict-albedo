# Predict mountain glacier albedo values
## Train, validate and test neural networks for albedo prediction

Code for the albedo prediction in the study of Draeger et al. (2024), *Assessing the impact of calibration and albedo models on regional glacier modeling with surface energy balance in southwestern Canada*

Below is a short guide to the data used, and for reproducing the training and testing.

## Data

**Input:** Meteorological and topographic data

Shape constants for the input data are defined in `CONSTANTS.py`.

**Output:** Modis albedo

### Data used in this study
**Input:** Daily climatic values, as well as overall climatic and topographic characteristics for each MODIS grid cell, including:
- Day of year
- $T_{t_{0}}$, $T_{t_{-1}}$, $T_{t_{-2}}$, $T_{t_{-3}}$, $T_{t_{-4}}$ and $T_{t_{-5}}$: \qty{2}{\metre} temperature, including lags of up to 5 days
- $P_{t_{0}}$, $P_{t_{-1}}$, $P_{t_{-2}}$, $P_{t_{-3}}$, $P_{t_{-4}}$ and $P_{t_{-5}}$: Total precipitation, including lags of up to 5 days
- MODIS grid cell elevation
- Normalized elevation (relative to the glacier's minimum and maximum elevation)
- Slope
- Aspect
- Multi-year average temperature range, including anomaly (anomalies indicate the discrepancy of the current mass balance year to the multi-year averages)
- Multi-year average summer temperature, including anomaly
- Multi-year average total winter precipitation, including anomaly

**Output:** Daily modis albedo values for each glacier grid cell based on [Solvik et al. (2019)](https://doi.org/10.3334/ORNLDAAC/1605)

## Methods

Code for training, validating and testing two different machine learning albedo models:
- Feed-forward Neural Network (FNN) 
- Long-short term memory (LSTM)

The neural network albedo models employed standard architectures with dropout layers of up to 0.2. The FNN albedo model has approximately 180k parameters across four layers, while the LSTM model contains around 35k parameters. We used 30 epochs, a batch size of 1024, Adam optimizer with a learning rate 0.01, and the Mean Squared Error (MSE) as the loss function.
