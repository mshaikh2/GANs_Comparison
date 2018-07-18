# MNIST reconstruction using Convnet, Neuralnet and CapsuleNets


## Deep Convolutionalnet GAN
The below GIF displays the sample of images generated from epoch 1 to 50 at every 5 epochs.

Conv layers enable GANs to generate better images much faster than neural net.

Each epoch takes around 60 seconds

![Images_generated_using_conv_net](/images/gan_cnn/digits/cnn_epoch_1_50.gif?raw=true "Images Generated using Conv Layers in GAN architecture")

### Graph of Loss over 50 epochs
![Graph1](/images/gan_cnn/conv_gan_loss.png?raw=true "Graph of the loss over 50 epochs")

## Deep Neuralnet GAN
The below GIF displays the sample of images generated from epoch 1 to 200 at every 20 epochs.

Neural net enables GANs to generate decent images but after much longer training epochs.

Each epoch takes around 15 seconds.

![Images_generated_using_conv_net](/images/gan_neuralnet/digits/gan_nn_epoch_1_to_200.gif?raw=true "Images Generated using NeuralNet Layers in GAN architecture")

## Capsule Nets
The below GIF displays the sample of images generated from epoch 1 to 9 at every epoch.

At the decoder end a 28x28 image is reconstructed by passing the latent vector along with its true class variable through two fully connected layers

Each epoch takes around 55 mins seconds.

![Images_generated_using_caps_net](/images/capsulenet/Selected/epochs.gif?raw=true "Images Generated using CapsNet")

### Graph of Loss over 9 epochs
![Graph3](/images/capsulenet/capsnet_graph.jpg?raw=true "Graph of the loss and accuracy over 9 epochs")

## Libraries
#### Tensorflow
#### Keras
#### openCV
#### PIL
#### numpy

## Refrences
#### [1] GANs, https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
#### [2] https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
#### [3] https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
#### [4] https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/
#### [5] https://kndrck.co/posts/capsule_networks_explained/
#### [6] https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html
#### [7] Overview of GANs, https://arxiv.org/pdf/1710.07035.pdf
#### [8] Capsule Nets, https://arxiv.org/pdf/1710.09829.pdf
#### [9] https://github.com/XifengGuo/CapsNet-Keras


## API USAGE
#### API_ENDPOINT:

    http://rbc-ai.eastus.cloudapp.azure.com:5001
    
#### API_URL:
    
    /api/v1/frequency/daily/steps/33/threshold/30/percent_offset/20/range_days/7

#### API_INPUT_TYPE: 
    
    application/JSON
    
#### API_INPUT_DATA_FORMAT:
```        
{
    "0": {
        "account_id": 33095452,
        "amount_amount": 79.78,
        "base_type": "CREDIT",
        "category": "Retirement Contributions",
        "date": "2017-03-22",
        "description_simple": null,
        "id": 50332312,
        "merchant_name": null
    },
    "1": {
        "account_id": 33095452,
        "amount_amount": 39.89,
        "base_type": "CREDIT",
        "category": "Retirement Contributions",
        "date": "2017-03-22",
        "description_simple": null,
        "id": 50332316,
        "merchant_name": null
    },
    "2": {
        "account_id": 33095452,
        "amount_amount": -2573.45,
        "base_type": "DEBIT",
        "category": "Transfers",
        "date": "2017-04-03",
        "description_simple": null,
        "id": 50332320,
        "merchant_name": null
    },
    "3": {
        "account_id": 33095452,
        "amount_amount": -2559.84,
        "base_type": "DEBIT",
        "category": "Transfers",
        "date": "2017-04-03",
        "description_simple": null,
        "id": 50332324,
        "merchant_name": null
    },
    ...
}
```       
## CODE USAGE
#### MODULE_NAME:

    nonRecurringForecast
    
#### METHOD_NAME:

    getNonRecurringForecast(df, steps, account_id)
 	
#### METHOD_PARAMETERS:
    
    df: A pandas dataframe containing the following columns
        + index (optional) (type int)
        + date (must be the indexed column if no index column is present) (type Datetime)
        + amount (numeric value) (type float32)
    steps: Integer value for the numebr of expected daily forecast steps
    account_id: an Integer that acts as unique identifier for the input dataframe
        
#### USAGE Example:
    
    import nonRecurringForecast
    n = 123456789
    steps = 32
    df = pd.read_json(jsonData)
    nonRecForecast = nonRecurringForecast.getNonRecurringForecast(df, steps, account_id = n)
    
 ### RECURRING: OUTPUT JSON FORMAT 
 
 #### Recurring transactions
 
 ```
{
    "33095400": {
        "0": {
            "account_id": 33095400,
            "date": "2017-05-25",
            "description_simple": "Transfer",
            "category": "Transfers",
            "merchant_name": null,
            "base_type": "CREDIT",
            "amount_amount": 89
        },
        ,
        "1": {
            "account_id": 33095400,
            "date": "2017-06-08",
            "description_simple": "Transfer",
            "category": "Transfers",
            "merchant_name": null,
            "base_type": "CREDIT",
            "amount_amount": 89
        },
        ...
    }
}
```

### NON-RECURRING: OUTPUT JSON FORMAT

#### When datapoints less than 30 rows in DataFrame
```
{
    "33095400": {
        "Account_Id": 33095400,
        "NR_scores": {},
        "NR_future_data": {},
        "NR_test_data_raw": {},
        "NR_outliers": {},
        "Message": "Number of data points less than 30"
    }
}
```

#### When datapoints greater than 30 rows in DataFrame
```
{
    "33095416": {
        "Account_Id": 33095416,
        "NR_scores": {
            "arima_error": 7034.520686033828,
            "prophet_error": 4112.1542510900645
        },
        "NR_future_data": {
            "0": {
                "ds": "2017-07-02",
                "yhat_lower": -864.3432091809314,
                "yhat_upper": 3911.282141106367,
                "yhat": 1503.652430435702
            },
            ...
            "31": {
                "ds": "2017-08-03",
                "yhat_lower": -696.973420495606,
                "yhat_upper": 3855.492904015699,
                "yhat": 1615.990963999707
            }
        },
        "NR_test_data_raw": {
            "0": {
                "ds": "2017-07-02",
                "y": 829.3162385321102
            },
            ...
            "31": {
                "ds": "2017-08-03",
                "y": 829.3162385321102
            }
        },
        "NR_outliers": {
            "0": {
                "ds": "2017-02-16",
                "y": 5763.39
            },
            "1": {
                "ds": "2017-03-13",
                "y": 2888.72
            },
            ...
        },
        "Message": "Forecast successful"
    }
}
```

    
    
