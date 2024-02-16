
## Digit_Recognition

A repository for recognizing digits using neural network built on this [kaggle dataset.](https://www.kaggle.com/competitions/digit-recognizer/) The neural network takes multiple (42000 as in this dataset) 784 sized vectors as input. It using forward propagation creates a 10 sized first intermediate layer. It uses randomly assigned weights and biases of appropriate sizes to construct this first layer. Similar process is used to create the send layer which gives the predict number. Initially this accuracy is low as weights and biases are randomly assigned. It uses backpropagation to adjust these values.

After 500 iterations using I got 83.8% as accuracy.



## Run Locally

Clone the project

```bash
  git clone https://github.com/ap5967ap/Number_Recog.git
```

Go to the project directory

```bash
  cd Number_Recog
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run script

```bash
  python ./num.py
```

## Dataset
The Dataset is available [here](https://www.kaggle.com/competitions/digit-recognizer). I have also added [here](https://github.com/ap5967ap/Number_Recog/blob/main/train.csv) for convinence
