import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# def load_data():
#     df = pd.read_csv('dataset/data.csv', sep=';')
#     df_features = df.iloc[:, :36]
#     df_labels = df.iloc[:, 36]
#
#     train_features, val_features, train_labels, val_labels = train_test_split(
#     df_features, df_labels, test_size=0.2, random_state=0
#     )
#
#     # Encode the labels with LabelEncoder
#     label_encoder = LabelEncoder()
#     encoded_train_labels = label_encoder.fit_transform(train_labels)
#     encoded_val_labels = label_encoder.transform(val_labels)
#
#     # Create the MinMaxScaler
#     scaler = MinMaxScaler()
#
#     # Normalize the training set
#     scaler.fit(train_features)
#     train_features_normalized = scaler.transform(train_features)
#
#     # Apply the same transformation to the validation set
#     val_features_normalized = scaler.transform(val_features)
#
#     # Convert to pandas DataFrame
#     train_features = pd.DataFrame(train_features_normalized, columns=train_features.columns)
#     val_features = pd.DataFrame(val_features_normalized, columns=val_features.columns)
#
#     # Converting data to PyTorch tensors
#     train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
#     train_labels_tensor = torch.tensor(encoded_train_labels, dtype=torch.long)
#
#     val_features_tensor = torch.tensor(val_features.values, dtype=torch.float32)
#     val_labels_tensor = torch.tensor(encoded_val_labels, dtype=torch.long)
#
#     # Define the datasets
#     train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
#     val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
#
#     return train_dataset, val_dataset


class CustomDataset(TensorDataset):
    def __init__(self, features, labels):
        super(CustomDataset, self).__init__(features, labels)#new
        self.labels = labels

def load_data():
    df = pd.read_csv('dataset/data.csv', sep=';')
    df_features = df.iloc[0:4410, :36]
    df_labels = df.iloc[0:4410, 36]
    #print(df_features.head)
    train_features, val_features, train_labels, val_labels = train_test_split(
        df_features, df_labels, test_size=0.2, random_state=0
    )

    label_encoder = LabelEncoder()
    encoded_train_labels = label_encoder.fit_transform(train_labels)
    encoded_val_labels = label_encoder.transform(val_labels)

    scaler = MinMaxScaler()
    scaler.fit(train_features)
    train_features_normalized = scaler.transform(train_features)
    val_features_normalized = scaler.transform(val_features)

    train_features_tensor = torch.tensor(train_features_normalized, dtype=torch.float32)
    train_labels_tensor = torch.tensor(encoded_train_labels, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features_normalized, dtype=torch.float32)
    val_labels_tensor = torch.tensor(encoded_val_labels, dtype=torch.long)

    train_dataset = CustomDataset(train_features_tensor, train_labels_tensor)
    val_dataset = CustomDataset(val_features_tensor, val_labels_tensor)

    return train_dataset, val_dataset, label_encoder
train_dataset = load_data()





