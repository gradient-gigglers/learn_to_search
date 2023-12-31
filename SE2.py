import torch
import torch.nn as nn
import torch.optim as optim
from random import choice, randint
import numpy as np


#define BILSTM model
class BiLSTMWithPooling(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,num_layers):
        super(BiLSTMWithPooling, self).__init__()

        self.bilstm = nn.LSTM(input_size, hidden_dim,output_size, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim,output_size)  # 2 * hidden_dim due to bidirectional

    def forward(self, x):

        #flatten into 1d
        shape = x.shape
        num_tokens = shape[0]
        token_dim = shape[1]
        view_num = num_tokens*token_dim
        x = x.view(1,1,view_num)
        # BiLSTM
        out, _ = self.bilstm(x)
        # Fully connected layer
        output = self.fc(out)
        return output
    

class BiLSTMWithPoolingD(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size,num_layers):
        super(BiLSTMWithPoolingD, self).__init__()
    
        self.bilstm = nn.LSTM(input_size, hidden_dim,output_size, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim,output_size)  # 2 * hidden_dim due to bidirectional

    def forward(self, x):
        x = x[0]
        shape = x.shape
        num_tokens = shape[0]
        token_dim = shape[1]
        view_num = num_tokens*token_dim
        x = x.view(1,1,view_num)
        
        # BiLSTM
        out, _ = self.bilstm(x)
        
        # Fully connected layer
        output = self.fc(out)
        return output
    

# Define model hyperparameters
hidden_dim = 64
input_size = 1000
num_layers = 1  # You can adjust the number of layers 
output_size = 200



############## SIAMESE MODEL ################
# Create two instances of the BiLSTM model (shared weights)
query_model = BiLSTMWithPooling(input_size, hidden_dim, output_size,num_layers,)
doc_model = BiLSTMWithPoolingD(input_size, hidden_dim, output_size, num_layers)

# Ensure weight sharing by setting model_right's weights to be the same as model_left's
query_model.load_state_dict(doc_model.state_dict())

###### TRIPLET LOSS IN BATCHES##########

# Training loop

query_list = [[1,[[ 0.6805, -0.0393,  0.3019, -0.1779,  0.4296,  0.0322, -0.4138,  0.1323,
         -0.2985, -0.0853,  0.1712,  0.2242, -0.1005, -0.4365,  0.3342,  0.6785,
          0.0572, -0.3445, -0.4279, -0.4327,  0.5596,  0.1003,  0.1868, -0.2685,
          0.0373, -2.0932,  0.2217, -0.3987,  0.2091, -0.5573,  3.8826,  0.4747,
         -0.9566, -0.3779,  0.2087, -0.3275,  0.1275,  0.0884,  0.1635, -0.2163,
         -0.0944,  0.0183,  0.2105, -0.0309, -0.1972,  0.0823, -0.0943, -0.0733,
         -0.0647, -0.2604],
        [ 1.2426,  0.5691,  1.0025,  0.1335,  0.0125, -0.2616, -0.6472, -0.3676,
         -0.0151,  0.2838, -0.2020, -0.1134, -0.0707, -0.0542, -0.7028,  0.7744,
          0.9624,  1.1434,  0.2568, -0.4468,  0.5363, -0.2846, -1.2776, -0.6634,
          0.0268, -1.1640,  0.2159, -0.3651,  0.6426, -0.4392,  3.3075,  0.3918,
         -0.6449, -0.2410,  0.0383,  0.3309, -0.0507,  0.5227, -0.2849, -0.2076,
         -0.2032, -0.9633,  0.3911, -0.0282,  0.1736,  0.5589, -0.5781, -0.1528,
         -0.6170, -0.1992],
        [ 0.2171,  0.4651, -0.4676,  0.1008,  1.0135,  0.7484, -0.5310, -0.2626,
          0.1681,  0.1318, -0.2491, -0.4419, -0.2174,  0.5100,  0.1345, -0.4314,
         -0.0312,  0.2067, -0.7814, -0.2015, -0.0974,  0.1609, -0.6184, -0.1850,
         -0.1246, -2.2526, -0.2232,  0.5043,  0.3226,  0.1531,  3.9636, -0.7136,
         -0.6701,  0.2839,  0.2174,  0.1443,  0.2593,  0.2343,  0.4274, -0.4445,
          0.1381,  0.3697, -0.6429,  0.0241, -0.0393, -0.2604,  0.1202, -0.0438,
          0.4101,  0.1796],
        [ 1.1461,  0.2718, -0.3112, -0.2363,  0.1209,  0.1616,  0.1589, -0.6279,
          0.5812, -0.0056, -0.6346,  0.4295, -0.3480,  0.8370,  0.6419, -0.1399,
         -0.5931,  0.9124, -0.3105,  0.8796,  0.3603,  0.0881, -0.6136,  0.5556,
         -0.1334,  0.4246, -1.2651,  0.1345,  0.5429, -0.1470, -0.7003,  0.4941,
          0.4209,  0.3181, -0.3124,  0.9501,  0.5753, -0.7458, -0.3967, -0.5260,
         -0.0413, -0.5993, -0.4431,  0.1954, -0.2038,  0.9744,  0.3118, -0.7737,
         -0.3760,  0.3618],
        [ 0.3304,  0.2500, -0.6087,  0.1092,  0.0364,  0.1510, -0.5508, -0.0742,
         -0.0923, -0.3282,  0.0960, -0.8227, -0.3672, -0.6701,  0.4291,  0.0165,
         -0.2357,  0.1286, -1.0953,  0.4333,  0.5707, -0.1036,  0.2042,  0.0783,
         -0.4279, -1.7984, -0.2786,  0.1195, -0.1269,  0.0317,  3.8631, -0.1779,
         -0.0824, -0.6270,  0.2650, -0.0572, -0.0735,  0.4610,  0.3086,  0.1250,
         -0.4861, -0.0080,  0.0312, -0.3658, -0.4270,  0.4216, -0.1167, -0.5070,
         -0.0273, -0.5329],
        [-0.0292,  0.8177,  0.3847, -0.7786,  1.1049, -0.1365, -0.0247, -0.0511,
          0.7795,  0.0514, -0.3575,  1.1748, -0.0982,  0.3311,  0.4043,  0.5868,
         -0.6254,  0.0948,  0.9702, -1.1437,  0.1383,  0.2814,  0.4669,  0.3523,
          0.6892, -1.9819, -1.4000,  0.1700,  1.5929, -1.0086,  3.6499,  1.3949,
         -0.7882,  0.4040, -0.3692,  0.7308,  0.0275, -0.1199,  0.7372, -1.0365,
          0.6866, -0.3029, -0.5518,  0.9647,  0.0531, -0.0848,  0.8512, -0.5419,
          0.3245,  0.5842]]],[2,[[ 0.6805, -0.0393,  0.3019, -0.1779,  0.4296,  0.0322, -0.4138,  0.1323,
         -0.2985, -0.0853,  0.1712,  0.2242, -0.1005, -0.4365,  0.3342,  0.6785,
          0.0572, -0.3445, -0.4279, -0.4327,  0.5596,  0.1003,  0.1868, -0.2685,
          0.0373, -2.0932,  0.2217, -0.3987,  0.2091, -0.5573,  3.8826,  0.4747,
         -0.9566, -0.3779,  0.2087, -0.3275,  0.1275,  0.0884,  0.1635, -0.2163,
         -0.0944,  0.0183,  0.2105, -0.0309, -0.1972,  0.0823, -0.0943, -0.0733,
         -0.0647, -0.2604],
        [ 0.2563, -0.0810, -0.2280, -0.2033,  0.3404,  0.0993, -0.6157,  0.1808,
         -0.1497,  0.1343,  0.4677,  0.3300, -0.1093,  0.5714,  0.2291,  0.5734,
          0.2596,  0.5906,  0.1831,  0.0414, -0.5339, -0.3595, -0.1096,  0.8020,
          0.0414, -1.8999, -0.5605, -0.1052,  0.3480,  0.6501,  3.2213, -0.0656,
          0.3535, -0.9390, -0.1355,  0.0454, -0.4506,  0.5396, -0.5322, -0.8970,
         -0.7165, -0.1155, -0.0952,  0.5525,  0.0367,  0.2905, -0.7533, -0.6383,
         -0.0719,  0.3016],
        [ 0.2171,  0.4651, -0.4676,  0.1008,  1.0135,  0.7484, -0.5310, -0.2626,
          0.1681,  0.1318, -0.2491, -0.4419, -0.2174,  0.5100,  0.1345, -0.4314,
         -0.0312,  0.2067, -0.7814, -0.2015, -0.0974,  0.1609, -0.6184, -0.1850,
         -0.1246, -2.2526, -0.2232,  0.5043,  0.3226,  0.1531,  3.9636, -0.7136,
         -0.6701,  0.2839,  0.2174,  0.1443,  0.2593,  0.2343,  0.4274, -0.4445,
          0.1381,  0.3697, -0.6429,  0.0241, -0.0393, -0.2604,  0.1202, -0.0438,
          0.4101,  0.1796],
        [-0.1417, -0.0821,  1.6416,  0.8134, -0.4922, -0.2083, -0.7185, -1.5552,
          0.7407,  1.1502,  1.0177,  0.0877,  0.1708,  0.7917, -0.4383,  0.8377,
         -1.0522,  0.1801,  0.1155, -0.2223,  1.3456,  0.1979, -0.2275,  0.4347,
          0.0160, -0.3821, -0.9619, -0.7869,  0.3866, -0.6203,  3.0104, -0.2978,
          0.1819, -1.0438, -0.4410,  0.3104,  0.1452,  0.0726, -0.5890, -0.7485,
          1.4246,  0.0995, -0.3702,  0.1325, -0.9582,  0.3370,  0.8876, -0.1076,
         -0.6597,  0.1977]]]]

doc_list = [[1,[[ 4.1800e-01,  2.4968e-01, -4.1242e-01,  1.2170e-01,  3.4527e-01,
         -4.4457e-02, -4.9688e-01, -1.7862e-01, -6.6023e-04, -6.5660e-01,
          2.7843e-01, -1.4767e-01, -5.5677e-01,  1.4658e-01, -9.5095e-03,
          1.1658e-02,  1.0204e-01, -1.2792e-01, -8.4430e-01, -1.2181e-01,
         -1.6801e-02, -3.3279e-01, -1.5520e-01, -2.3131e-01, -1.9181e-01,
         -1.8823e+00, -7.6746e-01,  9.9051e-02, -4.2125e-01, -1.9526e-01,
          4.0071e+00, -1.8594e-01, -5.2287e-01, -3.1681e-01,  5.9213e-04,
          7.4449e-03,  1.7778e-01, -1.5897e-01,  1.2041e-02, -5.4223e-02,
         -2.9871e-01, -1.5749e-01, -3.4758e-01, -4.5637e-02, -4.4251e-01,
          1.8785e-01,  2.7849e-03, -1.8411e-01, -1.1514e-01, -7.8581e-01],
        [-3.1905e-01, -9.5070e-02, -4.9458e-02, -4.5461e-01,  8.1447e-01,
          5.5730e-01, -3.7148e-01,  2.2142e-01, -1.8506e-01, -5.8731e-02,
          1.9556e-01,  2.5934e-01, -9.2544e-01,  6.0346e-01, -1.1877e-01,
          3.3422e-01,  5.8610e-01, -9.3036e-01,  5.7848e-02, -3.7820e-01,
         -1.1279e+00,  1.1801e-01, -2.6057e-01,  4.3562e-02,  2.5741e-01,
         -2.2502e+00,  1.4761e-01,  4.3313e-01,  6.9410e-01, -9.4411e-01,
          3.5944e+00,  4.7921e-01, -6.1145e-01,  1.6474e-01, -6.8397e-01,
         -9.1051e-02,  3.9399e-01,  1.7690e-01,  3.3553e-01, -1.3054e-01,
          1.2659e-01,  2.6288e-01, -4.7432e-01,  7.9129e-01, -7.5714e-01,
         -1.1700e-01, -9.3022e-02, -2.3004e-01,  8.1364e-02, -3.5147e-02],
        [ 6.2231e-01,  1.1986e+00, -1.4116e-02,  2.0125e-01,  6.9419e-01,
          1.2068e-01, -9.0399e-01, -1.4023e+00,  4.3357e-01, -4.8537e-01,
         -4.6450e-01,  1.5756e-01,  5.4261e-01, -3.2467e-01, -2.5646e-02,
          4.5742e-01,  1.6561e-01,  1.8819e-01,  6.2099e-02, -8.6418e-01,
         -1.0425e+00, -8.1157e-01,  3.1260e-01, -2.0279e-01,  5.5734e-01,
         -2.8634e-01, -1.4874e-01,  1.0098e+00,  2.5041e-01, -5.3195e-01,
          2.3793e+00, -7.6966e-01, -6.3219e-01,  3.2030e-01,  1.5072e-01,
          2.3326e-01, -2.6254e-01, -2.9461e-01,  7.6710e-01, -1.1577e-01,
         -6.8129e-01, -6.5413e-01, -5.8914e-01,  2.4684e-01,  1.5904e+00,
          3.3025e-01,  4.1513e-01, -1.7468e+00,  8.2453e-01, -1.0886e+00],
        [ 1.5272e-01,  3.6181e-01, -2.2168e-01,  6.6051e-02,  1.3029e-01,
          3.7075e-01, -7.5874e-01, -4.4722e-01,  2.2563e-01,  1.0208e-01,
          5.4225e-02,  1.3494e-01, -4.3052e-01, -2.1340e-01,  5.6139e-01,
         -2.1445e-01,  7.7974e-02,  1.0137e-01, -5.1306e-01, -4.0295e-01,
          4.0639e-01,  2.3309e-01,  2.0696e-01, -1.2668e-01, -5.0634e-01,
         -1.7131e+00,  7.7183e-02, -3.9138e-01, -1.0594e-01, -2.3743e-01,
          3.9552e+00,  6.6596e-01, -6.1841e-01, -3.2680e-01,  3.7021e-01,
          2.5764e-01,  3.8977e-01,  2.7121e-01,  4.3024e-02, -3.4322e-01,
          2.0339e-02,  2.1420e-01,  4.4097e-02,  1.4003e-01, -2.0079e-01,
          7.4794e-02, -3.6076e-01,  4.3382e-01, -8.4617e-02,  1.2140e-01],
        [-2.9163e-02,  8.1769e-01,  3.8470e-01, -7.7857e-01,  1.1049e+00,
         -1.3655e-01, -2.4691e-02, -5.1103e-02,  7.7950e-01,  5.1357e-02,
         -3.5748e-01,  1.1748e+00, -9.8244e-02,  3.3111e-01,  4.0426e-01,
          5.8685e-01, -6.2536e-01,  9.4833e-02,  9.7024e-01, -1.1437e+00,
          1.3826e-01,  2.8136e-01,  4.6693e-01,  3.5226e-01,  6.8916e-01,
         -1.9819e+00, -1.4000e+00,  1.7001e-01,  1.5929e+00, -1.0086e+00,
          3.6499e+00,  1.3949e+00, -7.8823e-01,  4.0404e-01, -3.6925e-01,
          7.3075e-01,  2.7513e-02, -1.1993e-01,  7.3716e-01, -1.0365e+00,
          6.8659e-01, -3.0294e-01, -5.5175e-01,  9.6466e-01,  5.3103e-02,
         -8.4807e-02,  8.5120e-01, -5.4186e-01,  3.2453e-01,  5.8425e-01],
        [ 1.1461e+00,  2.7184e-01, -3.1119e-01, -2.3631e-01,  1.2092e-01,
          1.6162e-01,  1.5892e-01, -6.2790e-01,  5.8122e-01, -5.5722e-03,
         -6.3458e-01,  4.2947e-01, -3.4805e-01,  8.3695e-01,  6.4189e-01,
         -1.3991e-01, -5.9314e-01,  9.1242e-01, -3.1046e-01,  8.7961e-01,
          3.6029e-01,  8.8143e-02, -6.1356e-01,  5.5563e-01, -1.3343e-01,
          4.2461e-01, -1.2651e+00,  1.3451e-01,  5.4289e-01, -1.4698e-01,
         -7.0035e-01,  4.9410e-01,  4.2095e-01,  3.1814e-01, -3.1238e-01,
          9.5009e-01,  5.7531e-01, -7.4582e-01, -3.9667e-01, -5.2604e-01,
         -4.1295e-02, -5.9930e-01, -4.4310e-01,  1.9535e-01, -2.0376e-01,
          9.7438e-01,  3.1176e-01, -7.7370e-01, -3.7603e-01,  3.6181e-01],
        [ 6.1850e-01,  6.4254e-01, -4.6552e-01,  3.7570e-01,  7.4838e-01,
          5.3739e-01,  2.2239e-03, -6.0577e-01,  2.6408e-01,  1.1703e-01,
          4.3722e-01,  2.0092e-01, -5.7859e-02, -3.4589e-01,  2.1664e-01,
          5.8573e-01,  5.3919e-01,  6.9490e-01, -1.5618e-01,  5.5830e-02,
         -6.0515e-01, -2.8997e-01, -2.5594e-02,  5.5593e-01,  2.5356e-01,
         -1.9612e+00, -5.1381e-01,  6.9096e-01,  6.6246e-02, -5.4224e-02,
          3.7871e+00, -7.7403e-01, -1.2689e-01, -5.1465e-01,  6.6705e-02,
         -3.2933e-01,  1.3483e-01,  1.9049e-01,  1.3812e-01, -2.1503e-01,
         -1.6573e-02,  3.1200e-01, -3.3189e-01, -2.6001e-02, -3.8203e-01,
          1.9403e-01, -1.2466e-01, -2.7557e-01,  3.0899e-01,  4.8497e-01],
        [ 4.1800e-01,  2.4968e-01, -4.1242e-01,  1.2170e-01,  3.4527e-01,
         -4.4457e-02, -4.9688e-01, -1.7862e-01, -6.6023e-04, -6.5660e-01,
          2.7843e-01, -1.4767e-01, -5.5677e-01,  1.4658e-01, -9.5095e-03,
          1.1658e-02,  1.0204e-01, -1.2792e-01, -8.4430e-01, -1.2181e-01,
         -1.6801e-02, -3.3279e-01, -1.5520e-01, -2.3131e-01, -1.9181e-01,
         -1.8823e+00, -7.6746e-01,  9.9051e-02, -4.2125e-01, -1.9526e-01,
          4.0071e+00, -1.8594e-01, -5.2287e-01, -3.1681e-01,  5.9213e-04,
          7.4449e-03,  1.7778e-01, -1.5897e-01,  1.2041e-02, -5.4223e-02,
         -2.9871e-01, -1.5749e-01, -3.4758e-01, -4.5637e-02, -4.4251e-01,
          1.8785e-01,  2.7849e-03, -1.8411e-01, -1.1514e-01, -7.8581e-01],
        [-1.4168e-01,  4.1108e-01, -3.1227e-01,  1.6633e-01,  2.6124e-01,
          4.5708e-01, -1.2001e+00,  1.4923e-02, -2.2779e-01, -1.6937e-01,
          3.4633e-01, -1.2419e-01, -6.5711e-01,  2.9226e-01,  6.2407e-01,
         -5.7916e-01, -3.3947e-01, -2.2046e-01, -1.4832e+00,  2.8958e-01,
          8.1396e-02, -2.1696e-01,  5.6613e-03, -5.4199e-02,  9.8504e-02,
         -1.5874e+00, -2.2867e-01, -6.2957e-01, -3.9542e-01, -8.0841e-02,
          3.5949e+00, -1.6872e-01, -3.9024e-01,  2.6912e-02,  5.2646e-01,
         -2.2844e-02,  6.3289e-01,  6.2702e-01, -2.2171e-01, -4.5045e-01,
         -1.4998e-01, -2.7723e-01, -4.6658e-01, -4.4268e-01, -4.3691e-01,
          3.8455e-01,  1.3690e-01, -2.5424e-01,  1.7821e-02, -1.4890e-01]]],
          [2,[[ 1.1823, -0.7009, -0.7532, -0.7419, -0.6719, -0.8703,  0.9353,  0.1455,
          0.8748, -0.5091, -0.0330,  0.3654,  0.5351, -0.6585,  0.1516,  0.6132,
         -0.2783, -0.6558, -0.0136, -0.9032, -0.1620,  0.2261, -0.0985, -0.5572,
          0.3720, -0.6295, -0.0518,  0.1855,  0.9656,  0.0846,  1.3024,  1.4936,
         -0.4172, -1.0918,  0.2510,  0.7594,  1.2611, -0.1826, -0.5669,  0.6394,
         -0.3957,  0.1741,  1.0844,  0.7240,  1.0084,  0.3140, -0.1674,  0.8532,
          0.8046,  0.5020],
        [ 0.1527,  0.3618, -0.2217,  0.0661,  0.1303,  0.3708, -0.7587, -0.4472,
          0.2256,  0.1021,  0.0542,  0.1349, -0.4305, -0.2134,  0.5614, -0.2145,
          0.0780,  0.1014, -0.5131, -0.4029,  0.4064,  0.2331,  0.2070, -0.1267,
         -0.5063, -1.7131,  0.0772, -0.3914, -0.1059, -0.2374,  3.9552,  0.6660,
         -0.6184, -0.3268,  0.3702,  0.2576,  0.3898,  0.2712,  0.0430, -0.3432,
          0.0203,  0.2142,  0.0441,  0.1400, -0.2008,  0.0748, -0.3608,  0.4338,
         -0.0846,  0.1214],
        [ 0.2171,  0.4651, -0.4676,  0.1008,  1.0135,  0.7484, -0.5310, -0.2626,
          0.1681,  0.1318, -0.2491, -0.4419, -0.2174,  0.5100,  0.1345, -0.4314,
         -0.0312,  0.2067, -0.7814, -0.2015, -0.0974,  0.1609, -0.6184, -0.1850,
         -0.1246, -2.2526, -0.2232,  0.5043,  0.3226,  0.1531,  3.9636, -0.7136,
         -0.6701,  0.2839,  0.2174,  0.1443,  0.2593,  0.2343,  0.4274, -0.4445,
          0.1381,  0.3697, -0.6429,  0.0241, -0.0393, -0.2604,  0.1202, -0.0438,
          0.4101,  0.1796],
        [-0.3243,  0.1686,  0.1212, -0.6778,  0.8112,  0.5839, -0.0360, -0.6226,
         -0.2246, -0.1755, -0.1513,  0.1261,  0.1847,  0.7260, -0.8578,  0.0833,
          0.4335,  0.7243,  0.7904, -1.2609, -0.8457, -0.0887, -0.8413, -0.5038,
          0.1706, -0.0152, -0.0997,  0.9277,  0.8889, -0.0264,  1.3868, -0.1678,
          0.4732,  0.6806,  0.2699,  0.8339,  0.0392,  0.9135, -0.2626, -0.6394,
          0.2340, -0.5282,  0.4160,  0.3609,  0.5462,  0.0737, -0.1454, -0.6721,
         -0.1689, -0.2377],
        [ 0.2682,  0.1435, -0.2788,  0.0163,  0.1138,  0.6992, -0.5133, -0.4737,
         -0.3307, -0.1383,  0.2702,  0.3094, -0.4501, -0.4127, -0.0993,  0.0381,
          0.0297,  0.1008, -0.2506, -0.5182,  0.3456,  0.4492,  0.4879, -0.0809,
         -0.1012, -1.3777, -0.1087, -0.2320,  0.0128, -0.4651,  3.8463,  0.3136,
          0.1364, -0.5224,  0.3302,  0.3371, -0.3560,  0.3243,  0.1204,  0.3512,
         -0.0690,  0.3688,  0.2517, -0.2452,  0.2538,  0.1367, -0.3118, -0.6321,
         -0.2503, -0.3810],
        [ 0.7651,  0.0310,  0.8590,  0.2947,  0.4552, -0.5741, -0.2548,  0.8755,
          0.5904, -0.0606, -0.0170, -0.1191, -0.0834,  0.0460, -0.4584,  0.4660,
          0.6944, -0.5100,  0.8268, -0.1514,  0.2568,  0.0601, -0.6651,  0.0622,
         -0.2493, -0.9901,  0.4785, -0.6103,  0.6996,  0.0620,  2.8677,  0.2979,
         -0.7999, -0.3917,  0.3242,  0.5239,  0.0399,  0.5043, -0.6622, -1.0425,
          0.5781, -0.5732,  0.3397,  0.0401, -0.6110, -0.0579, -0.3795,  0.2696,
          0.1326, -0.1822],
        [ 1.1461,  0.2718, -0.3112, -0.2363,  0.1209,  0.1616,  0.1589, -0.6279,
          0.5812, -0.0056, -0.6346,  0.4295, -0.3480,  0.8370,  0.6419, -0.1399,
         -0.5931,  0.9124, -0.3105,  0.8796,  0.3603,  0.0881, -0.6136,  0.5556,
         -0.1334,  0.4246, -1.2651,  0.1345,  0.5429, -0.1470, -0.7003,  0.4941,
          0.4209,  0.3181, -0.3124,  0.9501,  0.5753, -0.7458, -0.3967, -0.5260,
         -0.0413, -0.5993, -0.4431,  0.1954, -0.2038,  0.9744,  0.3118, -0.7737,
         -0.3760,  0.3618]]]]

 
for item in query_list:
    item[1] = torch.FloatTensor(item[1])

for item in doc_list:
    item[1] = torch.FloatTensor(item[1])


#############TRAINING CYCLE################

num_epochs = 2
for epoch in range(num_epochs): 

    #instantiate empty vectors 
    anchors = []
    positives = []
    negatives = []

    #select positive and negative document samples for training given specific query
    for anchor in query_list:

        target_index = anchor[0]

        #fill anchor tensor with query embedding
       
        anchors = anchor[1]
    

        #find positive examples
        while len(positives) < 1:
            for document in doc_list:
                if document[0] == target_index:
                    positives.append(document[1])
                    

        #find negative examples
        while len(negatives) < 1:

            rand_index = randint(0, len(query_list)-1)
            #choice(i for i in range(0,len(query_list)))
            if doc_list[rand_index][0]!= target_index:
                negatives.append(doc_list[rand_index][1])

       

    #calculate input size for anchor
        anchor_shape = anchors.shape
        anchor_num_tokens = anchor_shape[0]
        anchor_token_dim = anchor_shape[1]
        input_size= anchor_num_tokens*anchor_token_dim
      
        embed_anchors = query_model(anchors)

        #calculate input size for +ve docs
        pos_shape = positives[0].shape
        pos_num_tokens = pos_shape[0]
        pos_token_dim = pos_shape[1]
        input_size= pos_num_tokens*pos_token_dim
      
        embed_positives = doc_model(positives)

        #calculate input size for -ve docs
        neg_shape = negatives[0].shape
        neg_num_tokens = neg_shape[0]
        neg_token_dim = neg_shape[1]
        input_size= neg_num_tokens*neg_token_dim
      
        embed_negatives = doc_model(negatives)

        #calculate triplet loss for batch of samples
        loss_fn_triplet = nn.TripletMarginLoss(margin=0.2, p=2) 
        triplet_loss = loss_fn_triplet(embed_anchors,embed_positives,embed_negatives)


        #Perform backpropagation and optimization
        optimizer = optim.Adam(query_model.parameters(), lr=0.01)
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        # Print the triplet loss to monitor training progress
        print("Triplet Loss:", triplet_loss.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {triplet_loss.item()}')
