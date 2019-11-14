from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report,recall_score


batch_size = 32
num_classes = 8
epochs = 10000


def squash(x, axis=-1):
    s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_quared_norm) / (1.0 + s_quared_norm)
    result = scale * x
    return result

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    result = ex / K.sum(ex, axis=axis, keepdims=True)
    return result

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    result = K.sum(y_true * K.square(K.relu(1 - margin - y_pred))
                   + lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
    return result


class Capsule(Layer):

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)  # Capsule继承**kwargs参数
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activation.get(activation)  # 得到激活函数

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        super(Capsule, self).build(input_shape)  # 必须继承Layer的build方法

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                b += K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):  # 自动推断shape
        return (None, self.num_capsule, self.dim_capsule)


def MODEL():
    input_image = Input(shape=(1500, 20, 1))
    x = Conv2D(128, (5, 5), activation='relu')(input_image)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Reshape((-1, 16))(x)  # (None, 100, 128) 相当于前一层胶囊(None, input_num, input_dim)
    capsule = Capsule(num_capsule=8, dim_capsule=16, routings=3, share_weights=True)(x)  # capsule-（None,10, 16)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(capsule)  # 最后输出变成了10个概率值
    model = Model(inputs=input_image, output=output)
    return model

if __name__ == '__main__':
    # 加载数据
    x_train = pd.read_csv('PSSM_X_train', delim_whitespace=True, header=None)
    x_train = x_train.values
    x_train = x_train.reshape(3249,1500,20,1)

    x_test = pd.read_csv('PSSM_X_test', delim_whitespace=True, header=None)
    x_test = x_test.values
    x_test = x_test.reshape(4333, 1500, 20, 1)

    y_train = pd.read_csv('PSSM_y_train', delim_whitespace=True, header=None)
    y_train = y_train.values
    y_train = y_train.reshape(3249,1)

    y_test = pd.read_csv('PSSM_y_test', delim_whitespace=True, header=None)
    y_test = y_test.values
    y_test = y_test.reshape(4333,1)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    from sklearn.cross_validation import train_test_split

    X, X_val, YY, Y_vall = train_test_split(x_train, y_train, test_size=0.2, random_state=20, stratify=y_train)

    Y = utils.to_categorical(YY, num_classes)
    Y_val = utils.to_categorical(Y_vall, num_classes)

    y_test = utils.to_categorical(y_test, num_classes)

    print('propressing finished')

    model = MODEL()
    model.load_weights('weights.best_5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = model.predict(X_val)

    label = np.array([np.argmax(predict[i]) for i in range(len(Y_vall))])
    label2 = np.array([int(Y_vall[i][0]) for i in range(len(Y_vall))])

    print(np.sum(label == label2)/len(Y_vall))

    predict = model.predict(x_test)

    label = np.array([np.argmax(predict[i]) for i in range(len(y_test))])
    y_test__ = np.loadtxt('PSSM_y_test')

    print(np.sum(label == y_test__) / len(y_test__))
    print(recall_score(y_test__, label, average=None))
    print(classification_report(y_test__, label))
