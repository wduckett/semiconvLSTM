from keras.src import regularizers
from keras.src.layers.rnn.base_conv_lstm import ConvLSTMCell,ConvLSTM
import tensorflow as tf


class SemiConvLSTMCell(ConvLSTMCell):
    def build(self, input_shape):
        output_shape = super(SemiConvLSTMCell,self).build(input_shape)
        # intitialises the hidden state transition weight matrix
        self.recurrent_kernel = self.add_weight(
            shape=(input_shape[1]*input_shape[2]*4, input_shape[1]*input_shape[2]),
            initializer='orthogonal',
            name='recurrent_kernel')
        return output_shape
        
    def call(self, inputs, states):
        # attempted to keep the code as similar to the source code for the ConvLSTM

        # hidden state has seperate width and height dimensions
        h_tm1 = states[0]  

        # stores the shape of the hidden state
        hiddenStateShape = tf.shape(h_tm1)

        # reshape it so it is compatable with the broadcasting of matrix multiplcation
        h_tm1 = tf.reshape(h_tm1, [hiddenStateShape[0],hiddenStateShape[1]*hiddenStateShape[2],self.filters])

        # cell state
        c_tm1 = states[1]

        (kernel_i, kernel_f,
        kernel_c, kernel_o) = tf.split(self.kernel, 4, axis=3)

        bias_i, bias_f, bias_c, bias_o = tf.split(self.bias, 4)

        # calculates the convolution of the input

        x_i = self.input_conv(inputs, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs, kernel_o, bias_o, padding=self.padding)

        # calculates teh matrix multiplaction of the hidden states

        h_i = tf.matmul(self.recurrent_kernel[ : hiddenStateShape[1]*hiddenStateShape[2] , :], h_tm1)
        h_f = tf.matmul(self.recurrent_kernel[ hiddenStateShape[1]*hiddenStateShape[2] : hiddenStateShape[1]*hiddenStateShape[2] *  2, :],h_tm1)
        h_c = tf.matmul(self.recurrent_kernel[ hiddenStateShape[1]*hiddenStateShape[2] * 2 : hiddenStateShape[1]*hiddenStateShape[2] *  3, :],h_tm1)
        h_o = tf.matmul(self.recurrent_kernel[hiddenStateShape[1]*hiddenStateShape[2] * 3 : hiddenStateShape[1]*hiddenStateShape[2] *  4, :],h_tm1)
        

        # reshapes the hidden states so they can be added to the input convolutions
        h_i = tf.reshape(h_i, [hiddenStateShape[0],hiddenStateShape[1],hiddenStateShape[2],self.filters]) 
        h_f = tf.reshape(h_f, [hiddenStateShape[0],hiddenStateShape[1],hiddenStateShape[2],self.filters])
        h_c = tf.reshape(h_c, [hiddenStateShape[0],hiddenStateShape[1],hiddenStateShape[2],self.filters])
        h_o = tf.reshape(h_o, [hiddenStateShape[0],hiddenStateShape[1],hiddenStateShape[2],self.filters])

        # performs the LSTM equations
        # note the biases were already added in during the input convolution

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]

class SemiConvLSTM(ConvLSTM):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        dropout=0.0,
        recurrent_dropout=0.0,
        **kwargs,
        ):

        cell = SemiConvLSTMCell(
                rank=rank,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                recurrent_initializer=recurrent_initializer,
                bias_initializer=bias_initializer,
                unit_forget_bias=unit_forget_bias,
                kernel_regularizer=kernel_regularizer,
                recurrent_regularizer=recurrent_regularizer,
                bias_regularizer=bias_regularizer,
                kernel_constraint=kernel_constraint,
                recurrent_constraint=recurrent_constraint,
                bias_constraint=bias_constraint,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name="conv_lstm_cell",
                dtype=kwargs.get("dtype"),
            )

        
        super(ConvLSTM, self) .__init__(
            rank,
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            **kwargs,
        )

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def get_config(self):
        config = super(SemiConvLSTM, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        print('called from_config')
        return cls(rank = 2, **config)
        
