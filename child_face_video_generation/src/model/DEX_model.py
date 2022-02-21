from keras.applications import ResNet50
from keras.layers import Dense
from keras.models import Model


def get_DEX_model(weight_file=None):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling="avg")
    prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                       name="pred_age")(base_model.output)
    model = Model(inputs=base_model.input, outputs=prediction)

    if weight_file is not None:
        model.load_weights(weight_file)

    return model
