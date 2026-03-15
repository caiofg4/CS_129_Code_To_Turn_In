""" Cascade Two Convolutions Per Level Network Architecture
    written by Caio Gould based on the OpenKBP one convolution per level network architecture by Babier et al. (2021)
 """
 

from typing import Optional
from typing import Any
from tensorflow.keras.layers import (
    Activation,
    AveragePooling3D,
    Conv3D,
    Conv3DTranspose,
    Input,
    LeakyReLU,
    SpatialDropout3D,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.models import Model
from provided_code.data_shapes import DataShapes

class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def __init__(
        self,
        data_shapes: DataShapes,
        initial_number_of_filters: int,
        filter_size: tuple[int, int, int],
        stride_size: tuple[int, int, int],
        gen_optimizer: Any,
        #gen_optimizer: OptimizerV2,
    ):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer

    def make_convolution_block(self, x: Any, num_filters: int, strides=None, use_batch_norm: bool = True) -> Any:
        if strides is None:
            strides = self.stride_size
        x = Conv3D(num_filters, self.filter_size, strides=strides, padding="same", use_bias=False)(x)
        if use_batch_norm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def make_convolution_transpose_block(
        self, x: Any, num_filters: int, use_dropout: bool = True, skip_x: Optional[Any] = None, strides=None
    ) -> Any:
        if strides is None:
            strides = self.stride_size
        if skip_x is not None:
            x = concatenate([x, skip_x])
        x = Conv3DTranspose(num_filters, self.filter_size, strides=strides, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        if use_dropout:
            x = SpatialDropout3D(0.2)(x)
        x = LeakyReLU(alpha=0)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved
        return x

    def unet_stage(self, x: Any) -> Any:
        x1 = self.make_convolution_block(x, self.initial_number_of_filters)
        x1a = self.make_convolution_block(x1, self.initial_number_of_filters, strides=(1,1,1))
        x2 = self.make_convolution_block(x1a, 2 * self.initial_number_of_filters)
        x2a = self.make_convolution_block(x2, 2 * self.initial_number_of_filters, strides=(1,1,1))
        x3 = self.make_convolution_block(x2a, 4 * self.initial_number_of_filters)
        x3a = self.make_convolution_block(x3, 4 * self.initial_number_of_filters, strides=(1,1,1))
        x4 = self.make_convolution_block(x3a, 8 * self.initial_number_of_filters)
        x4a = self.make_convolution_block(x4, 8 * self.initial_number_of_filters,strides=(1,1,1))
        x5 = self.make_convolution_block(x4a, 16 * self.initial_number_of_filters)
        x5a = self.make_convolution_block(x5, 16 * self.initial_number_of_filters, strides=(1,1,1))
        x6 = self.make_convolution_block(x5a, 32 * self.initial_number_of_filters, use_batch_norm=False)
        x6a = self.make_convolution_block(x6, 32 * self.initial_number_of_filters, use_batch_norm=False, strides=(1,1,1))

        # Build model back up from bottleneck
        x5b = self.make_convolution_transpose_block(x6a, 16 * self.initial_number_of_filters, use_dropout=False)
        x5bc = self.make_convolution_block(x5b, 16 * self.initial_number_of_filters, strides=(1,1,1))
        x4b = self.make_convolution_transpose_block(x5bc, 8 * self.initial_number_of_filters, skip_x=x5a)
        x4bc = self.make_convolution_block(x4b, 8 * self.initial_number_of_filters, strides=(1,1,1))
        x3b = self.make_convolution_transpose_block(x4bc, 4 * self.initial_number_of_filters, use_dropout=False, skip_x=x4a)
        x3bc = self.make_convolution_block(x3b, 4 * self.initial_number_of_filters, strides=(1,1,1))
        x2b = self.make_convolution_transpose_block(x3bc, 2 * self.initial_number_of_filters, skip_x=x3a)
        x2bc = self.make_convolution_block(x2b, 2 * self.initial_number_of_filters, strides=(1,1,1))
        x1b = self.make_convolution_transpose_block(x2bc, self.initial_number_of_filters, use_dropout=False, skip_x=x2a)
        x1bc = self.make_convolution_block(x1b, self.initial_number_of_filters, strides=(1,1,1))

        # Final layer
        x0b = concatenate([x1bc, x1])
        x0b = Conv3DTranspose(1, self.filter_size, strides=self.stride_size, padding="same")(x0b)
        x_final = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b) # may comment out this line seems to run much faster without a final average pooling
        final_dose = Activation("relu")(x_final)
        return final_dose

    def define_generator(self) -> Model:
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        # Define inputs
        ct_image = Input(self.data_shapes.ct)
        roi_masks = Input(self.data_shapes.structure_masks)

        # Build Model starting with Conv3D layers
        x0 = concatenate([ct_image, roi_masks])
        pred1 = self.unet_stage(x0)
        x1 = concatenate([x0, pred1])
        pred2 = self.unet_stage(x1)

        # Compile model for use
        generator = Model(inputs=[ct_image, roi_masks], outputs=pred2, name="generator")
        generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer)
        generator.summary()
        return generator