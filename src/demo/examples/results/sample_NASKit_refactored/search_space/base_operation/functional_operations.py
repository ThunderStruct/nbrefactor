import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignConcat:
    """
    AlignConcat is a custom multiplexing functional module used to aggregate
    multiple inputs. Aligns the inputs' resolutions by upsampling all inputs to
    the largest resolution (aligning through downsampling results in a loss of
    information).
    """

    @staticmethod
    def functional(inputs: List[torch.Tensor]):
        """
        Functional forward pass

        Args:
            inputs (list): the operation's input data vector of \
            :class:`torch.Tensor` objects

        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        # infer spatial dimensions
        max_height = max([in_tensor.shape[2] for in_tensor in inputs])
        max_width = max([in_tensor.shape[3] for in_tensor in inputs])

        # align resolutions (upsample deprecated, using interpolate instead)
        # aligned_inputs = [F.upsample(input=input_tensor, size=(max_height,
        #                                    max_width)) \
        #                   for input_tensor in inputs]

        # align resolutions; spatial dimensions are altered, channel dimension
        # is unaffected
        # aligned_inputs = [F.interpolate(in_tensor,
        #                                 size=(max_height, max_width),
        #                                 mode='bilinear',
        #                                 align_corners=False) \
        #                   for in_tensor in inputs]

        aligned_inputs = []
        for tensor in inputs:
            if tensor.shape[2] != max_height or tensor.shape[3] != max_width:
                upsampled_tensor = F.interpolate(tensor,
                                                 size=(max_height, max_width),
                                                 mode='bilinear',
                                                 align_corners=False)
                aligned_inputs.append(upsampled_tensor)
            else:
                aligned_inputs.append(tensor)


        # stack inputs w.r.t. the channels
        out = torch.cat(aligned_inputs, dim=1)

        return out


    @staticmethod
    def compute_shape(in_shapes):
        """
        Gets partial functions with all valid hyperparameter combinations
        given the input shape.

        Uses lazy initialization over
        :func:`~BaseOperation.get_all_partials` for faster performance
        (the cache uses minimal memory as these are uninitialized partials)

        Args:
            in_shapes (list): list of :class:`torch.Tensor` objects, each \
            corressponding to a 4-dim shape of an input feature map (NxCxHxW)

        Returns:
            list: a 1-element list of partial functions with initialized valid \
            hyperparameters (in the form of a list to conform with primitives)
        """

        intermediary_shape = (
            max([o[0] for o in in_shapes]),
            sum([o[1] for o in in_shapes]),
            max([o[2] for o in in_shapes]),
            max([o[3] for o in in_shapes])
        )

        return intermediary_shape


class ShapeMatchTransform:
    """
    This Transform functional module takes an aggregated input tensor and
    transforms its shape to another module's expected shape.

    It is primarily applied to upon network extensions as it is difficult to
    constrain new paths with overlapping nodes to a specific shape.
    """

    @staticmethod
    def functional(input: torch.Tensor, shape):
        """
        Functional forward pass

        Args:
            input (:class:`torch.Tensor`): the operation's input data
            shape (:class:`np.ndarray`): the expected input shape
        Returns:
            :class:`torch.Tensor`: the operation's output data
        """

        # unpack
        _, c, h, w = shape
        x = input

        # use adaptive_avg_pool2d for downsampling & interpolate for upsampling
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, size=(h, w),
                              mode='bilinear', align_corners=False)

        # adapt channel dimension using a (somewhat) functional transform
        if x.size(1) != c:
            channel_transform = nn.Conv2d(in_channels=x.size(1),
                                          out_channels=c,
                                          kernel_size=(1, 1))

            # move the channel_transform parameters to the same device as x
            channel_transform.to(x.device)

            x = channel_transform(x)

        return x


