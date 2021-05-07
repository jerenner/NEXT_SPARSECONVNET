
import torch
import sparseconvnet as scn
from .building_blocks import ResidualBlock_downsample, ResidualBlock_basic, ResidualBlock_upsample, ConvBNBlock, calculate_output_dimension

class UNet(torch.nn.Module):
    '''
        This class implements a UNet structure, built with ResNet blocks. It takes a tuple of (coordinates, features)
        and passes it through the UNet

        ...

        Attributes
        ----------
        spatial_size : tuple
            The spatial size of the input layer. Size of the tuple is also the dimension.
        init_conv_nplaness : int
            Number of planes we want after the initial SubmanifoldConvolution, that is, to begin downsampling.
        init_conv_kernel : int
            Kernel for the first convolutional layer.
        kernel_sizes : list
            Sizes of the kernels applied in each layer or block. Its length also indicates the level depth of the UNet.
            Last element corresponds just to the kernel of the basic block in the bottom of the UNet.
        stride_sizes : list
            Sizes of the strides applied in each downsample/upsample.
        basic_num : int
            Number of times a basic residual block is passed in each level.
        nclasses : int, optional
            Number of output classes for predictions. The default is 3.
        dim : int, optional
            Number of dimensions of the input. The default is 3.
        start_planes : int, optional
            Number of planes that enter the UNet. The default is 1.

        Methods
        -------
        forward(x)
            Passes the input through the UNet
     '''

    def __init__(self, spatial_size, init_conv_nplanes, init_conv_kernel, kernel_sizes, stride_sizes, basic_num, nclasses = 3, dim = 3, start_planes = 1):
        '''
            Parameters
            ----------
            spatial_size : tuple
                The spatial size of the input layer. Size of the tuple is also the dimension.
            init_conv_nplaness : int
                Number of planes we want after the initial SubmanifoldConvolution, that is, to begin downsampling.
            init_conv_kernel : int
                Kernel for the first convolutional layer.
            kernel_sizes : list
                Sizes of the kernels applied in each layer or block. Its length also indicates the level depth of the UNet.
                Last element corresponds just to the kernel of the basic block in the bottom of the UNet.
            stride_sizes : list
                Sizes of the strides applied in each downsample/upsample.
            basic_num : int
                Number of times a basic residual block is passed in each level.
            nclasses : int, optional
                Number of output classes for predictions. The default is 3.
            dim : int, optional
                Number of dimensions of the input. The default is 3.
            start_planes : int, optional
                Number of planes that enter the UNet. The default is 1.
        '''

        bottom_spatial_size = calculate_output_dimension(spatial_size, kernel_sizes, stride_sizes)

        torch.nn.Module.__init__(self)

        self.basic_num   = basic_num
        self.level_depth = len(kernel_sizes)

        #Initial layers
        self.inp     = scn.InputLayer(dim, spatial_size)
        self.convBN  = ConvBNBlock(start_planes, init_conv_nplanes, init_conv_kernel)
        inplanes = init_conv_nplanes

        #Final layers
        self.output = scn.OutputLayer(dim)
        self.linear = torch.nn.Linear(inplanes, nclasses)

        #Branch layers
        self.downsample = torch.nn.ModuleList([])
        self.upsample   = torch.nn.ModuleList([])
        self.basic_down = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(self.level_depth - 1)])
        self.bottom     = torch.nn.ModuleList([])
        self.basic_up   = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(self.level_depth - 1)])

        for i in range(self.level_depth - 1):
            for j in range(basic_num):
                self.basic_down[i].append(ResidualBlock_basic(inplanes, kernel_sizes[i])) #basic blocks for downsample branch
                self.basic_up[i].append(ResidualBlock_basic(inplanes, kernel_sizes[i])) #basic blocks for upsample branch
            self.downsample.append(ResidualBlock_downsample(inplanes, kernel_sizes[i], stride_sizes[i])) #downsamples
            self.upsample.append(ResidualBlock_upsample(2 * inplanes, kernel_sizes[i], stride_sizes[i])) #upsamples, backwards

            inplanes = inplanes * 2

        #Bottom layer
        for j in range(basic_num):
            self.bottom.append(ResidualBlock_basic(inplanes, kernel_sizes[-1])) #basic blocks for the lowest layer, kernel is the last element in the list

        self.add = scn.AddTable()

    def forward(self, x):
        '''
        Passes x through the UNet.

        Parameters
        ----------
        x : tuple
            It takes a tuple with (coord, features), where coord is a torch tensor with size [features_number, batch_size],
            and features is another torch tensor, with size [features_number, start_planes]

        Returns
        -------
        x : torch.Tensor
            A tensor with size [features_number, nclasses].

        '''
        x = self.inp(x)
        x = self.convBN(x)

        tmp_layers = []

        #downsample
        for i in range(self.level_depth - 1):
            for j in range(self.basic_num):
                x = self.basic_down[i][j](x)
            tmp_layers.append(x)
            x = self.downsample[i](x)

        #bottom
        for i in range(self.basic_num):
            x = self.bottom[i](x)

        #upsample
        for i in range(self.level_depth - 2, -1, -1):
            x = self.upsample[i](x)
            to_add = tmp_layers.pop()
            x = self.add([x, to_add])
            for j in range(self.basic_num):
                x = self.basic_up[i][j](x)

        x = self.output(x)
        x = self.linear(x)
        return x


class ResNet(torch.nn.Module):
    '''
        This class implements a net structure built with ResNet blocks. It takes a tuple of (coordinates, features)
        and passes it through the ResNet.

        ...

        Attributes
        ----------
        spatial_size : tuple
        The spatial size of the input layer. Size of the tuple is also the dimension.
        inplanes : int
            Number of planes we want after the initial SubmanifoldConvolution, that is, to begin downsampling.
        kernel : list
            Size of the kernels applied in each layer or block. Its length also indicates the level depth of the net.
        stride : int
            Applied stride in every layer or block.
        conv_kernel : int
            Kernel for the first convolutional layer.
        basic_num : int
            Number of times a basic residual block is passed in each level.
        nclasses : int, optional
            Number of output classes for predictions. The default is 2.
        dim : int, optional
            Number of dimensions of the input. The default is 3.
        start_planes : int, optional
            Number of planes that enter the ResNet. The default is 1.

        Methods
        -------
        forward(x)
            Passes the input through the ResNet
    '''
    def __init__(self, spatial_size, init_conv_nplanes, init_conv_kernel, kernel_sizes, stride_sizes, basic_num, nclasses = 2, dim = 3, start_planes = 1):
        '''
            Parameters
            ----------
            spatial_size : tuple
                The spatial size of the input layer. Size of the tuple is also the dimension.
            init_conv_nplaness : int
                Number of planes we want after the initial SubmanifoldConvolution, that is, to begin downsampling.
            init_conv_kernel : int
                Kernel for the first convolutional layer.
            kernel_sizes : list
                Size of the kernels applied in each layer or block. Its length also indicates the level depth of the net.
                Last element corresponds just to the kernel of the basic block in the bottom of the net.
            stride_sizes : list
                Sizes of the kernels applied in each layer or block. Its length also indicates the level depth of the net.
                Last element corresponds just to the kernel of the basic block in the bottom of the net.
            basic_num : int
                Number of times a basic residual block is passed in each level.
            nclasses : int, optional
                Number of output classes for predictions. The default is 2.
            dim : int, optional
                Number of dimensions of the input. The default is 3.
            start_planes : int, optional
                Number of planes that enter the net. The default is 1.
        '''
        torch.nn.Module.__init__(self)

        self.basic_num   = basic_num
        self.level_depth = len(kernel_sizes)

        self.inp     = scn.InputLayer(dim, spatial_size)
        self.convBN  = ConvBNBlock(start_planes, init_conv_nplanes, init_conv_kernel)
        inplanes     = init_conv_nplanes

        self.downsample = torch.nn.ModuleList([])
        self.basic      = torch.nn.ModuleList([torch.nn.ModuleList([]) for i in range(self.level_depth - 1)])
        self.bottom     = torch.nn.ModuleList([])

        for i in range(self.level_depth - 1):
            for j in range(basic_num):
                self.basic[i].append(ResidualBlock_basic(inplanes, kernel_sizes[i]))

            self.downsample.append(ResidualBlock_downsample(inplanes, kernel_sizes[i], stride_sizes[i]))

            inplanes = inplanes * 2

            for j in range(basic_num):
                self.bottom.append(ResidualBlock_basic(inplanes, kernel_sizes[-1]))

        self.max     = scn.MaxPooling(dim,max(calculate_output_dimension(spatial_size, kernel_sizes, stride_sizes)),1)
        self.sparse  = scn.SparseToDense(dim,inplanes)
        self.linear1 = torch.nn.Linear(inplanes, 32, nclasses)
        self.linear2 = torch.nn.Linear(32, 1, nclasses)

    def forward(self, x):
        '''
        Passes x through the net.

        Parameters
        ----------
        x : tuple
            It takes a tuple with (coord, features), where coord is a torch tensor with size [features_number, batch_size],
            and features is another torch tensor, with size [features_number, start_planes]

        Returns
        -------
        x : torch.Tensor
            A tensor with size [features_number, nclasses].

        '''
        x = self.inp(x)
        x = self.convBN(x)

        for i in range(self.level_depth - 1):
            for j in range(self.basic_num):
                x = self.basic[i][j](x)
            x = self.downsample[i](x)

        for i in range(self.basic_num):
            x = self.bottom[i](x)

        x = self.max(x)
        x = self.sparse(x)
        x = x.squeeze()
        x = self.linear1(x)
        x = self.linear2(x)

        return x
