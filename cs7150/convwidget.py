import torch, os
from torchvision.models import alexnet
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from baukit import ImageFolderSet, show, renormalize, set_requires_grad
from torchvision.datasets.utils import download_and_extract_archive
from baukit import Widget, Img, TraceDict, get_module, Numberbox
from torch.nn import Conv2d

class ConvolutionWidget(Widget):
    def __init__(self, input, net=None, kernel_size=3, padding='same', depth=1):
        super().__init__()
        self.input = input
        if net is None:
            net = torch.nn.Sequential(*[
                Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding,
                     bias=False) for _ in range(depth)])
        self.net = net
        for p in self.net.parameters():
            p.requires_grad = False
        self.output = self.net(input)
        self.im_input = Img(renormalize.as_image(self.input))
        self.im_input.p = self.input
        self.im_input.on('click', self.handle_click)
        self.im_conv = []
        for p in self.net.parameters():
            img = Img(renormalize.as_image(p[0]))
            img.p = p
            img.on('click', self.handle_click)
            self.im_conv.append(img)
        self.im_output = Img(renormalize.as_image(self.output))
        im_style = show.style(width=150, imageRendering='pixelated')
        self.content = [[['input (click me)', [im_style, self.im_input]]] +
                        [[f'convolution {i+1} (click me)', [im_style, im]]
                         for i, im in enumerate(self.im_conv)] +
                        [['output', [im_style, self.im_output]]]]
    def widget_html(self):
        return show.html(self.content)
    def redraw(self):
        self.im_input.render(renormalize.as_image(self.input))
        for imc in self.im_conv:
            imc.render(renormalize.as_image(imc.p[0]))
        self.output = self.net(self.input)
        self.im_output.render(renormalize.as_image(self.output))
    def handle_click(self, e):
        x = int(e.value['x'])
        y = int(e.value['y'])
        e.target.p[..., y, x] = -torch.sign(e.target.p[..., y, x] - 1e-10)
        self.redraw()

class ConvolutionNetWidget(Widget):
    def __init__(self, input, net=None, kernel_size=3, padding='same', depth=1):
        super().__init__()
        self.input = input
        if net is None:
            net = torch.nn.Sequential(*[
                Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding,
                     bias=False) for _ in range(depth)])
        self.layers = [n for n, m in net.named_modules() if isinstance(m, Conv2d)]                
        self.net = net
        for p in self.net.parameters():
            p.requires_grad = False
        with TraceDict(self.net, self.layers, retain_input=True) as td:
            self.output = self.net(input)
        self.nb_chan = []
        self.im_input = []
        self.im_conv = []
        for n in self.layers:
            img = Img(renormalize.as_image(td[n].input[0][None]))
            img.n = n
            self.im_input.append(img)
            nb = Numberbox(0, size=2)
            nb.on('value', self.redraw)
            self.nb_chan.append(nb)
            p = get_module(net, n).weight
            img = Img(renormalize.as_image(p[0,0][None]))
            img.p = p
            img.n = n
            img.nbi = nb
            img.on('click', self.handle_click)
            self.im_conv.append(img)
        nb = Numberbox(0, size=2)
        nb.on('value', self.redraw)
        self.nb_chan.append(nb)
        self.im_output = Img(renormalize.as_image(self.output[0][None]))
        for imc, nbo in zip(self.im_conv, self.nb_chan[1:]):
            imc.nbo = nbo
        im_style = show.style(width=150, imageRendering='pixelated')
        self.content = [sum([[[['chan', nb], [im_style, imi]],
                              [f'conv {i+1}', [im_style, imc]]]
                         for i, (nb, imi, imc) in
							 enumerate(zip(self.nb_chan, self.im_input, self.im_conv))], []) +
                        [[['output', self.nb_chan[-1]], [im_style, self.im_output]]]]
    def widget_html(self):
        return show.html(self.content)
    def redraw(self):
        for nbi, nbo, imc in zip(self.nb_chan[:-1], self.nb_chan[1:], self.im_conv):
            imc.render(renormalize.as_image(
                imc.p[max(nbo.value, imc.p.shape[0]-1),
                      max(nbi.value, imc.p.shape[1]-1)][None]))
        with TraceDict(self.net, self.layers, retain_input=True) as td:
            self.output = self.net(self.input)
        for nbi, imi in zip(self.nb_chan[:-1], self.im_input):
            imi.render(renormalize.as_image(td[imi.n].input[
                max(nbi.value, td[imi.n].input.shape[0]-1)][None]))
        self.im_output.render(renormalize.as_image(self.output[
            max(self.nb_chan[-1].value, self.output.shape[0]-1)][None]))
    def redraw(self):
        for nbi, nbo, imc in zip(self.nb_chan[:-1], self.nb_chan[1:], self.im_conv):
            imc.render(renormalize.as_image(
                imc.p[nbo.value,nbi.value][None]))
        with TraceDict(self.net, self.layers, retain_input=True) as td:
            self.output = self.net(self.input)
        for nbi, imi in zip(self.nb_chan[:-1], self.im_input):
            imi.render(renormalize.as_image(td[imi.n].input[nbi.value][None]))
        self.im_output.render(renormalize.as_image(self.output[self.nb_chan[-1].value][None]))

    def handle_click(self, e):
        x = int(e.value['x'])
        y = int(e.value['y'])
        e.target.p[e.target.nbo.value, e.target.nbi.value, y, x] = (
            -torch.sign(e.target.p[e.target.nbo.value, e.target.nbi.value, y, x] - 1e-10))
        self.redraw()
