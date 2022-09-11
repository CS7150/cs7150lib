from collections import OrderedDict
from copy import deepcopy
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh
from torch.nn.functional import mse_loss, cross_entropy
from torch.optim import Adam, SGD
from baukit import PlotWidget, show, pbar, Widget, Range, Numberbox
from matplotlib import pyplot as plt

# This widget will let us see wthat is going on inside an MLP network as it trains.
# It assumes that the network operates as a binary classifier with +/1 outputs,
# classifying 2-dimensional inputs in the range [-3, 3].
class MLPHistoryWidget(Widget):
  def __init__(self, data=None, labels=None, history=None):
    super().__init__()
    self.history = history or []
    self.data = data or []
    self.labels = labels or []
    self.plot = PlotWidget(self.visualize_net, mosaic='012\n333', figsize=(11,6),
                 bbox_inches='tight', gridspec_kw={'hspace': 0.25, 'height_ratios': [2,1]})
    scrubber = Range(min=0, max=len(self.history), value=self.plot.prop('index'))
    numbox = Numberbox(value=self.plot.prop('index'))
    self.content = [
      [
        [show.style(alignContent='center'), 'Iteration'],
        numbox,
        show.style(flex=20), scrubber
      ],
      self.plot
    ]
    self.plot.on('click', self.plot_click)

  def _repr_html_(self):
    return show.html(self.content)

  def visualize_net(self, fig, index=0):
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.98)
    def endpoints(w, b, scale=10):
      if abs(w[1]) > abs(w[0]):
        x0 = torch.tensor([-scale, scale]).to(w.device)
        x1 = (-b - w[0] * x0) / w[1]
      else:
        x1 = torch.tensor([-scale, scale]).to(w.device)
        x0 = (-b - w[1] * x1) / w[0]
      return torch.stack([x0, x1], dim=1)

    ax1, ax2, ax3, ax4 = fig.axes
    ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
    if index >= len(self.history):
      return
    net, data = self.history[index]
    grid = torch.stack([
      torch.linspace(-3, 3, 200)[None, :].expand(200, 200),
      torch.linspace(3, -3, 200)[:, None].expand(200, 200),
    ])
    x, y = grid
    ax1.set_title('network output')
    score = net(grid.permute(1, 2, 0).reshape(-1, 2))
    ax1.imshow(score[:,0].reshape(200, 200).detach().cpu(),
            cmap='hot', extent=[-3,3,-3,3], vmin=-1, vmax=1)
    ax2.imshow(score[:,0].reshape(200, 200).detach().cpu(),
            cmap='hot', extent=[-3,3,-3,3], alpha=0.2, vmin=-1, vmax=1)

    ax2.set_title('training data')
    ax2.set_ylim(-3, 3)
    ax2.set_xlim(-3, 3)
    ax2.set_aspect(1.0)
    ax2.scatter([d[0] for d, l in zip(self.data, self.labels) if l > 0],
          [d[1] for d, l in zip(self.data, self.labels) if l > 0])
    ax2.scatter([d[0] for d, l in zip(self.data, self.labels) if l <= 0],
          [d[1] for d, l in zip(self.data, self.labels) if l <= 0])

    ax3.set_title('first layer folds')
    module = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][0]
    w = module.weight.detach().cpu()
    b = module.bias.detach().cpu()
    e = torch.stack([endpoints(wc, bc) for wc, bc in zip(w, b)])
    for ep in e:
      ax3.plot(ep[:,0], ep[:,1], '#00aa00', linewidth=0.75, alpha=0.33)
    ax3.set_ylim(-3, 3)
    ax3.set_xlim(-3, 3)
    ax3.set_aspect(1.0)

    ax4.set_title('training curve')
    ax4.set_xlabel('iteration')
    ax4.set_yscale('log')
    ax4.axvline(index, color='red', linewidth=0.5)
    for k, v in data.items():
      label = f'{k} = {v:.3g}'
      ax4.plot(range(len(self.history)), [h[1][k] for h in self.history],
              linewidth=0.5, label=label)
    ax4.legend()

  def plot_click(self, e):
    loc = self.plot.event_location(e)
    if loc.axis == 3:
      self.plot.index = max(0, min(len(self.history) - 1, int(loc.x + 0.5)))
      self.plot.redraw()
