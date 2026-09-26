# The layers of nn/conv.vs and gpu/neural's audio kernels as torch.nn and
# torch compute them, at Kokoro's own shapes, strides and paddings (or
# smaller ones of the same kind): writes testdata/layers.safetensors,
# every case's input, weights and output, for cmd/test-layers.
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(7)
out = {}
def put(name, t): out[name] = t.detach().to(torch.float32).clone().contiguous()

def conv(name, cin, cout, t, k, stride=1, pad=0, dil=1, groups=1, bias=True):
    m = nn.Conv1d(cin, cout, k, stride, pad, dil, groups, bias=bias)
    x = torch.randn(1, cin, t)
    put(name + ".x", x[0]); put(name + ".w", m.weight)
    if bias: put(name + ".b", m.bias)
    put(name + ".y", m(x)[0])

conv("conv.k3p1", 16, 24, 37, 3, pad=1)
conv("conv.dil3", 8, 8, 40, 3, pad=3, dil=3)
conv("conv.s2", 1, 1, 41, 3, stride=2, pad=1)           # F0_conv, N_conv
conv("conv.noise", 22, 16, 90, 12, stride=6, pad=3)    # noise_convs.0
conv("conv.groups", 12, 6, 30, 5, pad=2, groups=3, bias=False)
conv("conv.1x1", 16, 8, 25, 1, bias=False)             # conv1x1

def convT(name, cin, cout, t, k, stride, pad, opad=0, groups=1):
    m = nn.ConvTranspose1d(cin, cout, k, stride, pad, opad, groups)
    x = torch.randn(1, cin, t)
    put(name + ".x", x[0]); put(name + ".w", m.weight); put(name + ".b", m.bias)
    put(name + ".y", m(x)[0])

convT("convT.up10", 16, 8, 9, 20, 10, 5)                  # generator ups.0
convT("convT.up6", 8, 4, 11, 12, 6, 3)                    # generator ups.1
convT("convT.pool", 10, 10, 13, 3, 2, 1, opad=1, groups=10)  # AdainResBlk1d pool

m = nn.utils.weight_norm(nn.Conv1d(16, 24, 3))
with torch.no_grad():
    m.weight_g.mul_(torch.rand_like(m.weight_g) + 0.5)
    m(torch.zeros(1, 16, 3))  # computes weight from g and v
put("wn.g", m.weight_g); put("wn.v", m.weight_v); put("wn.w", m.weight)

lstm = nn.LSTM(20, 12, 1, batch_first=True, bidirectional=True)
x = torch.randn(1, 15, 20)
for n, p in lstm.named_parameters(): put("lstm." + n, p)
put("lstm.x", x[0]); put("lstm.y", lstm(x)[0][0])

inorm = nn.InstanceNorm1d(12, affine=True)
with torch.no_grad():
    inorm.weight.copy_(torch.randn(12)); inorm.bias.copy_(torch.randn(12))
x = torch.randn(1, 12, 33) * 3 + 1
put("inorm.x", x[0]); put("inorm.w", inorm.weight); put("inorm.b", inorm.bias); put("inorm.y", inorm(x)[0])

fc = nn.Linear(16, 24)
s = torch.randn(1, 16)
h = fc(s).view(1, 24, 1); gamma, beta = torch.chunk(h, 2, dim=1)
put("adain.fc.w", fc.weight); put("adain.fc.b", fc.bias); put("adain.s", s[0]); put("adain.x", x[0])
put("adain.y", ((1 + gamma) * F.instance_norm(x, eps=1e-5) + beta)[0])

x = torch.randn(1, 12, 20) * 2
g, b = torch.randn(12), torch.randn(12)
put("chnorm.x", x[0]); put("chnorm.g", g); put("chnorm.b", b)
put("chnorm.y", F.layer_norm(x.transpose(1, -1), (12,), g, b, 1e-5).transpose(1, -1)[0])
h = fc(s); gamma, beta = torch.chunk(h.view(1, 24, 1), 2, dim=1)
y = F.layer_norm(x.transpose(1, -1), (12,), eps=1e-5).transpose(1, -1)
put("adaln.y", ((1 + gamma) * y + beta)[0])

x = torch.randn(1, 6, 50) * 2
a = torch.rand(1, 6, 1) * 1.5 + 0.5
put("snake.x", x[0]); put("snake.a", a.view(6)); put("snake.y", (x + (1 / a) * torch.sin(a * x) ** 2)[0])
x = torch.randn(200)
put("leaky.x", x); put("leaky.y", F.leaky_relu(x, 0.2))

x = torch.randn(1, 3, 17); put("up.n2.x", x[0]); put("up.n2.y", F.interpolate(x, scale_factor=2, mode="nearest")[0])
x = torch.rand(1, 1, 7) * 300; put("up.n300.x", x[0]); put("up.n300.y", nn.Upsample(scale_factor=300)(x)[0])
x = torch.rand(1, 9, 3000); put("up.ldown.x", x[0]); put("up.ldown.y", F.interpolate(x, scale_factor=1/300, mode="linear")[0])
x = torch.rand(1, 9, 10) * 50; put("up.lup.x", x[0]); put("up.lup.y", F.interpolate(x, scale_factor=300, mode="linear")[0])

x = torch.rand(9, 400) * 0.1; put("cumsum.x", x); put("cumsum.y", torch.cumsum(x, dim=1))
x = torch.randn(1, 4, 10); put("pad.x", x[0]); put("pad.y", nn.ReflectionPad1d((1, 0))(x)[0])

t = torch.arange(1000) / 24000
x = 0.3 * torch.sin(2 * math.pi * 440 * t) + 0.05 * torch.randn(1000)
win = torch.hann_window(20, periodic=True)
spec = torch.stft(x, 20, 5, 20, window=win, return_complex=True)
put("stft.x", x); put("stft.mag", torch.abs(spec)); put("stft.phase", torch.angle(spec))
a, b = torch.randn(11, 60) * 0.5, torch.randn(11, 60) * 2
put("istft.a", a); put("istft.b", b)
put("istft.y", torch.istft(torch.exp(a) * torch.exp(torch.sin(b) * 1j), 20, 5, 20, window=win))
put("istft.mag", torch.exp(a)); put("istft.phase", torch.sin(b))

save_file(out, os.path.join(os.path.dirname(__file__), "..", "layers.safetensors"))
print(len(out), "tensors")
