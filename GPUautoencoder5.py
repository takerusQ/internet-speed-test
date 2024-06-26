C:\Users\kota\Desktop\suyama>python venv37nonGPUautoencoder5.py
Traceback (most recent call last):
  File "venv37nonGPUautoencoder5.py", line 130, in <module>
    z = encoder(x)
  File "C:\Users\kota\miniconda3\lib\site-packages\torch\nn\modules\module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "venv37nonGPUautoencoder5.py", line 84, in forward
    x = torch.relu(self.conv1(x))
  File "C:\Users\kota\miniconda3\lib\site-packages\torch\nn\modules\module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "C:\Users\kota\miniconda3\lib\site-packages\torch\nn\modules\conv.py", line 443, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "C:\Users\kota\miniconda3\lib\site-packages\torch\nn\modules\conv.py", line 439, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
RuntimeError: Expected 4-dimensional input for 4-dimensional weight [32, 1, 3, 3], but got 3-dimensional input of size [1, 512, 512] instead
