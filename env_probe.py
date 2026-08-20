import importlib.util

packages = ['torch', 'torchvision', 'tensorflow', 'onnx', 'onnxruntime', 'sklearn', 'pandas', 'PIL', 'cv2', 'timm', 'transformers']
for name in packages:
    spec = importlib.util.find_spec(name)
    print(f'{name}: {bool(spec)}')
    if spec:
        try:
            module = __import__(name)
            print('  version:', getattr(module, '__version__', 'unknown'))
        except Exception as exc:
            print('  import_error:', repr(exc))
