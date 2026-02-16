# ComfyUI-binuai-SceneSelect

파이썬 코드로 구현해 놓은거 ComfyUI 노드로 래핑하는 느낌으로 만들어봄

## how to use

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sjh354/ComfyUI-binuai-SceneSelect.git
cd ComfyUI-binuai-SceneSelect
python3 -m pip install -r requirements.txt

sudo apt-get update && sudo apt-get install -y ffmpeg
```

## TODO

지금은 그냥 바닥면 찾을 때 depth 모델 써서 하는데 이거 좀 별로인거 같아서 SAM써볼려고
