``` bash
# clone the original repo
git clone https://github.com/akmtn/pytorch-siggraph2017-inpainting

# download the legacy model, in lua-torch format
wget --continue -O completionnet_places2.t7 http://hi.cs.waseda.ac.jp/~iizuka/data/completionnet_places2.t7

# convert lua-torch model to py-torch model
# this produces "completionnet_places2.pt"
python convert_lua_pytorch.py 

# try out the converted py-torch model
python inpaint.py --gpu --postproc
eog out.png
```
