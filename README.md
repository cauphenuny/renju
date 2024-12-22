# renju

`cmake -Bbuild && cmake --build build --target main`

targets: 
- main: `bin/main`, game launcher
- botzone: `bin/botzone`, bot for botzone.org.cn
- test: `bin/test`, unit tests 
- generate_bindings: `train/lib/librenju.py`, generate python bindings for training neural network

[tutorial](tutorial.md)

---

includes a tiny web UI compatible with any bot using [botzone simplified I/O protocol](https://wiki.botzone.org.cn/index.php?title=Bot#.E4.BA.A4.E4.BA.92)

just launch the `web/server.py`, and then open `web/index.html`
