import  socket
import time

class Tcp:
    def __init__(self, host_ip = "192.168.136.137", host_port = 8123):
        self._host_ip = host_ip
        self._host_port = host_port
        self._mylist = ['palm1', 'palm2', 'fist']
        self._obj = socket.socket()

    def connect(self):
        print("connecting host")
        self._obj.connect((self._host_ip, self._host_port))
        print("connected")

    def sendmessa(self, s):
        if s not in self._mylist:
            print("please input right message!")
        else:
            self._obj.send(s.encode())
            data = self._obj.recv(1024)
            print("got message-->", str(data,"utf8"))

    def disconnect(self):
        self._obj.send("quit".encode())

if __name__ == "__main__":
    mytcp = Tcp()
    mytcp.connect()
    mytcp.sendmessa("palm1")
    mytcp.sendmessa("palm2")
    mytcp.sendmessa("fist")
    mytcp.disconnect()